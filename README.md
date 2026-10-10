# Eye Anomaly Detection - AI-Assisted Screening

This system classifies a fundus (retinal) photograph into five classes:
**Normal, Cataract, Glaucoma, AMD, Diabetic Retinopathy**.

It is an **academic screening aid, not a medical device**. It has not been clinically validated,
and every result must be reviewed by a qualified eye-care professional.

## What it does

- **Primary model: ResNet50**, with Monte Carlo Dropout (30 passes) for an uncertainty estimate.
- **Second opinion: EfficientNet-B0.** Disagreement between the two models flags the result for review.
- **Grad-CAM** heatmap showing which image regions drove the prediction.
- **Web app** (Flask + HTML): upload an image, see the prediction, probabilities, second opinion,
  heatmap, session history, and download a PDF screening report.
- **Batch analysis:** up to 30 images at once, with results exported as CSV.
- **Live dashboard:** statistics come from a local scan log (`scans.db`, created on first run). Only results are
  stored, never the uploaded images.
- **Review flag:** a result is flagged when the two models disagree or the MC Dropout spread is high.

## Results (final held-out test set, 583 real images)

| Model | Test accuracy | Macro F1 |
|---|---|---|
| ResNet50 (primary) | 95.20% (555/583) | 0.96 |
| EfficientNet-B0 (second opinion) | 96.23% (561/583) | 0.96 |

The two models agreed on 552/583 images (98.37% accuracy when they agree) and disagreed on 31.

## Dataset (4,600 images)

| Class | Images | Train / Val / Test | Notes |
|---|---|---|---|
| Normal | 900 | 630 / 135 / 135 | 847 real (eyedis 598, NL 249) + 53 augmented |
| Cataract | 900 | 630 / 135 / 135 | 876 real (eyedis 511, odir 279, cataract 86) + 24 augmented |
| Glaucoma | 900 | 630 / 135 / 135 | 864 real (eyedis 552, odir 236, Glaucoma 76) + 36 augmented |
| AMD | 900 | 844 / 28 / 28 | 188 real sources (RFMiD 100, ADAM 88); 712 augmented, training only |
| Diabetic Retinopathy | 1,000 | 700 / 150 / 150 | 971 unique real images + 29 augmented (training only) |

Validation and test sets contain **real images only**; augmented images exist only in training.
The split is fixed on disk (`data/cataract_final/{train,val,test}`) and is never re-split in code.
Datasets are not included in this repository.

## Data sources and attribution

Images come from several publicly available ophthalmic datasets; the attribution and access terms of each
source apply, and no licence is claimed here. AMD: RFMiD (Pachade et al., 2021, *Data* 6(2), 14,
doi:10.3390/data6020014) and ADAM. Diabetic Retinopathy: APTOS 2019 (Kaggle competition rules apply).
Normal/Cataract/Glaucoma: three shared source folders (`eyedis`, `odir`, and files prefixed `NL`/`cataract`/`Glaucoma`);
`eyedis` is the Kaggle Eye Diseases Classification dataset (gunavenkatdoddi/eye-diseases-classification), which the download script notes partly sources from ODIR-5K. Augmented images are generated from training
sources only and are not independent patient cases.

## Setup (Windows)

```bash
git clone https://github.com/Marufa-mulla7/cataract-detection-ml.git
cd cataract-detection-ml
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Model weights are **not** in the repository. Place them in `models/`:

- `models/best_resnet50_5class_final.pt`
- `models/best_efficientnet_b0_5class_final.pt`

Because of their size, the weights are not included; contact the author to obtain them.

Class order used everywhere: `normal, cataract, glaucoma, amd, diabetic_retinopathy`.

## Run the app

```bash
python backend_app.py
```

The browser opens automatically; otherwise open http://127.0.0.1:5000

## Evaluate and check

```bash
python run_eval.py                    # EfficientNet-B0 on test and validation sets
python pipeline_check.py 20           # sends test images through the real /predict endpoint
python compare_modes.py               # deterministic vs MC Dropout, agreement statistics
python uncertainty_analysis.py 2 val  # how well each review signal catches errors (use `test` only to report a chosen rule)
python run_cost_weighted_evaluation.py  # cost-weighted evaluation of ResNet50
python run_calibration_analysis.py      # reliability diagrams and ECE
python dataset_stats.py                 # real vs augmented images and source mix per class
```

These scripts need the dataset in `data/cataract_final` and the weights in `models/`.

## Training

Early experiments on the smaller dataset were trained on Google Colab (GPU). The final models on the
4,600-image dataset were trained on CPU (ResNet50 about 6 h 56 min, EfficientNet-B0 about 3 h), for up to
20 epochs with learning rate 1e-3 and early-stopping patience 5; the best validation checkpoint is kept.
Training augmentations: horizontal flip, rotation up to 15 degrees, brightness/contrast jitter 0.2.
Scripts: `run_train_resnet50.py` and `run_train.py` (EfficientNet-B0). Class weights are computed from the training split only
and passed to the training loop. Optimizer: Adam (lr 1e-3); loss: class-weighted cross-entropy; schedule: cosine annealing (T_max = 20).

## Project structure

```
backend_app.py             Flask server: /predict, /predict_batch, /generate_report, /api/stats
ocuvision-concept.html     Web interface
run_train.py               Train EfficientNet-B0 (final dataset)
run_train_resnet50.py      Train ResNet50 (final dataset)
run_eval.py                Evaluate EfficientNet-B0 (test and validation sets)
run_eval_resnet50.py       Evaluate ResNet50
run_cost_weighted_evaluation.py   Cost-weighted evaluation
run_calibration_analysis.py       Reliability diagrams and ECE
run_disagreement_gallery.py       Where the two models disagree, with Grad-CAM
pipeline_check.py          End-to-end test through the app
compare_modes.py           Deterministic vs MC Dropout, agreement statistics
uncertainty_analysis.py    Review-flag analysis
dataset_stats.py           Dataset composition from manifest.csv
build_final_dataset.py     Builds data/cataract_final (SHA-256 de-duplication, seed 42)
check_external_images.py   Runs a folder of external images through the running app
src/
  data_loader.py           Datasets, transforms (224x224, ImageNet normalisation), class names
  models.py                ResNet50 / EfficientNet-B0 via timm, with a Dropout classifier head
  train.py                 Training loop (Adam, weighted cross-entropy, cosine annealing)
  evaluate.py              Metrics: report, confusion matrix, ROC-AUC
  gradcam.py               Grad-CAM heatmaps
  uncertainty.py           MC Dropout
scripts/archive/           One-off dataset-building and earlier experiment scripts
app/streamlit_app.py       Earlier Streamlit prototype (older checkpoints; not part of the final system)
docs/images/               Screenshots
```

## Limitations

- Only 188 unique real AMD sources; 84% of AMD training images are augmented.
- AMD (RFMiD, ADAM) and DR (APTOS) each come only from sources that appear in no other class, so their near-perfect scores may partly reflect source or camera cues. 27 of the 28 ResNet50 errors are in Normal/Cataract/Glaucoma, which share the same sources.
- No external validation yet; one informal external DR image was predicted as Normal.
- Patient-level separation between splits is not confirmed.
- Confidence is only approximately calibrated (ECE about 0.031 for both models) and some errors carry high confidence. The review flag is raised when the models disagree or the MC Dropout std of the predicted class exceeds 0.02
  (chosen on the validation set, where it caught 13 of 15 ResNet50 errors and flagged 13.7% of images; on a test subset it caught 11 of 14 and flagged 13.0%).
- Fundus images only; not validated on other cameras, hospitals or populations.