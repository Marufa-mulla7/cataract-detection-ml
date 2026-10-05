# Cataract & Ocular Disease Detection using Deep Learning

An advanced eye-anomaly detection system that classifies fundus (retina) images into
**Normal / Cataract / Glaucoma / Other Retinal Disease**, built to go beyond a typical
"CNN classifier" student project.

## What makes this project different

Most student projects in this space stop at "train a CNN, report accuracy." This project adds:

1. **Multi-architecture benchmarking** — MobileNetV2, EfficientNet-B0, ResNet50, and a Vision
   Transformer (ViT) are all trained and compared on the same data using one shared codebase
   (via `timm`), instead of picking one architecture blindly.
2. **Uncertainty estimation (MC Dropout)** — the model doesn't just output a label, it outputs
   a confidence/uncertainty score, so it can flag "I'm not sure, please have a specialist review
   this" — a genuinely clinically-relevant feature almost no student project includes.
3. **External dataset validation** — the best model is additionally tested on a second, unseen
   dataset (ODIR-5K subset) to demonstrate real generalization, not just a good train/test split
   on one source.
4. **Explainability** — Grad-CAM heatmaps show *which region of the eye* drove the prediction.
5. **Deployed diagnostic demo** — a Streamlit web app that takes an uploaded eye image and
   produces a prediction, confidence score, Grad-CAM overlay, and a downloadable PDF report.

## Project structure

```
cataract-detection-ml/
├── data/                   # datasets go here (not committed to git — see .gitignore)
├── notebooks/              # Colab notebooks for training (GPU-heavy work)
├── src/
│   ├── data_loader.py      # dataset classes, transforms, train/val/test split
│   ├── models.py           # model builders (MobileNetV2 / EfficientNet / ResNet50 / ViT)
│   ├── train.py             # training loop, class-weighted loss, checkpointing
│   ├── evaluate.py          # metrics: accuracy, precision/recall/F1, ROC-AUC, confusion matrix
│   ├── gradcam.py           # Grad-CAM heatmap generation
│   └── uncertainty.py       # MC Dropout confidence estimation
├── app/
│   └── streamlit_app.py     # deployed demo: upload → predict → heatmap → PDF report
└── docs/
    └── team_plan.md         # 4-6 week timeline + role division for a team of 3-4
```

## Setup (Windows, local machine)

```bash
git clone <your-repo-url>
cd cataract-detection-ml
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Where training actually happens

Since local machines here are CPU-only, **all model training happens in Google Colab**
(free GPU). The notebooks in `notebooks/` are written to be uploaded to Colab directly.
Once a model is trained, download the `.pt` checkpoint and place it in a local `models/`
folder — that's what the Streamlit app and evaluation scripts use.

Workflow: **Colab (train) → download checkpoint → local repo (evaluate, demo, GitHub)**

## Datasets

- **Primary**: Kaggle Cataract Dataset — `kaggle.com/datasets/jr2ngb/cataractdataset`
  (601 images: normal, cataract, glaucoma, other retinal disease)
- **External validation**: ODIR-5K subset — `kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k`
  (used only for testing generalization, not training)

## Running each piece

```bash
# after training in Colab and placing checkpoint in models/best_model.pt
python src/evaluate.py --model models/best_model.pt --data data/test

# launch the demo app
streamlit run app/streamlit_app.py
```

## Team roles (team of 3-4)

See `docs/team_plan.md` for the full week-by-week breakdown and suggested role split.
