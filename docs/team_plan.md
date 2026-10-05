# Team Plan (4-6 weeks, team of 3-4)

## Suggested role split

- **Person A — Data & Preprocessing Lead**
  Dataset download/organization, folder structure, augmentation pipeline, class imbalance
  analysis, EDA notebook with visualizations (class distribution, sample images per class).

- **Person B — Modeling Lead**
  Runs `notebooks/train_colab.py` in Colab, trains and tunes all four architectures
  (MobileNetV2, EfficientNet-B0, ResNet50, ViT-Small), manages checkpoints.

- **Person C — Evaluation & Explainability Lead**
  Owns `src/evaluate.py`, `src/gradcam.py`, `src/uncertainty.py`. Produces the confusion
  matrices, ROC curves, architecture comparison chart, Grad-CAM samples, and the external
  validation results. This person also drafts the "Results" section of the report.

- **Person D — Deployment & Documentation Lead** (if team of 4)
  Owns `app/streamlit_app.py`, deploys to Streamlit Community Cloud, writes the README,
  builds the presentation deck, records a demo video.
  *(If team of 3, split this role's tasks across A/B/C in Week 4.)*

## Week-by-week timeline

### Week 1 — Setup & Data
- Everyone: environment setup (Python, VS Code, Git, Colab access)
- Person A: download primary + external datasets, verify folder structure, run EDA
- Person B: get one baseline model (MobileNetV2) training end-to-end in Colab — goal is a
  working pipeline, not a good score yet
- Team: agree on GitHub workflow (branches, who commits what)

### Week 2 — Multi-architecture training
- Person B: train all 4 architectures using `train_colab.py`, log results
- Person A: refine augmentation based on Week 1 results, address class imbalance
- Person C: start building evaluation scripts, test them on the Week 1 baseline checkpoint

### Week 3 — Evaluation, explainability, uncertainty
- Person C: full evaluation suite on best model — confusion matrix, ROC-AUC, Grad-CAM samples,
  MC Dropout uncertainty testing, external dataset validation
- Person B: fine-tune the winning architecture further based on Person C's feedback
- Person D (or rotating): start Streamlit app skeleton

### Week 4 — Deployment & polish
- Person D: finish Streamlit app (Grad-CAM overlay, confidence display, PDF report), deploy
  to Streamlit Community Cloud
- Person A + C: finalize all charts/tables for the report
- Team: write project report, prepare slides

### Week 5-6 (if available) — Buffer & extension
- Polish GitHub README, record a short demo video
- If time allows: try one additional idea (e.g. ensembling the top 2 models, or testing
  on a third dataset) — good talking point for viva questions
- Rehearse viva / demo presentation as a team

## Viva/interview talking points this project gives you
- Why transfer learning instead of training from scratch (small dataset justification)
- Why you compared CNNs vs a Vision Transformer, and what the result implied
- What MC Dropout is doing and why uncertainty matters in a medical context
- Why external validation matters (train/test split on one dataset doesn't prove generalization)
- How Grad-CAM works and what it revealed about the model's focus areas
- Trade-offs made given time/compute constraints (CPU-only local, Colab for training)
