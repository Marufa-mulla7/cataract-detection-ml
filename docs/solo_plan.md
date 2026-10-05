# Solo Project Plan (4-6 weeks, working alone)

This replaces `team_plan.md` — same project, same codebase, just sequenced for one person
working on a Windows laptop (CPU) + Google Colab (GPU for training).

## Priority tiers (important if time runs short)

**Tier 1 — Must have (this is your working project, don't skip these):**
- Data pipeline + one solid baseline model (EfficientNet-B0, transfer learning)
- Proper evaluation (confusion matrix, precision/recall/F1, ROC-AUC)
- Grad-CAM explainability
- Streamlit demo app
- GitHub repo + README + report

**Tier 2 — Strong differentiators (do these if Tier 1 is done with time to spare):**
- Second architecture comparison (add ResNet50 or ViT — pick ONE, not both)
- MC Dropout uncertainty estimation
- External dataset validation (ODIR-5K)

**Tier 3 — Stretch (only if you're ahead of schedule):**
- Second comparison architecture (the one you didn't pick above)
- PDF report generation in the app
- Test-time augmentation

This way, if Week 4 arrives and you're behind, you still have a complete, working,
resume-worthy project — you just trim from Tier 3 down, never from Tier 1.

## Week-by-week (solo)

### Week 1 — Setup + data + first working pipeline
- Day 1-2: Install Python, VS Code, Git (see earlier setup steps). Create GitHub repo,
  push this folder structure.
- Day 3: Download the Kaggle cataract dataset, organize into `data/cataract/<class>/` folders.
  Run a quick EDA — how many images per class, view a few samples per class.
- Day 4-5: Get `notebooks/train_colab.py` running in Colab with **just EfficientNet-B0**.
  Goal: one full train → save checkpoint → download it. Don't worry about accuracy yet,
  just get the pipeline working end to end.
- Weekend: buffer / catch up if anything broke (very likely something will — that's normal)

### Week 2 — Make the baseline actually good
- Tune augmentation, try both frozen-backbone and fine-tuned versions
- Address class imbalance (the `compute_class_weights` function already handles this —
  just make sure you're actually using it in the training call)
- Once EfficientNet-B0 is solid, add **one** more architecture for comparison
  (ResNet50 is the safer/faster choice; ViT is the flashier one but may underperform
  on this small dataset — either way, the finding is worth reporting)

### Week 3 — Evaluation + explainability
- Run `src/evaluate.py` on your best model: confusion matrix, classification report, ROC-AUC
- Implement Grad-CAM (`src/gradcam.py`) and generate sample heatmaps for your report —
  pick 2-3 correctly classified and 1-2 misclassified examples, this makes for a great
  report figure ("here's what the model got right/wrong and why")
- If time allows: download the ODIR-5K subset and run external validation
  (`get_external_test_loader` + `evaluate_model`) — even a small external test set
  strengthens your generalization claim significantly

### Week 4 — Uncertainty + demo app
- Implement MC Dropout (`src/uncertainty.py`) — test it on a few images, confirm the
  uncertainty is higher on ambiguous/blurry images than clear ones (spot-check this manually,
  it's a good sanity check and a good thing to show in your viva)
- Build and test `app/streamlit_app.py` locally (`streamlit run app/streamlit_app.py`)
- Deploy to Streamlit Community Cloud (free) so you have a live demo link

### Week 5 — Report, polish, GitHub
- Write your project report (problem statement, dataset, methodology, architecture
  comparison results, evaluation metrics, Grad-CAM findings, limitations, future work)
- Polish README with screenshots, results table, live demo link
- Record a 2-3 minute demo video (screen recording of the app in action) — genuinely
  useful for both your submission and to link on your resume/LinkedIn

### Week 6 (buffer, if you have it) — Viva prep
- Rehearse explaining: why transfer learning, why you compared architectures, what MC
  Dropout tells you, why external validation matters, what Grad-CAM revealed
- Anticipate the "why didn't you use dataset X / model Y" question — you already have a
  good answer: time and compute constraints on a solo project, documented as future work

## Daily rhythm suggestion
Since you're solo, working in focused 2-3 hour blocks daily is more sustainable than
occasional long weekend pushes — ML debugging (data loader errors, shape mismatches,
Colab disconnects) eats time unpredictably, so steady daily progress protects your deadline
better than big irregular sessions.
