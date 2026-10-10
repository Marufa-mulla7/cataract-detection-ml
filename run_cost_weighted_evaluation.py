"""
Cost-weighted evaluation.

Plain accuracy treats every mistake equally. In a real screening context,
that's wrong: missing an actual disease (calling a diseased eye "Normal")
is far more harmful than a false alarm (calling a Normal eye "diseased" -
the patient just gets an unnecessary follow-up). Confusing two different
diseases with each other is a real misdiagnosis, but the patient at least
gets flagged for professional review.

This script re-scores the existing test-set predictions (no retraining
needed) using an asymmetric cost matrix reflecting these clinical
priorities, and reports a cost-weighted score alongside plain accuracy.

Cost matrix (true row -> predicted column):
  - Correct prediction:                      cost 0
  - Normal misclassified as a disease:        cost 1  (false alarm - low harm)
  - A disease misclassified as another disease: cost 5  (misdiagnosis, still flagged)
  - A disease misclassified as Normal:        cost 10 (MISSED DISEASE - worst outcome)

These specific weights are a reasonable, commonly-used illustrative choice
(10x worse to miss a disease than to raise a false alarm) - not derived
from a formal clinical cost-benefit study. The point is the asymmetry,
not the exact numbers; feel free to adjust COST_MATRIX below and re-run
if you want to argue for different weights in your report.
"""

import os
import numpy as np
import torch

from src.models import build_model
from src.data_loader import get_dataloaders, CLASS_NAMES
from src.evaluate import get_predictions

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract_final")

# Primary model (ResNet50, current best performer post-augmentation-fix)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_resnet50_5class_final.pt")
MODEL_KEY = "resnet50"

NORMAL_IDX = CLASS_NAMES.index("normal")


def build_cost_matrix():
    """
    Returns a (n_classes, n_classes) cost matrix, cost[true][pred].
    """
    n = len(CLASS_NAMES)
    cost = np.zeros((n, n))
    for true_idx in range(n):
        for pred_idx in range(n):
            if true_idx == pred_idx:
                cost[true_idx][pred_idx] = 0
            elif true_idx == NORMAL_IDX and pred_idx != NORMAL_IDX:
                cost[true_idx][pred_idx] = 1   # false alarm
            elif true_idx != NORMAL_IDX and pred_idx == NORMAL_IDX:
                cost[true_idx][pred_idx] = 10  # missed disease - worst case
            else:
                cost[true_idx][pred_idx] = 5   # disease confused with another disease
    return cost


def load_model(model_key, checkpoint_path, device):
    model = build_model(model_key, num_classes=len(CLASS_NAMES), freeze_backbone=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model.to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _, val_loader, test_loader = get_dataloaders(DATA_ROOT)

    print("Loading primary model (ResNet50)...")
    model = load_model(MODEL_KEY, MODEL_PATH, device)

    print("Running on test set...")
    labels, preds, probs = get_predictions(model, test_loader, device)

    cost_matrix = build_cost_matrix()
    n_total = len(labels)

    total_cost = 0.0
    n_missed_disease = 0
    n_false_alarm = 0
    n_disease_confusion = 0
    n_correct = 0

    for true_idx, pred_idx in zip(labels, preds):
        c = cost_matrix[true_idx][pred_idx]
        total_cost += c
        if true_idx == pred_idx:
            n_correct += 1
        elif true_idx == NORMAL_IDX:
            n_false_alarm += 1
        elif pred_idx == NORMAL_IDX:
            n_missed_disease += 1
        else:
            n_disease_confusion += 1

    plain_accuracy = n_correct / n_total
    avg_cost_per_sample = total_cost / n_total
    # Cost-weighted "score": 1.0 minus normalized average cost (worst possible
    # average cost is 10, if every single prediction missed a disease).
    cost_weighted_score = 1.0 - (avg_cost_per_sample / 10.0)

    print(f"\n{'=' * 60}")
    print("COST-WEIGHTED EVALUATION")
    print(f"{'=' * 60}")
    print(f"Total test images: {n_total}")
    print(f"Correct predictions: {n_correct} ({plain_accuracy * 100:.1f}%)")
    print(f"\nError breakdown:")
    print(f"  Missed disease (disease -> Normal):        {n_missed_disease}  [cost 10 each - WORST]")
    print(f"  Disease confused with another disease:      {n_disease_confusion}  [cost 5 each]")
    print(f"  False alarm (Normal -> disease):            {n_false_alarm}  [cost 1 each]")
    print(f"\nTotal cost: {total_cost:.0f}")
    print(f"Average cost per sample: {avg_cost_per_sample:.3f}")
    print(f"Cost-weighted score (1.0 = perfect, 0.0 = worst possible): {cost_weighted_score:.4f}")
    print(f"\nFor comparison, plain accuracy: {plain_accuracy:.4f}")
    print(
        f"\nInterpretation: plain accuracy treats a missed disease and a false "
        f"alarm as equally bad. This cost-weighted view shows the {n_missed_disease} "
        f"missed-disease case(s) specifically, which matter far more clinically "
        f"than the {n_false_alarm} false alarm(s), even though a plain accuracy "
        f"score would count them identically."
    )


if __name__ == "__main__":
    main()