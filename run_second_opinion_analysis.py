"""
Disagreement-based second-opinion analysis.

Runs both trained models (EfficientNet-B0 and ResNet50) on the same test set
and flags cases where their top predictions disagree. The idea: when two
independently-trained architectures disagree on an image, that disagreement
itself is a useful signal for "this case may need manual review" - similar
in spirit to how a second-opinion works in clinical practice.

This also reports whether accuracy is meaningfully lower on the disagreement
subset than on the agreement subset, which is the evidence that disagreement
is actually a meaningful signal and not just noise.
"""

import os
import numpy as np
import torch

from src.models import build_model
from src.data_loader import get_dataloaders, CLASS_NAMES
from src.evaluate import get_predictions

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract")
EFFNET_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_5class.pt")
RESNET_PATH = os.path.join(PROJECT_ROOT, "models", "best_resnet50_5class.pt")


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

    print("Loading EfficientNet-B0...")
    effnet = load_model("efficientnet_b0", EFFNET_PATH, device)
    print("Loading ResNet50...")
    resnet = load_model("resnet50", RESNET_PATH, device)

    print("\nRunning EfficientNet-B0 on test set...")
    labels, effnet_preds, effnet_probs = get_predictions(effnet, test_loader, device)
    print("Running ResNet50 on test set...")
    _, resnet_preds, resnet_probs = get_predictions(resnet, test_loader, device)

    agree_mask = effnet_preds == resnet_preds
    disagree_mask = ~agree_mask

    n_total = len(labels)
    n_agree = int(agree_mask.sum())
    n_disagree = int(disagree_mask.sum())

    print(f"\n{'=' * 60}")
    print("SECOND-OPINION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total test images: {n_total}")
    print(f"Models agree:    {n_agree} ({n_agree / n_total * 100:.1f}%)")
    print(f"Models disagree: {n_disagree} ({n_disagree / n_total * 100:.1f}%)")

    if n_agree > 0:
        agree_acc = (effnet_preds[agree_mask] == labels[agree_mask]).mean()
        print(f"\nAccuracy when models AGREE: {agree_acc:.4f}")
    else:
        print("\nNo agreement cases found.")

    if n_disagree > 0:
        disagree_effnet_acc = (effnet_preds[disagree_mask] == labels[disagree_mask]).mean()
        disagree_resnet_acc = (resnet_preds[disagree_mask] == labels[disagree_mask]).mean()
        print(f"When models DISAGREE - EfficientNet-B0 accuracy: {disagree_effnet_acc:.4f}")
        print(f"When models DISAGREE - ResNet50 accuracy:        {disagree_resnet_acc:.4f}")

        print(f"\n{'=' * 60}")
        print("DISAGREEMENT CASES (flagged for manual review)")
        print(f"{'=' * 60}")
        disagree_indices = np.where(disagree_mask)[0]
        for idx in disagree_indices:
            true_label = CLASS_NAMES[labels[idx]]
            e_pred = CLASS_NAMES[effnet_preds[idx]]
            e_conf = effnet_probs[idx][effnet_preds[idx]]
            r_pred = CLASS_NAMES[resnet_preds[idx]]
            r_conf = resnet_probs[idx][resnet_preds[idx]]
            print(
                f"  [test image #{idx}] true={true_label} | "
                f"EfficientNet-B0={e_pred} ({e_conf:.2f}) | "
                f"ResNet50={r_pred} ({r_conf:.2f})"
            )
    else:
        print("\nNo disagreement cases found - both models agreed on every test image.")


if __name__ == "__main__":
    main()