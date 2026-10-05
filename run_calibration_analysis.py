"""
Per-class calibration (reliability) analysis.

For each class, bucket the model's confidence into bins (e.g. 0.5-0.6,
0.6-0.7, ... 0.9-1.0) and check whether predictions in each bin are
actually correct as often as their stated confidence claims. A
well-calibrated model's "90% confident" predictions should be correct
about 90% of the time; a poorly-calibrated model's confidence doesn't
track its real accuracy.

This is the evidence behind the honest caveat already in the project:
AMD and Diabetic Retinopathy's near-perfect F1 scores may partly reflect
memorization/class imbalance rather than genuinely well-calibrated
confidence. This script checks that directly, per class, for both
EfficientNet-B0 and ResNet50.

Output: reliability_diagram_effnet.png, reliability_diagram_resnet50.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.models import build_model
from src.data_loader import get_dataloaders, CLASS_NAMES
from src.evaluate import get_predictions

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract")
EFFNET_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_5class.pt")
RESNET_PATH = os.path.join(PROJECT_ROOT, "models", "best_resnet50_5class.pt")

N_BINS = 10  # confidence buckets: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0


def load_model(model_key, checkpoint_path, device):
    model = build_model(model_key, num_classes=len(CLASS_NAMES), freeze_backbone=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model.to(device)


def compute_calibration_per_class(labels, preds, probs, class_idx, n_bins=N_BINS):
    """
    For a given class, treat it as a one-vs-rest problem:
    - confidence = model's predicted probability for this class, for every sample
    - correct = whether the true label actually is this class, restricted to
      samples where this class was the model's top prediction
    Returns bin centers, mean predicted confidence per bin, and actual accuracy per bin.
    """
    is_predicted_this_class = (preds == class_idx)
    if is_predicted_this_class.sum() == 0:
        return None

    confidences = probs[is_predicted_this_class, class_idx]
    correct = (labels[is_predicted_this_class] == class_idx).astype(np.float32)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_confidences, bin_accuracies, bin_counts = [], [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        if in_bin.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_confidences.append(confidences[in_bin].mean())
        bin_accuracies.append(correct[in_bin].mean())
        bin_counts.append(int(in_bin.sum()))

    return {
        "bin_centers": np.array(bin_centers),
        "bin_confidences": np.array(bin_confidences),
        "bin_accuracies": np.array(bin_accuracies),
        "bin_counts": np.array(bin_counts),
    }


def plot_reliability_diagram(labels, preds, probs, model_name, save_path):
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(4 * len(CLASS_NAMES), 4.2), sharey=True)

    for ax, class_idx, class_name in zip(axes, range(len(CLASS_NAMES)), CLASS_NAMES):
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")

        result = compute_calibration_per_class(labels, preds, probs, class_idx)
        if result is not None and len(result["bin_centers"]) > 0:
            ax.plot(
                result["bin_confidences"],
                result["bin_accuracies"],
                marker="o",
                color="#2E7DD1",
                label="Model",
            )
            for x, y, n in zip(result["bin_confidences"], result["bin_accuracies"], result["bin_counts"]):
                ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), fontsize=7, color="#555555")
        else:
            ax.text(0.5, 0.5, "No predictions\nfor this class", ha="center", va="center", fontsize=9)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(class_name.replace("_", " ").title(), fontsize=11)
        ax.set_xlabel("Predicted confidence")

    axes[0].set_ylabel("Actual accuracy")
    axes[0].legend(loc="upper left", fontsize=8)

    fig.suptitle(f"Per-Class Reliability Diagram - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved {save_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    _, val_loader, test_loader = get_dataloaders(DATA_ROOT)

    print("\nLoading EfficientNet-B0...")
    effnet = load_model("efficientnet_b0", EFFNET_PATH, device)
    print("Running EfficientNet-B0 on test set...")
    labels, effnet_preds, effnet_probs = get_predictions(effnet, test_loader, device)
    plot_reliability_diagram(
        labels, effnet_preds, effnet_probs, "EfficientNet-B0",
        os.path.join(PROJECT_ROOT, "reliability_diagram_effnet.png"),
    )

    print("\nLoading ResNet50...")
    resnet = load_model("resnet50", RESNET_PATH, device)
    print("Running ResNet50 on test set...")
    _, resnet_preds, resnet_probs = get_predictions(resnet, test_loader, device)
    plot_reliability_diagram(
        labels, resnet_preds, resnet_probs, "ResNet50",
        os.path.join(PROJECT_ROOT, "reliability_diagram_resnet50.png"),
    )

    print("\nDone. Both reliability diagrams saved to project root.")


if __name__ == "__main__":
    main()