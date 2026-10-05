"""
Evaluation utilities. Produces the metrics that actually matter for a medical
classification project (not just accuracy): per-class precision/recall/F1,
confusion matrix, and ROC-AUC (one-vs-rest for the multi-class case).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)


@torch.no_grad()
def get_predictions(model, data_loader, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    all_labels, all_preds, all_probs = [], [], []
    for images, labels in data_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def evaluate_model(model, data_loader, class_names, device=None, save_prefix="results"):
    labels, preds, probs = get_predictions(model, data_loader, device)

    print("Classification report:")
    print(classification_report(labels, preds, target_names=class_names))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_confusion_matrix.png")
    plt.close()

    # one-vs-rest ROC-AUC per class
    try:
        auc_scores = roc_auc_score(labels, probs, multi_class="ovr", average=None)
        for name, score in zip(class_names, auc_scores):
            print(f"ROC-AUC ({name}): {score:.4f}")
    except ValueError as e:
        print(f"Could not compute ROC-AUC: {e}")

    return {"confusion_matrix": cm, "labels": labels, "preds": preds, "probs": probs}


def compare_architectures(results_dict, save_path="architecture_comparison.png"):
    """
    results_dict: {"mobilenet_v2": val_acc, "efficientnet_b0": val_acc, ...}
    Plots a simple bar chart comparing final validation accuracy across models —
    this is the plot you'll want in your report/PPT for the "we compared architectures" story.
    """
    names = list(results_dict.keys())
    accs = list(results_dict.values())

    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, accs, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][:len(names)])
    plt.ylabel("Validation Accuracy")
    plt.title("Architecture Comparison: EfficientNet-B0 vs ResNet50")
    plt.ylim(0, 1)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.01, f"{acc:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved comparison chart to {save_path}")
