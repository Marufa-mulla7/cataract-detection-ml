"""
Failure case gallery.

For every test image where the primary model predicted the wrong class,
saves a 2-panel image: the original photo and its Grad-CAM heatmap for the
model's (incorrect) predicted class. This lets you visually inspect what
the model was actually looking at when it got something wrong, rather than
only knowing the error rate per class.

NOTE: after the data-augmentation bug fix and retrain, ResNet50 outperformed
EfficientNet-B0 on the real test set and is now the primary/deployed model -
this script has been updated accordingly (was originally EfficientNet-B0
only). Internal variable names below still say "EFFNET"/"effnet" for
historical reasons but now load ResNet50's checkpoint.

Output: one PNG per misclassified test image, saved to failure_gallery/.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.models import build_model
from src.data_loader import get_dataloaders, get_transforms, CLASS_NAMES
from src.evaluate import get_predictions
from src.gradcam import generate_gradcam, preprocess_for_display

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract")
EFFNET_PATH = os.path.join(PROJECT_ROOT, "models", "best_resnet50_5class.pt")
MODEL_KEY = "resnet50"
GALLERY_DIR = os.path.join(PROJECT_ROOT, "failure_gallery")


def load_model(model_key, checkpoint_path, device):
    model = build_model(model_key, num_classes=len(CLASS_NAMES), freeze_backbone=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model.to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(GALLERY_DIR, exist_ok=True)

    _, val_loader, test_loader = get_dataloaders(DATA_ROOT)

    # Map test-set batch order back to real file paths.
    test_subset = test_loader.dataset
    file_paths = [test_subset.samples[i][0] for i in test_subset.indices]

    print("Loading EfficientNet-B0...")
    model = load_model(MODEL_KEY, EFFNET_PATH, device)

    print("Running EfficientNet-B0 on test set...")
    labels, preds, probs = get_predictions(model, test_loader, device)

    misclassified_mask = preds != labels
    misclassified_indices = np.where(misclassified_mask)[0]

    print(f"\nFound {len(misclassified_indices)} misclassified test images "
          f"out of {len(labels)} total.")

    # Quick per-class breakdown so it's obvious where the failures cluster.
    print("\nMisclassifications by true class:")
    for class_idx, class_name in enumerate(CLASS_NAMES):
        n_class_total = int((labels == class_idx).sum())
        n_class_wrong = int(((labels == class_idx) & misclassified_mask).sum())
        print(f"  {class_name}: {n_class_wrong} / {n_class_total} wrong")

    print(f"\nBuilding Grad-CAM failure gallery...\n")

    eval_transform = get_transforms(train=False)

    for idx in misclassified_indices:
        path = file_paths[idx]
        true_label = CLASS_NAMES[labels[idx]]
        pred_idx = int(preds[idx])
        pred_label = CLASS_NAMES[pred_idx]
        confidence = float(probs[idx][pred_idx])

        pil_image = Image.open(path).convert("RGB")
        input_tensor = eval_transform(pil_image).unsqueeze(0)
        display_img = preprocess_for_display(pil_image)

        cam = generate_gradcam(
            model, MODEL_KEY, input_tensor, display_img,
            target_class=pred_idx, device=device,
        )

        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

        axes[0].imshow(display_img)
        axes[0].set_title(f"Original\nTrue: {true_label}")
        axes[0].axis("off")

        axes[1].imshow(cam)
        axes[1].set_title(f"Grad-CAM\nPredicted: {pred_label} ({confidence:.2f})")
        axes[1].axis("off")

        fig.suptitle(
            f"Misclassification - test image #{idx}",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()

        out_path = os.path.join(
            GALLERY_DIR,
            f"failure_{idx:03d}_true-{true_label}_pred-{pred_label}.png",
        )
        plt.savefig(out_path, dpi=120)
        plt.close(fig)

        print(f"  Saved {out_path}")

    print(f"\nDone. {len(misclassified_indices)} images saved to {GALLERY_DIR}/")


if __name__ == "__main__":
    main()