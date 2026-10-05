"""
Disagreement gallery with Grad-CAM.

For every test image where EfficientNet-B0 and ResNet50 disagree on the
predicted class, saves a 3-panel image: original photo, EfficientNet-B0's
Grad-CAM (for its own predicted class), and ResNet50's Grad-CAM (for its
own predicted class). This lets you visually inspect *why* the two models
reached different conclusions on the same image, rather than only knowing
that they disagreed.

Output: one PNG per disagreement case, saved to disagreement_gallery/.
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
EFFNET_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_5class.pt")
RESNET_PATH = os.path.join(PROJECT_ROOT, "models", "best_resnet50_5class.pt")
GALLERY_DIR = os.path.join(PROJECT_ROOT, "disagreement_gallery")


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
    # test_loader.dataset is a TransformedSubset (see the fixed data_loader.py),
    # which holds .samples and .indices directly - no nested .dataset needed.
    test_subset = test_loader.dataset
    file_paths = [test_subset.samples[i][0] for i in test_subset.indices]

    print("Loading EfficientNet-B0...")
    effnet = load_model("efficientnet_b0", EFFNET_PATH, device)
    print("Loading ResNet50...")
    resnet = load_model("resnet50", RESNET_PATH, device)

    print("\nRunning EfficientNet-B0 on test set...")
    labels, effnet_preds, effnet_probs = get_predictions(effnet, test_loader, device)
    print("Running ResNet50 on test set...")
    _, resnet_preds, resnet_probs = get_predictions(resnet, test_loader, device)

    disagree_mask = effnet_preds != resnet_preds
    disagree_indices = np.where(disagree_mask)[0]

    print(f"\nFound {len(disagree_indices)} disagreement cases. Building Grad-CAM gallery...\n")

    eval_transform = get_transforms(train=False)

    for idx in disagree_indices:
        path = file_paths[idx]
        true_label = CLASS_NAMES[labels[idx]]
        e_pred_idx = int(effnet_preds[idx])
        r_pred_idx = int(resnet_preds[idx])
        e_pred = CLASS_NAMES[e_pred_idx]
        r_pred = CLASS_NAMES[r_pred_idx]
        e_conf = float(effnet_probs[idx][e_pred_idx])
        r_conf = float(resnet_probs[idx][r_pred_idx])

        pil_image = Image.open(path).convert("RGB")
        input_tensor = eval_transform(pil_image).unsqueeze(0)
        display_img = preprocess_for_display(pil_image)

        effnet_cam = generate_gradcam(
            effnet, "efficientnet_b0", input_tensor, display_img,
            target_class=e_pred_idx, device=device,
        )
        resnet_cam = generate_gradcam(
            resnet, "resnet50", input_tensor, display_img,
            target_class=r_pred_idx, device=device,
        )

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

        axes[0].imshow(display_img)
        axes[0].set_title(f"Original\nTrue: {true_label}")
        axes[0].axis("off")

        axes[1].imshow(effnet_cam)
        axes[1].set_title(f"EfficientNet-B0\nPred: {e_pred} ({e_conf:.2f})")
        axes[1].axis("off")

        axes[2].imshow(resnet_cam)
        axes[2].set_title(f"ResNet50\nPred: {r_pred} ({r_conf:.2f})")
        axes[2].axis("off")

        fig.suptitle(f"Disagreement case - test image #{idx}", fontsize=13, fontweight="bold")
        plt.tight_layout()

        out_path = os.path.join(GALLERY_DIR, f"disagreement_{idx:03d}_true-{true_label}.png")
        plt.savefig(out_path, dpi=120)
        plt.close(fig)

        print(f"  Saved {out_path}")

    print(f"\nDone. {len(disagree_indices)} images saved to {GALLERY_DIR}/")


if __name__ == "__main__":
    main()