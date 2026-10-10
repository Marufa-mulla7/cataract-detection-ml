"""
Final ResNet50 vs EfficientNet-B0 disagreement analysis.

Uses the final leakage-free 583-image test set and the final model
checkpoints. For every disagreement, saves a 3-panel Grad-CAM image:
original image, EfficientNet-B0 Grad-CAM, and ResNet50 Grad-CAM.
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

# FINAL DATASET
DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "cataract_final"
)

# FINAL EFFICIENTNET-B0 CHECKPOINT
EFFNET_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_efficientnet_b0_5class_final.pt"
)

# FINAL RESNET50 CHECKPOINT
RESNET_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_resnet50_5class_final.pt"
)

# Separate output folder so old gallery results are not mixed
GALLERY_DIR = os.path.join(
    PROJECT_ROOT,
    "disagreement_gallery_final"
)


def load_model(model_key, checkpoint_path, device):
    model = build_model(
        model_key,
        num_classes=len(CLASS_NAMES),
        freeze_backbone=False
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(GALLERY_DIR, exist_ok=True)

    # Load FINAL dataset
    _, val_loader, test_loader = get_dataloaders(DATA_ROOT)

    # Map test-set order back to image paths
    # test_loader is not shuffled, so its order equals dataset.samples order
    file_paths = [p for p, _ in test_loader.dataset.samples]

    print("\nLoading final EfficientNet-B0...")
    effnet = load_model(
        "efficientnet_b0",
        EFFNET_PATH,
        device
    )

    print("Loading final ResNet50...")
    resnet = load_model(
        "resnet50",
        RESNET_PATH,
        device
    )

    print("\nRunning EfficientNet-B0 on FINAL test set...")
    labels, effnet_preds, effnet_probs = get_predictions(
        effnet,
        test_loader,
        device
    )

    print("Running ResNet50 on FINAL test set...")
    _, resnet_preds, resnet_probs = get_predictions(
        resnet,
        test_loader,
        device
    )

    # ---------------------------------------------------------
    # FINAL AGREEMENT / DISAGREEMENT ANALYSIS
    # ---------------------------------------------------------

    disagree_mask = effnet_preds != resnet_preds
    disagree_indices = np.where(disagree_mask)[0]

    total = len(labels)
    disagreements = len(disagree_indices)
    agreements = total - disagreements

    agreement_rate = agreements / total * 100
    disagreement_rate = disagreements / total * 100

    print("\n" + "=" * 60)
    print("FINAL MODEL AGREEMENT ANALYSIS")
    print("=" * 60)

    print(f"Total test images       : {total}")
    print(f"Models agreed           : {agreements}")
    print(f"Models disagreed        : {disagreements}")
    print(f"Agreement rate          : {agreement_rate:.2f}%")
    print(f"Disagreement rate       : {disagreement_rate:.2f}%")

    # ---------------------------------------------------------
    # WHICH MODEL WAS CORRECT ON DISAGREEMENTS?
    # ---------------------------------------------------------

    effnet_correct = 0
    resnet_correct = 0

    for idx in disagree_indices:
        true_label = labels[idx]

        if effnet_preds[idx] == true_label:
            effnet_correct += 1

        if resnet_preds[idx] == true_label:
            resnet_correct += 1

    print("\nOn disagreement cases:")

    if disagreements > 0:
        print(
            f"EfficientNet-B0 correct : "
            f"{effnet_correct}/{disagreements} "
            f"({effnet_correct / disagreements * 100:.2f}%)"
        )

        print(
            f"ResNet50 correct        : "
            f"{resnet_correct}/{disagreements} "
            f"({resnet_correct / disagreements * 100:.2f}%)"
        )

    # ---------------------------------------------------------
    # BUILD FINAL GRAD-CAM GALLERY
    # ---------------------------------------------------------

    print(
        f"\nFound {disagreements} disagreement cases."
    )

    if disagreements == 0:
        print("No disagreement gallery is needed.")
        return

    eval_transform = get_transforms(train=False)

    for idx in disagree_indices:

        path = file_paths[idx]

        true_label = CLASS_NAMES[int(labels[idx])]

        e_pred_idx = int(effnet_preds[idx])
        r_pred_idx = int(resnet_preds[idx])

        e_pred = CLASS_NAMES[e_pred_idx]
        r_pred = CLASS_NAMES[r_pred_idx]

        e_conf = float(
            effnet_probs[idx][e_pred_idx]
        )

        r_conf = float(
            resnet_probs[idx][r_pred_idx]
        )

        pil_image = Image.open(path).convert("RGB")

        input_tensor = eval_transform(
            pil_image
        ).unsqueeze(0)

        display_img = preprocess_for_display(
            pil_image
        )

        # EfficientNet Grad-CAM
        effnet_cam = generate_gradcam(
            effnet,
            "efficientnet_b0",
            input_tensor,
            display_img,
            target_class=e_pred_idx,
            device=device
        )

        # ResNet50 Grad-CAM
        resnet_cam = generate_gradcam(
            resnet,
            "resnet50",
            input_tensor,
            display_img,
            target_class=r_pred_idx,
            device=device
        )

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(13, 4.5)
        )

        axes[0].imshow(display_img)
        axes[0].set_title(
            f"Original\nTrue: {true_label}"
        )
        axes[0].axis("off")

        axes[1].imshow(effnet_cam)
        axes[1].set_title(
            f"EfficientNet-B0\n"
            f"Pred: {e_pred} ({e_conf:.2f})"
        )
        axes[1].axis("off")

        axes[2].imshow(resnet_cam)
        axes[2].set_title(
            f"ResNet50\n"
            f"Pred: {r_pred} ({r_conf:.2f})"
        )
        axes[2].axis("off")

        fig.suptitle(
            f"Final disagreement case - test image #{idx}",
            fontsize=13,
            fontweight="bold"
        )

        plt.tight_layout()

        output_path = os.path.join(
            GALLERY_DIR,
            f"disagreement_{idx:03d}_true-{true_label}.png"
        )

        plt.savefig(
            output_path,
            dpi=120
        )

        plt.close(fig)

        print(f"  Saved {output_path}")

    print("\n" + "=" * 60)
    print("FINAL DISAGREEMENT ANALYSIS COMPLETE")
    print("=" * 60)
    print(
        f"Gallery: {GALLERY_DIR}"
    )


if __name__ == "__main__":
    main()