"""
Final ResNet50 vs EfficientNet-B0 disagreement analysis with Grad-CAM.

Uses:
    data/cataract_final
    best_efficientnet_b0_5class_final.pt
    best_resnet50_5class_final.pt

Evaluates both models on the same final 583-image test set.

Reports:
    - Total test images
    - Agreement count and rate
    - Disagreement count and rate
    - Which model was correct on disagreements

Also creates a Grad-CAM gallery for disagreement cases.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.models import build_model
from src.data_loader import (
    get_dataloaders,
    get_transforms,
    CLASS_NAMES
)
from src.evaluate import get_predictions
from src.gradcam import (
    generate_gradcam,
    preprocess_for_display
)


# ============================================================
# PROJECT PATHS
# ============================================================

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

# FINAL OUTPUT DIRECTORY
GALLERY_DIR = os.path.join(
    PROJECT_ROOT,
    "disagreement_gallery_final"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_key, checkpoint_path, device):

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{checkpoint_path}"
        )

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


# ============================================================
# MAIN
# ============================================================

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Create gallery directory
    # --------------------------------------------------------

    os.makedirs(
        GALLERY_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load FINAL dataset
    # --------------------------------------------------------

    print("\nLoading final dataset...")

    _, val_loader, test_loader = get_dataloaders(
        DATA_ROOT
    )

    # --------------------------------------------------------
    # Get test image paths
    #
    # The current EyeDataset stores samples directly.
    # There is no .indices attribute.
    # --------------------------------------------------------

    test_dataset = test_loader.dataset

    file_paths = [
        sample[0]
        for sample in test_dataset.samples
    ]

    # Safety check
    if len(file_paths) != len(test_dataset):
        raise RuntimeError(
            "Number of image paths does not match "
            "the test dataset length."
        )

    # --------------------------------------------------------
    # Load final EfficientNet-B0
    # --------------------------------------------------------

    print("\nLoading final EfficientNet-B0...")

    effnet = load_model(
        "efficientnet_b0",
        EFFNET_PATH,
        device
    )

    # --------------------------------------------------------
    # Load final ResNet50
    # --------------------------------------------------------

    print("Loading final ResNet50...")

    resnet = load_model(
        "resnet50",
        RESNET_PATH,
        device
    )

    # --------------------------------------------------------
    # EfficientNet predictions
    # --------------------------------------------------------

    print(
        "\nRunning EfficientNet-B0 "
        "on FINAL test set..."
    )

    labels, effnet_preds, effnet_probs = get_predictions(
        effnet,
        test_loader,
        device
    )

    # --------------------------------------------------------
    # ResNet50 predictions
    # --------------------------------------------------------

    print(
        "Running ResNet50 "
        "on FINAL test set..."
    )

    resnet_labels, resnet_preds, resnet_probs = get_predictions(
        resnet,
        test_loader,
        device
    )

    # --------------------------------------------------------
    # Verify labels match
    # --------------------------------------------------------

    if not np.array_equal(labels, resnet_labels):
        raise RuntimeError(
            "The two model evaluations produced "
            "different ground-truth label orders."
        )

    # ========================================================
    # AGREEMENT / DISAGREEMENT ANALYSIS
    # ========================================================

    disagree_mask = (
        effnet_preds != resnet_preds
    )

    disagree_indices = np.where(
        disagree_mask
    )[0]

    total = len(labels)

    disagreements = len(
        disagree_indices
    )

    agreements = total - disagreements

    agreement_rate = (
        agreements / total * 100
    )

    disagreement_rate = (
        disagreements / total * 100
    )

    # ========================================================
    # PRINT AGREEMENT RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL MODEL AGREEMENT ANALYSIS")
    print("=" * 60)

    print(
        f"Total test images       : {total}"
    )

    print(
        f"Models agreed           : {agreements}"
    )

    print(
        f"Models disagreed        : {disagreements}"
    )

    print(
        f"Agreement rate          : "
        f"{agreement_rate:.2f}%"
    )

    print(
        f"Disagreement rate       : "
        f"{disagreement_rate:.2f}%"
    )

    # ========================================================
    # DETERMINE WHICH MODEL WAS CORRECT
    # ========================================================

    effnet_correct = 0
    resnet_correct = 0

    # Cases where both models are wrong or both predictions
    # happen to differ from the true class are also possible.

    both_wrong = 0

    for idx in disagree_indices:

        true_label = labels[idx]

        effnet_is_correct = (
            effnet_preds[idx] == true_label
        )

        resnet_is_correct = (
            resnet_preds[idx] == true_label
        )

        if effnet_is_correct:
            effnet_correct += 1

        if resnet_is_correct:
            resnet_correct += 1

        if (
            not effnet_is_correct
            and not resnet_is_correct
        ):
            both_wrong += 1

    print("\nOn disagreement cases:")

    if disagreements > 0:

        effnet_disagreement_accuracy = (
            effnet_correct
            / disagreements
            * 100
        )

        resnet_disagreement_accuracy = (
            resnet_correct
            / disagreements
            * 100
        )

        both_wrong_rate = (
            both_wrong
            / disagreements
            * 100
        )

        print(
            f"EfficientNet-B0 correct : "
            f"{effnet_correct}/{disagreements} "
            f"({effnet_disagreement_accuracy:.2f}%)"
        )

        print(
            f"ResNet50 correct        : "
            f"{resnet_correct}/{disagreements} "
            f"({resnet_disagreement_accuracy:.2f}%)"
        )

        print(
            f"Both models wrong       : "
            f"{both_wrong}/{disagreements} "
            f"({both_wrong_rate:.2f}%)"
        )

    else:

        print(
            "The models made identical "
            "predictions on every test image."
        )

    # ========================================================
    # SHOW INDIVIDUAL DISAGREEMENT CASES
    # ========================================================

    if disagreements > 0:

        print("\n" + "=" * 60)
        print("DISAGREEMENT CASES")
        print("=" * 60)

        for idx in disagree_indices:

            true_label = CLASS_NAMES[
                int(labels[idx])
            ]

            e_pred_idx = int(
                effnet_preds[idx]
            )

            r_pred_idx = int(
                resnet_preds[idx]
            )

            e_pred = CLASS_NAMES[
                e_pred_idx
            ]

            r_pred = CLASS_NAMES[
                r_pred_idx
            ]

            e_conf = float(
                effnet_probs[idx][e_pred_idx]
            )

            r_conf = float(
                resnet_probs[idx][r_pred_idx]
            )

            print(
                f"CASE {idx:03d} | "
                f"TRUE={true_label} | "
                f"EFFNET={e_pred} ({e_conf:.2f}) | "
                f"RESNET={r_pred} ({r_conf:.2f})"
            )

    # ========================================================
    # GRAD-CAM GALLERY
    # ========================================================

    print("\n" + "=" * 60)
    print("BUILDING GRAD-CAM GALLERY")
    print("=" * 60)

    if disagreements == 0:

        print(
            "No disagreement cases found."
        )

        print(
            f"\nGallery folder: {GALLERY_DIR}"
        )

        return

    eval_transform = get_transforms(
        train=False
    )

    # ========================================================
    # CREATE GRAD-CAM FOR EACH DISAGREEMENT
    # ========================================================

    for count, idx in enumerate(
        disagree_indices,
        start=1
    ):

        path = file_paths[idx]

        true_label = CLASS_NAMES[
            int(labels[idx])
        ]

        e_pred_idx = int(
            effnet_preds[idx]
        )

        r_pred_idx = int(
            resnet_preds[idx]
        )

        e_pred = CLASS_NAMES[
            e_pred_idx
        ]

        r_pred = CLASS_NAMES[
            r_pred_idx
        ]

        e_conf = float(
            effnet_probs[idx][e_pred_idx]
        )

        r_conf = float(
            resnet_probs[idx][r_pred_idx]
        )

        print(
            f"\nGenerating Grad-CAM "
            f"{count}/{disagreements}..."
        )

        print(
            f"  Test index : {idx}"
        )

        print(
            f"  True       : {true_label}"
        )

        print(
            f"  EfficientNet-B0 : "
            f"{e_pred} ({e_conf:.2f})"
        )

        print(
            f"  ResNet50        : "
            f"{r_pred} ({r_conf:.2f})"
        )

        # ----------------------------------------------------
        # Load original image
        # ----------------------------------------------------

        pil_image = Image.open(
            path
        ).convert("RGB")

        # ----------------------------------------------------
        # Apply evaluation transform
        # ----------------------------------------------------

        input_tensor = (
            eval_transform(
                pil_image
            )
            .unsqueeze(0)
        )

        # ----------------------------------------------------
        # Prepare display image
        # ----------------------------------------------------

        display_img = (
            preprocess_for_display(
                pil_image
            )
        )

        # ----------------------------------------------------
        # EfficientNet Grad-CAM
        # ----------------------------------------------------

        effnet_cam = generate_gradcam(
            effnet,
            "efficientnet_b0",
            input_tensor,
            display_img,
            target_class=e_pred_idx,
            device=device
        )

        # ----------------------------------------------------
        # ResNet50 Grad-CAM
        # ----------------------------------------------------

        resnet_cam = generate_gradcam(
            resnet,
            "resnet50",
            input_tensor,
            display_img,
            target_class=r_pred_idx,
            device=device
        )

        # ----------------------------------------------------
        # Create 3-panel figure
        # ----------------------------------------------------

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(13, 4.5)
        )

        # Original
        axes[0].imshow(
            display_img
        )

        axes[0].set_title(
            f"Original\n"
            f"True: {true_label}"
        )

        axes[0].axis("off")

        # EfficientNet
        axes[1].imshow(
            effnet_cam
        )

        axes[1].set_title(
            f"EfficientNet-B0\n"
            f"Pred: {e_pred} "
            f"({e_conf:.2f})"
        )

        axes[1].axis("off")

        # ResNet50
        axes[2].imshow(
            resnet_cam
        )

        axes[2].set_title(
            f"ResNet50\n"
            f"Pred: {r_pred} "
            f"({r_conf:.2f})"
        )

        axes[2].axis("off")

        fig.suptitle(
            f"Final disagreement case - "
            f"test image #{idx}",
            fontsize=13,
            fontweight="bold"
        )

        plt.tight_layout()

        # ----------------------------------------------------
        # Save Grad-CAM image
        # ----------------------------------------------------

        output_path = os.path.join(
            GALLERY_DIR,
            (
                f"disagreement_{idx:03d}_"
                f"true-{true_label}.png"
            )
        )

        plt.savefig(
            output_path,
            dpi=120,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            f"  Saved: {output_path}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL DISAGREEMENT ANALYSIS COMPLETE")
    print("=" * 60)

    print(
        f"Total test images : {total}"
    )

    print(
        f"Agreement cases   : {agreements}"
    )

    print(
        f"Disagreement cases: {disagreements}"
    )

    print(
        f"Agreement rate    : "
        f"{agreement_rate:.2f}%"
    )

    print(
        f"Disagreement rate : "
        f"{disagreement_rate:.2f}%"
    )

    if disagreements > 0:

        print(
            f"EfficientNet correct on disagreements: "
            f"{effnet_correct}/{disagreements}"
        )

        print(
            f"ResNet50 correct on disagreements: "
            f"{resnet_correct}/{disagreements}"
        )

        print(
            f"Both wrong on disagreements: "
            f"{both_wrong}/{disagreements}"
        )

    print(
        f"\nGallery folder: {GALLERY_DIR}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()