import os
import torch

from src.models import build_model
from src.data_loader import get_dataloaders, CLASS_NAMES
from src.evaluate import evaluate_model


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Final leakage-free dataset
DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "cataract_final"
)

# Best ResNet50 checkpoint from the final training run
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_resnet50_5class_final.pt"
)

MODEL_KEY = "resnet50"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build ResNet50
    model = build_model(
        MODEL_KEY,
        num_classes=len(CLASS_NAMES),
        freeze_backbone=False
    )

    # Load the best checkpoint from training
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Load final train/validation/test dataset
    _, val_loader, test_loader = get_dataloaders(DATA_ROOT)

    # ---------------------------------------------------------
    # FINAL TEST SET EVALUATION
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS (ResNet50)")
    print("=" * 60)

    evaluate_model(
        model,
        test_loader,
        CLASS_NAMES,
        device=device,
        save_prefix="test_resnet50_final"
    )

    # ---------------------------------------------------------
    # FINAL VALIDATION SET EVALUATION
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SET RESULTS (ResNet50)")
    print("=" * 60)

    evaluate_model(
        model,
        val_loader,
        CLASS_NAMES,
        device=device,
        save_prefix="val_resnet50_final"
    )


if __name__ == "__main__":
    main()