import os
import torch

from src.models import build_model
from src.data_loader import get_dataloaders, CLASS_NAMES
from src.evaluate import evaluate_model

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Final dataset
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract_final")

# Final EfficientNet-B0 checkpoint
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_efficientnet_b0_5class_final.pt"
)

MODEL_KEY = "efficientnet_b0"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = build_model(
        MODEL_KEY,
        num_classes=len(CLASS_NAMES),
        freeze_backbone=False
    )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    _, val_loader, test_loader = get_dataloaders(DATA_ROOT)

    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS (EfficientNet-B0)")
    print("=" * 60)

    evaluate_model(
        model,
        test_loader,
        CLASS_NAMES,
        device=device,
        save_prefix="test_efficientnet_b0_final"
    )

    print("\n" + "=" * 60)
    print("FINAL VALIDATION SET RESULTS (EfficientNet-B0)")
    print("=" * 60)

    evaluate_model(
        model,
        val_loader,
        CLASS_NAMES,
        device=device,
        save_prefix="val_efficientnet_b0_final"
    )


if __name__ == "__main__":
    main()