import os
import time
import torch

from src.models import build_model
from src.data_loader import get_dataloaders, compute_class_weights, CLASS_NAMES
from src.train import train_model

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract_balanced")
SAVE_PATH = os.path.join(PROJECT_ROOT, "models", "best_model_5class_balanced.pt")
MODEL_KEY = "efficientnet_b0"

EPOCHS = 20
LEARNING_RATE = 1e-3
PATIENCE = 5


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training on BALANCED dataset: {DATA_ROOT}")
    print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

    print("\nBuilding dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(DATA_ROOT)

    print("\nComputing class weights (to handle any remaining imbalance)...")
    class_weights = compute_class_weights(DATA_ROOT, class_names=CLASS_NAMES)
    print(f"Class weights: {dict(zip(CLASS_NAMES, class_weights.tolist()))}")

    print("\nBuilding model...")
    model = build_model(MODEL_KEY, num_classes=len(CLASS_NAMES), freeze_backbone=False)

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print(f"\nStarting training for up to {EPOCHS} epochs (patience={PATIENCE})...")
    print(f"Checkpoints will be saved to: {SAVE_PATH}\n")

    start_time = time.time()

    history = train_model(
        model, train_loader, val_loader,
        class_weights=class_weights, epochs=EPOCHS, lr=LEARNING_RATE,
        save_path=SAVE_PATH, patience=PATIENCE, device=device,
    )

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\nTraining finished in {elapsed_minutes:.1f} minutes.")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
    print(f"Model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    main()