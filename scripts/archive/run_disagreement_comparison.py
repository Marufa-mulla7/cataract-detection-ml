import os
from collections import Counter

import torch

from src.models import build_model
from src.data_loader import get_dataloaders, CLASS_NAMES


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "cataract_final"
)

RESNET_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_resnet50_5class_final.pt"
)

EFFICIENTNET_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "best_efficientnet_b0_5class_final.pt"
)


def load_model(model_key, model_path, device):
    model = build_model(
        model_key,
        num_classes=len(CLASS_NAMES),
        freeze_backbone=False
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the exact same final test loader used for evaluation
    _, _, test_loader = get_dataloaders(DATA_ROOT)

    print("\nLoading ResNet50...")
    resnet = load_model(
        "resnet50",
        RESNET_PATH,
        device
    )

    print("Loading EfficientNet-B0...")
    efficientnet = load_model(
        "efficientnet_b0",
        EFFICIENTNET_PATH,
        device
    )

    resnet_predictions = []
    efficientnet_predictions = []
    true_labels = []

    print("\nRunning both models on the final test set...")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            resnet_outputs = resnet(images)
            efficientnet_outputs = efficientnet(images)

            resnet_preds = torch.argmax(
                resnet_outputs,
                dim=1
            )

            efficientnet_preds = torch.argmax(
                efficientnet_outputs,
                dim=1
            )

            resnet_predictions.extend(
                resnet_preds.cpu().tolist()
            )

            efficientnet_predictions.extend(
                efficientnet_preds.cpu().tolist()
            )

            true_labels.extend(
                labels.tolist()
            )

    total = len(true_labels)

    agreements = 0
    disagreements = 0

    resnet_correct_disagreements = 0
    efficientnet_correct_disagreements = 0
    both_wrong_disagreements = 0

    disagreement_pairs = Counter()

    for true, resnet_pred, efficientnet_pred in zip(
        true_labels,
        resnet_predictions,
        efficientnet_predictions
    ):
        if resnet_pred == efficientnet_pred:
            agreements += 1
        else:
            disagreements += 1

            disagreement_pairs[
                (
                    CLASS_NAMES[resnet_pred],
                    CLASS_NAMES[efficientnet_pred]
                )
            ] += 1

            if resnet_pred == true:
                resnet_correct_disagreements += 1

            if efficientnet_pred == true:
                efficientnet_correct_disagreements += 1

            if resnet_pred != true and efficientnet_pred != true:
                both_wrong_disagreements += 1

    resnet_correct = sum(
        pred == true
        for pred, true in zip(
            resnet_predictions,
            true_labels
        )
    )

    efficientnet_correct = sum(
        pred == true
        for pred, true in zip(
            efficientnet_predictions,
            true_labels
        )
    )

    print("\n" + "=" * 65)
    print("FINAL RESNET50 vs EFFICIENTNET-B0 COMPARISON")
    print("=" * 65)

    print(f"\nTotal test images: {total}")

    print("\nOverall accuracy:")
    print(
        f"  ResNet50:       "
        f"{resnet_correct}/{total} "
        f"({resnet_correct / total * 100:.2f}%)"
    )

    print(
        f"  EfficientNet-B0:"
        f" {efficientnet_correct}/{total} "
        f"({efficientnet_correct / total * 100:.2f}%)"
    )

    print("\nPrediction agreement:")
    print(
        f"  Agreements:    {agreements}/{total} "
        f"({agreements / total * 100:.2f}%)"
    )

    print(
        f"  Disagreements: {disagreements}/{total} "
        f"({disagreements / total * 100:.2f}%)"
    )

    if disagreements > 0:
        print("\nAmong disagreements:")

        print(
            f"  ResNet50 correct:        "
            f"{resnet_correct_disagreements}/{disagreements} "
            f"({resnet_correct_disagreements / disagreements * 100:.2f}%)"
        )

        print(
            f"  EfficientNet-B0 correct: "
            f"{efficientnet_correct_disagreements}/{disagreements} "
            f"({efficientnet_correct_disagreements / disagreements * 100:.2f}%)"
        )

        print(
            f"  Both incorrect:          "
            f"{both_wrong_disagreements}/{disagreements} "
            f"({both_wrong_disagreements / disagreements * 100:.2f}%)"
        )

        print("\nDisagreement prediction pairs:")
        print(
            "  ResNet50 prediction -> EfficientNet-B0 prediction"
        )

        for (resnet_class, efficientnet_class), count in sorted(
            disagreement_pairs.items(),
            key=lambda x: (-x[1], x[0])
        ):
            print(
                f"  {resnet_class:22s} -> "
                f"{efficientnet_class:22s}: {count}"
            )

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()