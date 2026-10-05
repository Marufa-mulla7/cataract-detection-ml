"""
Model builders. Uses `timm` so CNNs and Vision Transformers share one interface,
which is what makes the architecture-comparison experiment easy to run fairly.
"""

import timm
import torch.nn as nn

SUPPORTED_MODELS = {
    "mobilenet_v2": "mobilenetv2_100",
    "efficientnet_b0": "efficientnet_b0",
    "resnet50": "resnet50",
    "vit_small": "vit_small_patch16_224",
}


def build_model(model_key: str, num_classes: int, freeze_backbone: bool = True, dropout: float = 0.3):
    """
    Build a model by key (see SUPPORTED_MODELS).
    freeze_backbone=True trains only the classifier head first (fast, good for small datasets).
    Unfreeze later for fine-tuning once the head has converged.
    """
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model_key '{model_key}'. Choose from {list(SUPPORTED_MODELS)}")

    timm_name = SUPPORTED_MODELS[model_key]
    model = timm.create_model(timm_name, pretrained=True, num_classes=0)  # strip default head

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.num_features
    classifier = nn.Sequential(
        nn.Dropout(dropout),          # kept active at inference time for MC Dropout uncertainty
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, num_classes),
    )

    model = nn.Sequential(model, classifier)
    return model


def unfreeze_backbone(model, num_layers_from_end: int = 20):
    """Unfreeze the last N parameters for fine-tuning after initial head training."""
    backbone = model[0]
    params = list(backbone.parameters())
    for p in params[-num_layers_from_end:]:
        p.requires_grad = True
    return model
