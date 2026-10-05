"""
Grad-CAM explainability for the CNN models (MobileNetV2 / EfficientNet / ResNet50).

Note: Grad-CAM is designed for CNNs where feature maps have clear spatial structure.
For the Vision Transformer, use attention-rollout visualization instead if you want
to extend this further (a good "future work" talking point for your report/viva).
"""

import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def get_target_layer(model, model_key: str):
    """Return the last convolutional layer to hook, per architecture."""
    backbone = model[0]
    if model_key == "resnet50":
        return backbone.layer4[-1]
    elif model_key == "mobilenet_v2":
        return backbone.blocks[-1]
    elif model_key == "efficientnet_b0":
        return backbone.blocks[-1]
    else:
        raise ValueError(f"Grad-CAM target layer not defined for '{model_key}'")


def generate_gradcam(model, model_key, input_tensor, original_image_rgb, target_class=None, device=None):
    """
    input_tensor: preprocessed image tensor, shape (1, 3, H, W)
    original_image_rgb: numpy array (H, W, 3), float 0-1, for overlay
    Returns: overlayed heatmap image (numpy array, uint8)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    target_layer = get_target_layer(model, model_key)

    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = None if target_class is None else [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(original_image_rgb, grayscale_cam, use_rgb=True)
    return visualization


def preprocess_for_display(pil_image, size=224):
    """Resize + normalize to 0-1 float for use as the Grad-CAM overlay background."""
    img = np.array(pil_image.resize((size, size))).astype(np.float32) / 255.0
    return img
