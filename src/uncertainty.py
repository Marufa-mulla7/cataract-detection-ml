"""
Monte Carlo Dropout uncertainty estimation.

The classifier head in models.py keeps Dropout layers, so we can run many
stochastic forward passes at inference time and use the spread of predictions
as a confidence/uncertainty signal — a genuinely useful feature for a medical
screening tool ("the model is unsure, please have this reviewed by a specialist").
"""

import torch
import numpy as np


def enable_mc_dropout(model):
    """Set only Dropout layers to train mode, keep everything else (BatchNorm etc.) in eval mode."""
    model.eval()
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()
    return model


@torch.no_grad()
def mc_dropout_predict(model, input_tensor, num_samples=30, device=None):
    """
    Returns:
        mean_probs: average predicted probability per class, shape (num_classes,)
        uncertainty: standard deviation across samples per class (higher = less confident)
        predicted_class: argmax of mean_probs
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = enable_mc_dropout(model)

    input_tensor = input_tensor.to(device)
    samples = []
    for _ in range(num_samples):
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()
        samples.append(probs)

    samples = np.stack(samples, axis=0)  # (num_samples, batch, num_classes)
    mean_probs = samples.mean(axis=0)[0]
    uncertainty = samples.std(axis=0)[0]
    predicted_class = int(mean_probs.argmax())

    return mean_probs, uncertainty, predicted_class


def confidence_flag(mean_probs, uncertainty, threshold=0.15):
    """
    Simple rule for the demo app: if the uncertainty for the predicted class
    exceeds the threshold, flag the result as low-confidence.
    """
    predicted_class = int(mean_probs.argmax())
    is_uncertain = uncertainty[predicted_class] > threshold
    return is_uncertain
