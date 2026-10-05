"""
Generates the architecture comparison chart (EfficientNet-B0 vs ResNet50)
using the final validation accuracies from each model's training run.
Uses the existing compare_architectures() function in src/evaluate.py.

FULL HISTORY of this comparison across the project's iterations:
  1. Pre-augmentation-fix:  EfficientNet-B0 96.01% vs ResNet50 94.96% (EffNet won)
  2. Post-augmentation-fix: EfficientNet-B0 94.33% vs ResNet50 96.01% (ResNet won)
  3. Final (rebalanced + de-duplicated real dataset):
     EfficientNet-B0 96.48% vs ResNet50 95.47% (EffNet won, narrowly)

Despite EfficientNet-B0 narrowly winning in this final round, ResNet50 was
deliberately kept as the primary model in both applications - the real
test-set gap between them is negligible (96% accuracy / 0.97 vs 0.96 macro
F1), so this was a judgment call, not a performance-driven necessity.
"""

from src.evaluate import compare_architectures

# Final validation accuracies from the FINAL retrain, on the rebalanced
# (Normal/Cataract/Glaucoma boosted with real images, Diabetic Retinopathy
# reduced from 1857 to 1000, cross-class duplicates removed) dataset:
results = {
    "EfficientNet-B0": 0.9648,
    "ResNet50": 0.9547,
}

compare_architectures(results, save_path="architecture_comparison.png")