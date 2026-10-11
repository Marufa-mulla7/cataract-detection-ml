# %% [markdown]
# # Cataract/Ocular Disease Detection - Training Notebook
# Upload this file to Google Colab (or paste cells into a new Colab notebook).
# Runtime -> Change runtime type -> GPU (T4) before running.

# %% [markdown]
# ## 1. Setup: clone repo + install deps

# %%
# !git clone <your-repo-url>
# %cd cataract-detection-ml
# !pip install -q -r requirements.txt

# %% [markdown]
# ## 2. Download dataset from Kaggle
# Upload your kaggle.json first (Kaggle account -> Create New API Token)

# %%
# from google.colab import files
# files.upload()  # select kaggle.json
# !mkdir -p ~/.kaggle
# !cp kaggle.json ~/.kaggle/
# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle datasets download -d jr2ngb/cataractdataset
# !unzip -q cataractdataset.zip -d data/cataract

# Optional: external validation set
# !kaggle datasets download -d andrewmvd/ocular-disease-recognition-odir5k
# !unzip -q ocular-disease-recognition-odir5k.zip -d data/odir_external

# %% [markdown]
# ## 3. Load data

# %%
import sys
sys.path.append(".")

from src.data_loader import get_dataloaders, compute_class_weights, CLASS_NAMES

train_loader, val_loader, test_loader = get_dataloaders("data/cataract", batch_size=32)
class_weights = compute_class_weights("data/cataract")
print("Class weights:", class_weights)

# %% [markdown]
# ## 4. Train and compare multiple architectures
# This is the core "advanced" experiment: train each architecture with the
# same data/hyperparameters, compare validation accuracy fairly.

# %%
from src.models import build_model, unfreeze_backbone
from src.train import train_model

import os
os.makedirs("models", exist_ok=True)

architectures = ["mobilenet_v2", "efficientnet_b0", "resnet50", "vit_small"]
results = {}

for arch in architectures:
    print(f"\n=== Training {arch} ===")
    model = build_model(arch, num_classes=len(CLASS_NAMES), freeze_backbone=True)

    # phase 1: train classifier head only (fast)
    history = train_model(
        model, train_loader, val_loader, class_weights,
        epochs=10, lr=1e-3, save_path=f"models/{arch}_head.pt"
    )

    # phase 2: unfreeze last layers, fine-tune with lower LR
    model = unfreeze_backbone(model, num_layers_from_end=20)
    history_ft = train_model(
        model, train_loader, val_loader, class_weights,
        epochs=10, lr=1e-4, save_path=f"models/{arch}_best.pt"
    )

    results[arch] = max(history_ft["val_acc"])

print("\nFinal comparison:", results)

# %% [markdown]
# ## 5. Plot architecture comparison

# %%
from src.evaluate import compare_architectures
compare_architectures(results, save_path="architecture_comparison.png")

# %% [markdown]
# ## 6. Evaluate the best model on the held-out test set

# %%
best_arch = max(results, key=results.get)
print(f"Best architecture: {best_arch}")

from src.evaluate import evaluate_model

best_model = build_model(best_arch, num_classes=len(CLASS_NAMES), freeze_backbone=False)
import torch
best_model.load_state_dict(torch.load(f"models/{best_arch}_best.pt"))

evaluate_model(best_model, test_loader, CLASS_NAMES, save_prefix="test_results")

# %% [markdown]
# ## 7. External validation (generalization test)
# Test the best model on the second, unseen dataset it never trained on.

# %%
from src.data_loader import get_external_test_loader

external_loader = get_external_test_loader("data/odir_external")
evaluate_model(best_model, external_loader, CLASS_NAMES, save_prefix="external_validation")

# %% [markdown]
# ## 8. Download the winning checkpoint
# Rename it to best_model.pt and place it in your local repo's models/ folder
# for use by the Streamlit app.

# %%
# from google.colab import files
# files.download(f"models/{best_arch}_best.pt")
