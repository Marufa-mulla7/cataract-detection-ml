"""
Data loading utilities for the cataract/ocular disease dataset.

Expected folder layout (after downloading + unzipping the Kaggle dataset):

data/
  cataract/
    normal/
    cataract/
    glaucoma/
    retina_disease/
  odir_external/       # optional, used only for external validation
    normal/
    cataract/
    ...

FIX (see README "Known limitations" for the original bug this replaces):
train_loader, val_loader, and test_loader now each wrap their own independent
TransformedSubset instance instead of sharing one mutable EyeDataset object.
Previously, val_set.dataset.transform = ... and test_set.dataset.transform = ...
both silently overwrote the SAME underlying dataset's transform (since
random_split's three Subsets all reference one shared dataset instance), which
meant train_loader ended up using the no-augmentation eval transform too -
RandomHorizontalFlip / RandomRotation / ColorJitter were never actually applied
during training. This version fixes that by giving each split its own
TransformedSubset with its own independent transform.
"""

import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

CLASS_NAMES = ["normal", "cataract", "glaucoma", "amd", "diabetic_retinopathy"]
IMG_SIZE = 224


def get_transforms(train: bool):
    """Return torchvision transforms. Training gets augmentation, eval doesn't."""
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


class EyeDataset(Dataset):
    """Generic image-folder-style dataset for eye images.

    Used directly for the training split, and also as the source of file
    paths/labels for val/test (via TransformedSubset below), so directory
    scanning only happens once.
    """

    def __init__(self, root_dir, class_names=CLASS_NAMES, transform=None):
        self.samples = []
        self.transform = transform
        self.class_names = class_names

        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(class_dir, fname), label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {root_dir}. Check your folder layout.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_distribution(self):
        counts = Counter(label for _, label in self.samples)
        return {self.class_names[k]: v for k, v in counts.items()}


class TransformedSubset(Dataset):
    """
    A subset of (path, label) samples with its OWN independent transform.
    Unlike torch.utils.data.Subset, this does not share a mutable .transform
    attribute with any other split - each instance owns its transform, so
    setting one split's transform can never silently affect another's.
    """

    def __init__(self, samples, indices, transform):
        self.samples = samples
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.samples[self.indices[i]]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(data_root, batch_size=32, val_split=0.15, test_split=0.15, seed=42):
    """Build train/val/test dataloaders with a 70/15/15 split by default.

    Uses the same seeded shuffle order as the original torch.utils.data.random_split
    call (same seed, same train/val/test order), so the split boundaries are
    identical to the ones used in the project's original results - only the
    per-split transform assignment is fixed, not the split itself.
    """
    base_dataset = EyeDataset(data_root)  # transform=None; only used for indexing

    n_total = len(base_dataset)
    n_val = int(n_total * val_split)
    n_test = int(n_total * test_split)
    n_train = n_total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(n_total, generator=generator).tolist()

    train_idx = shuffled_indices[:n_train]
    val_idx = shuffled_indices[n_train:n_train + n_val]
    test_idx = shuffled_indices[n_train + n_val:n_train + n_val + n_test]

    train_dataset = TransformedSubset(base_dataset.samples, train_idx, get_transforms(train=True))
    val_dataset = TransformedSubset(base_dataset.samples, val_idx, get_transforms(train=False))
    test_dataset = TransformedSubset(base_dataset.samples, test_idx, get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Dataset split -> train: {n_train}, val: {n_val}, test: {n_test}")
    print(f"Class distribution (full dataset): {base_dataset.class_distribution()}")

    return train_loader, val_loader, test_loader


def compute_class_weights(data_root, class_names=CLASS_NAMES):
    """Compute inverse-frequency class weights to handle imbalance (e.g. few cataract images)."""
    dataset = EyeDataset(data_root, class_names=class_names)
    dist = dataset.class_distribution()
    counts = np.array([dist.get(c, 1) for c in class_names], dtype=np.float32)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_external_test_loader(data_root, batch_size=32):
    """Loader for the second, unseen dataset used purely for generalization testing."""
    dataset = EyeDataset(data_root, transform=get_transforms(train=False))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)