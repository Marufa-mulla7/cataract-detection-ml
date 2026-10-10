"""
Data loading utilities for the final leakage-free cataract/ocular disease dataset.

Expected folder layout:

data/
  cataract_final/
    train/
      normal/
      cataract/
      glaucoma/
      amd/
      diabetic_retinopathy/
    val/
      normal/
      cataract/
      glaucoma/
      amd/
      diabetic_retinopathy/
    test/
      normal/
      cataract/
      glaucoma/
      amd/
      diabetic_retinopathy/

IMPORTANT:
- The dataset has already been split by build_final_dataset.py.
- DO NOT randomly split train/val/test again.
- Validation and test contain REAL images only.
- Augmented images exist only in training.
- Each split has its own independent transform.
"""

import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


CLASS_NAMES = [
    "normal",
    "cataract",
    "glaucoma",
    "amd",
    "diabetic_retinopathy",
]

IMG_SIZE = 224
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def get_transforms(train: bool):
    """
    Return image transforms.

    Training:
        - Resize
        - Random horizontal flip
        - Random rotation
        - Color jitter
        - Normalize

    Validation/Test:
        - Resize
        - Normalize only
    """

    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class EyeDataset(Dataset):
    """
    Image-folder-style dataset.

    Each instance points to ONE split only:
        train
        val
        test
    """

    def __init__(
        self,
        root_dir,
        class_names=CLASS_NAMES,
        transform=None,
    ):
        self.samples = []
        self.transform = transform
        self.class_names = class_names

        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)

            if not os.path.isdir(class_dir):
                continue

            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(IMAGE_EXTENSIONS):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under {root_dir}. "
                f"Check the dataset structure."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def class_distribution(self):
        counts = Counter(label for _, label in self.samples)

        return {
            self.class_names[k]: v
            for k, v in sorted(counts.items())
        }


def get_dataloaders(
    data_root,
    batch_size=32,
    num_workers=2,
):
    """
    Build train/validation/test dataloaders.

    IMPORTANT:
    The train/val/test split already exists on disk.
    We do NOT perform another random split here.
    """

    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")
    test_root = os.path.join(data_root, "test")

    # Check that all three splits exist.
    for split_name, split_root in [
        ("train", train_root),
        ("val", val_root),
        ("test", test_root),
    ]:
        if not os.path.isdir(split_root):
            raise RuntimeError(
                f"Missing '{split_name}' directory:\n"
                f"{split_root}\n\n"
                f"Expected dataset structure:\n"
                f"{data_root}\\train\n"
                f"{data_root}\\val\n"
                f"{data_root}\\test"
            )

    # Each split gets its own dataset and transform.
    train_dataset = EyeDataset(
        train_root,
        class_names=CLASS_NAMES,
        transform=get_transforms(train=True),
    )

    val_dataset = EyeDataset(
        val_root,
        class_names=CLASS_NAMES,
        transform=get_transforms(train=False),
    )

    test_dataset = EyeDataset(
        test_root,
        class_names=CLASS_NAMES,
        transform=get_transforms(train=False),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nDataset split:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val  : {len(val_dataset)}")
    print(f"  Test : {len(test_dataset)}")
    print(f"  Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")

    print("\nClass distribution:")
    print("  TRAIN:")
    for class_name, count in train_dataset.class_distribution().items():
        print(f"    {class_name:25s}: {count}")

    print("  VALIDATION:")
    for class_name, count in val_dataset.class_distribution().items():
        print(f"    {class_name:25s}: {count}")

    print("  TEST:")
    for class_name, count in test_dataset.class_distribution().items():
        print(f"    {class_name:25s}: {count}")

    return train_loader, val_loader, test_loader


def compute_class_weights(
    data_root,
    class_names=CLASS_NAMES,
):
    """
    Compute class weights using the TRAINING split only.

    This is important because validation/test data must not influence
    training decisions.
    """

    train_root = os.path.join(data_root, "train")

    dataset = EyeDataset(
        train_root,
        class_names=class_names,
        transform=None,
    )

    dist = dataset.class_distribution()

    counts = np.array(
        [dist.get(class_name, 1) for class_name in class_names],
        dtype=np.float32,
    )

    weights = counts.sum() / (
        len(counts) * counts
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
    )


def get_external_test_loader(
    data_root,
    batch_size=32,
    num_workers=2,
):
    """
    Loader for an independent external dataset.
    """

    dataset = EyeDataset(
        data_root,
        class_names=CLASS_NAMES,
        transform=get_transforms(train=False),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )