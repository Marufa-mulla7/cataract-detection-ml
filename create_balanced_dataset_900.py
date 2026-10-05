"""
Dataset balancing - 900/400 target, directly addressing Diabetic Retinopathy
dominance.

Diabetic Retinopathy currently makes up 53% of the dataset (1,857 of 3,479
images). This script undersamples it down to 900 real images (still 100%
real, just less overwhelming), while topping up Normal, Cataract, and
Glaucoma to 900 each via synthetic augmentation, and AMD to 400 (adding
100 more synthetic on top of its existing 200).

Honest composition at these targets:
  Normal:                900 total (300 real + 600 synthetic, 33% real)
  Cataract:               900 total (382 real + 518 synthetic, 42% real)
  Glaucoma:               900 total (340 real + 560 synthetic, 38% real)
  Diabetic Retinopathy:   900 total (900 real, undersampled from 1857, 100% real)
  AMD:                    400 total (100 real + 300 synthetic, 25% real)
  TOTAL:                 4000 images

This is NON-DESTRUCTIVE: builds a new data/cataract_balanced_900/ folder.
Your original data/cataract/ and any earlier balanced folders are untouched.
"""

import os
import random
import shutil

from PIL import Image
from torchvision import transforms

SOURCE_ROOT = os.path.join("data", "cataract")
DEST_ROOT = os.path.join("data", "cataract_balanced_900")

TOP_UP_TARGETS = {
    "normal": 900,
    "cataract": 900,
    "glaucoma": 900,
    "amd": 400,
}
UNDERSAMPLE_TARGETS = {
    "diabetic_retinopathy": 900,
}

AUGMENT_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
])


def list_images(folder):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def top_up_class(class_name, target_count):
    src_dir = os.path.join(SOURCE_ROOT, class_name)
    dst_dir = os.path.join(DEST_ROOT, class_name)
    os.makedirs(dst_dir, exist_ok=True)

    real_images = list_images(src_dir)
    n_real = len(real_images)

    for fname in real_images:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    n_needed = max(0, target_count - n_real)
    n_synthetic = 0

    while n_synthetic < n_needed:
        source_fname = random.choice(real_images)
        source_path = os.path.join(src_dir, source_fname)

        try:
            img = Image.open(source_path).convert("RGB")
        except Exception as e:
            print(f"    Skipping unreadable file {source_fname}: {e}")
            continue

        augmented = AUGMENT_TRANSFORM(img)
        base_name, ext = os.path.splitext(source_fname)
        new_fname = f"aug_{n_synthetic:04d}_{base_name}{ext}"
        augmented.save(os.path.join(dst_dir, new_fname))
        n_synthetic += 1

    print(f"  {class_name}: {n_real} real + {n_synthetic} synthetic = "
          f"{n_real + n_synthetic} total ({n_real / (n_real + n_synthetic) * 100:.0f}% real).")
    return n_real, n_synthetic


def undersample_class(class_name, target_count):
    src_dir = os.path.join(SOURCE_ROOT, class_name)
    dst_dir = os.path.join(DEST_ROOT, class_name)
    os.makedirs(dst_dir, exist_ok=True)

    all_images = list_images(src_dir)
    n_available = len(all_images)

    selected = all_images if n_available <= target_count else random.sample(all_images, target_count)

    for fname in selected:
        shutil.copy2(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

    print(f"  {class_name}: kept {len(selected)} of {n_available} real images "
          f"({'no change' if n_available <= target_count else f'dropped {n_available - len(selected)}'}) - 100% real.")
    return len(selected), 0


def main():
    random.seed(42)

    print("Building 900/400-target balanced dataset...")
    print(f"Source: {SOURCE_ROOT}")
    print(f"Destination: {DEST_ROOT}\n")

    os.makedirs(DEST_ROOT, exist_ok=True)

    summary = {}

    print("Topped-up classes (synthetic augmentation added):")
    for class_name, target in TOP_UP_TARGETS.items():
        summary[class_name] = top_up_class(class_name, target)

    print("\nUndersampled classes (real images only, dropped excess):")
    for class_name, target in UNDERSAMPLE_TARGETS.items():
        summary[class_name] = undersample_class(class_name, target)

    print(f"\n{'=' * 60}")
    print("BALANCED DATASET SUMMARY (900/400 target)")
    print(f"{'=' * 60}")
    total_real = 0
    total_synth = 0
    for class_name, (n_real, n_synth) in summary.items():
        total = n_real + n_synth
        pct_real = (n_real / total * 100) if total > 0 else 0
        print(f"  {class_name:22s}: {total:5d} total  ({n_real} real, {n_synth} synthetic, {pct_real:.0f}% real)")
        total_real += n_real
        total_synth += n_synth

    grand_total = total_real + total_synth
    print(f"\n  {'TOTAL':22s}: {grand_total:5d} images "
          f"({total_real} real, {total_synth} synthetic, {total_real / grand_total * 100:.0f}% real overall)")
    print(f"\nBalanced dataset saved to: {DEST_ROOT}")
    print("Your original data/cataract/ folder is untouched.")


if __name__ == "__main__":
    main()