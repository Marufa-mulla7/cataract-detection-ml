"""
Reduce Diabetic Retinopathy from 1,857 to 1,000 images.

This is a deliberate reversal of the earlier "never decrease" rule,
specifically for this one class, per an explicit decision to address
DR's dominance directly (it was 53% of the original dataset).

SAFE BY DESIGN: excess images are MOVED to a backup folder
(data/cataract/diabetic_retinopathy_excess_backup/), never deleted.
If you change your mind later, you can move them straight back.
"""

import os
import random
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract")

DR_DIR = os.path.join(DATA_ROOT, "diabetic_retinopathy")
BACKUP_DIR = os.path.join(DATA_ROOT, "diabetic_retinopathy_excess_backup")

TARGET_COUNT = 1000
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_images(folder):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]


def main():
    random.seed(42)

    if not os.path.isdir(DR_DIR):
        print(f"Could not find: {DR_DIR}")
        return

    all_images = list_images(DR_DIR)
    n_total = len(all_images)
    print(f"Current Diabetic Retinopathy count: {n_total}")

    if n_total <= TARGET_COUNT:
        print(f"Already at or below target ({TARGET_COUNT}) - nothing to do.")
        return

    n_to_move = n_total - TARGET_COUNT
    to_move = random.sample(all_images, n_to_move)

    os.makedirs(BACKUP_DIR, exist_ok=True)

    n_moved = 0
    for fname in to_move:
        src = os.path.join(DR_DIR, fname)
        dst = os.path.join(BACKUP_DIR, fname)
        shutil.move(src, dst)
        n_moved += 1

    remaining = len(list_images(DR_DIR))

    print(f"\nMoved {n_moved} images to: {BACKUP_DIR}")
    print(f"Diabetic Retinopathy now has: {remaining} images (target was {TARGET_COUNT}).")
    print("\nThese images are NOT deleted - they're sitting safely in the backup folder.")
    print("To undo this, move the files back from diabetic_retinopathy_excess_backup/ into diabetic_retinopathy/.")


if __name__ == "__main__":
    main()