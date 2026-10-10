"""
Remove confirmed cross-class duplicate images.

For every pair of images across two different classes with a perceptual
hash distance <= 2 (confirmed via check_cross_class_duplicates.py to be
real duplicates, not false positives), this removes BOTH copies - since
we have no ground truth for which label is actually correct, keeping
either one risks training on a mislabeled example.

SAFE: moves files to data/cataract/_cross_class_conflicts_removed/
rather than deleting them. Nothing is permanently lost.
"""

import os
import shutil

from PIL import Image
import imagehash

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract")
REMOVED_DIR = os.path.join(DATA_ROOT, "_cross_class_conflicts_removed")

CLASS_NAMES = ["normal", "cataract", "glaucoma", "amd", "diabetic_retinopathy"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
HAMMING_THRESHOLD = 2


def hash_class_folder(class_name):
    folder = os.path.join(DATA_ROOT, class_name)
    hashes = {}
    if not os.path.isdir(folder):
        return hashes
    for fname in os.listdir(folder):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            try:
                img = Image.open(os.path.join(folder, fname))
                hashes[fname] = imagehash.phash(img)
            except Exception:
                continue
    return hashes


def main():
    print("Hashing every image in every class...\n")
    class_hashes = {}
    for class_name in CLASS_NAMES:
        hashes = hash_class_folder(class_name)
        class_hashes[class_name] = hashes
        print(f"  {class_name}: {len(hashes)} images hashed")

    to_remove = {}  # class_name -> set of filenames to remove

    class_list = list(CLASS_NAMES)
    for i in range(len(class_list)):
        for j in range(i + 1, len(class_list)):
            class_a, class_b = class_list[i], class_list[j]
            for fname_a, hash_a in class_hashes[class_a].items():
                for fname_b, hash_b in class_hashes[class_b].items():
                    if hash_a - hash_b <= HAMMING_THRESHOLD:
                        to_remove.setdefault(class_a, set()).add(fname_a)
                        to_remove.setdefault(class_b, set()).add(fname_b)

    total_to_remove = sum(len(v) for v in to_remove.values())
    print(f"\nFound {total_to_remove} individual image files involved in cross-class conflicts.\n")

    if total_to_remove == 0:
        print("Nothing to remove. Dataset is clean.")
        return

    os.makedirs(REMOVED_DIR, exist_ok=True)

    for class_name, filenames in to_remove.items():
        class_removed_dir = os.path.join(REMOVED_DIR, class_name)
        os.makedirs(class_removed_dir, exist_ok=True)
        for fname in filenames:
            src = os.path.join(DATA_ROOT, class_name, fname)
            dst = os.path.join(class_removed_dir, fname)
            if os.path.exists(src):
                shutil.move(src, dst)

    print(f"{'=' * 60}")
    print("CLEANUP SUMMARY")
    print(f"{'=' * 60}")
    for class_name in CLASS_NAMES:
        removed_count = len(to_remove.get(class_name, []))
        remaining = len([
            f for f in os.listdir(os.path.join(DATA_ROOT, class_name))
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])
        print(f"  {class_name:22s}: removed {removed_count:3d}, now has {remaining} images")

    print(f"\nAll removed files moved to: {REMOVED_DIR}")
    print("Nothing was deleted - files are safely sitting there if you want to review them.")


if __name__ == "__main__":
    main()