"""
Verify the final dataset composition directly from data/cataract/,
counting actual files on disk - not trusting terminal scrollback from
several different scripts run over multiple sessions.

Also explicitly checks that backup folders (amd_backup_300, etc.) are
correctly ignored - they should never be counted, since they aren't
part of CLASS_NAMES and your training scripts never look at them.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract")

CLASS_NAMES = ["normal", "cataract", "glaucoma", "amd", "diabetic_retinopathy"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def count_images(folder):
    if not os.path.isdir(folder):
        return None
    return len([
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])


def main():
    print(f"Checking: {DATA_ROOT}\n")
    print(f"{'Class':25s} {'Count':>8s}")
    print("-" * 35)

    total = 0
    for class_name in CLASS_NAMES:
        folder = os.path.join(DATA_ROOT, class_name)
        count = count_images(folder)
        if count is None:
            print(f"{class_name:25s} {'MISSING':>8s}")
        else:
            print(f"{class_name:25s} {count:8d}")
            total += count

    print("-" * 35)
    print(f"{'TOTAL (used for training)':25s} {total:8d}")

    # Sanity check: list any OTHER folders sitting in data/cataract/ that
    # are NOT one of the 5 official classes - these should all be backups
    # or excess folders, correctly ignored by training.
    print("\nOther folders present (backups / excess - correctly ignored by training):")
    other_found = False
    for entry in os.listdir(DATA_ROOT):
        full_path = os.path.join(DATA_ROOT, entry)
        if os.path.isdir(full_path) and entry not in CLASS_NAMES:
            count = count_images(full_path)
            print(f"  {entry:35s} ({count} images)")
            other_found = True
    if not other_found:
        print("  (none found)")

    print(f"\nDiabetic Retinopathy percentage of total: "
          f"{(count_images(os.path.join(DATA_ROOT, 'diabetic_retinopathy')) or 0) / total * 100:.1f}%")


if __name__ == "__main__":
    main()