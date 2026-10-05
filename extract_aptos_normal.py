"""
Extract unused real "Normal" images from your existing APTOS 2019 download.

Your project already downloaded APTOS 2019 for Diabetic Retinopathy, but only
used the DR-positive images (diagnosis grades 1-4, totaling 1,857). The
"No DR" images (grade 0, genuinely healthy) were never touched - and there
are far more of them than we need (roughly 1,805 available, we only need 600).

This does NOT touch your Diabetic Retinopathy class at all - it only reads
from APTOS's own train.csv/train_images folder (wherever prepare_dr_dataset.py
originally downloaded them) and copies grade-0 images into your Normal class.

IMPORTANT: update APTOS_CSV_PATH and APTOS_IMAGES_DIR below to match wherever
your project's APTOS download actually lives - this project's exact download
location wasn't shared, so these are reasonable guesses based on the standard
APTOS 2019 competition layout. If the paths are wrong, this script will print
a clear "not found" message rather than fail silently - check the printed
search suggestions and correct the paths.
"""

import os
import csv
import random
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- ADJUST THESE TWO PATHS if they don't match your actual APTOS download location ---
APTOS_CSV_PATH = os.path.join(PROJECT_ROOT, "aptos2019-blindness-detection", "train.csv")
APTOS_IMAGES_DIR = os.path.join(PROJECT_ROOT, "aptos2019-blindness-detection", "train_images")
# ----------------------------------------------------------------------------------

NORMAL_DEST_DIR = os.path.join(PROJECT_ROOT, "data", "cataract", "normal")
N_TO_ADD = 600
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def find_aptos_files():
    """Help locate the APTOS files if the hardcoded paths above are wrong."""
    print("Searching for train.csv anywhere under the project root (this may take a moment)...")
    candidates = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if "venv" in root or ".git" in root:
            continue
        for f in files:
            if f == "train.csv":
                candidates.append(os.path.join(root, f))
    if candidates:
        print("Found possible train.csv file(s) at:")
        for c in candidates:
            print(f"  {c}")
    else:
        print("No train.csv found anywhere under the project root.")
    print("\nUpdate APTOS_CSV_PATH and APTOS_IMAGES_DIR at the top of this script to the correct location, then rerun.")


def main():
    if not os.path.exists(APTOS_CSV_PATH):
        print(f"Could not find APTOS train.csv at: {APTOS_CSV_PATH}")
        find_aptos_files()
        return

    if not os.path.isdir(APTOS_IMAGES_DIR):
        print(f"Could not find APTOS images folder at: {APTOS_IMAGES_DIR}")
        find_aptos_files()
        return

    print(f"Reading: {APTOS_CSV_PATH}")
    grade_0_ids = []
    with open(APTOS_CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Standard APTOS columns: id_code, diagnosis (0-4)
            if row.get("diagnosis", "").strip() == "0":
                grade_0_ids.append(row["id_code"].strip())

    print(f"Found {len(grade_0_ids)} grade-0 (No DR / healthy) entries in train.csv.")

    if len(grade_0_ids) == 0:
        print("No grade-0 entries found - check that 'diagnosis' is the correct column name in your train.csv.")
        return

    random.seed(42)
    selected_ids = random.sample(grade_0_ids, min(N_TO_ADD, len(grade_0_ids)))

    os.makedirs(NORMAL_DEST_DIR, exist_ok=True)

    n_copied = 0
    n_missing = 0
    for image_id in selected_ids:
        found = False
        for ext in IMAGE_EXTENSIONS:
            src_path = os.path.join(APTOS_IMAGES_DIR, image_id + ext)
            if os.path.exists(src_path):
                dst_path = os.path.join(NORMAL_DEST_DIR, f"aptos_normal_{image_id}{ext}")
                shutil.copy2(src_path, dst_path)
                n_copied += 1
                found = True
                break
        if not found:
            n_missing += 1

    print(f"\nCopied {n_copied} real APTOS grade-0 images into: {NORMAL_DEST_DIR}")
    if n_missing > 0:
        print(f"Warning: {n_missing} image files listed in train.csv were not found on disk (check file extension).")
    print(f"\nYour Normal class in data/cataract/normal/ now has its original 300 images PLUS these {n_copied} new ones.")
    print("Note: this modifies data/cataract/normal/ directly (adds files, does not remove any).")


if __name__ == "__main__":
    main()