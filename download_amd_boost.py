"""
Download and integrate real AMD-positive images from the ADAM dataset
(Kaggle mirror: xiaoliang2121/adamdataset) to grow AMD from 300 to 400.

Same de-duplication approach as download_eye_diseases_boost.py - hashes
existing images first, skips anything that turns out to be a duplicate.

IMPORTANT: this dataset's exact label format isn't confirmed (could be a
CSV with an AMD/non-AMD column, or separate folders per class, or filename
prefixes). This script tries several common patterns and prints clearly
what it found, so if none of them match, you can look at the actual
downloaded structure yourself and tell me what's there.

Adds directly into data/cataract/amd/ - back it up first (same as the
other three folders) if you want an undo option.
"""

import os
import csv
import shutil
import subprocess
import zipfile

from PIL import Image
import imagehash

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "_adam_download")
KAGGLE_DATASET = "xiaoliang2121/adamdataset"

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract")
AMD_DEST_DIR = os.path.join(DATA_ROOT, "amd")

N_TO_ADD = 100
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def download_dataset():
    if os.path.isdir(DOWNLOAD_DIR) and any(os.scandir(DOWNLOAD_DIR)):
        print(f"Download folder already has content, skipping download: {DOWNLOAD_DIR}")
        return

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} via Kaggle API...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", DOWNLOAD_DIR],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("Kaggle download failed:")
        print(result.stderr)
        return

    for fname in os.listdir(DOWNLOAD_DIR):
        if fname.endswith(".zip"):
            zip_path = os.path.join(DOWNLOAD_DIR, fname)
            print(f"Extracting {fname}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(DOWNLOAD_DIR)


def inspect_download():
    """Print the folder structure so we can figure out the label format."""
    print("\nDownloaded folder structure (first few levels):")
    for root, dirs, files in os.walk(DOWNLOAD_DIR):
        depth = root[len(DOWNLOAD_DIR):].count(os.sep)
        if depth > 2:
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/")
        image_count = len([f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)])
        csv_files = [f for f in files if f.lower().endswith(".csv")]
        if image_count:
            print(f"{indent}  ({image_count} image files)")
        for c in csv_files:
            print(f"{indent}  CSV found: {c}")


def find_amd_positive_images():
    """
    Try common patterns for how AMD-positive images might be identified:
    1. A CSV with a label column mentioning 'AMD' or 'category'/'class'.
    2. A folder literally named 'AMD' or containing 'amd' (excluding 'non-amd'/'normal').
    """
    candidates = []

    # Pattern 1: CSV-based labels
    for root, dirs, files in os.walk(DOWNLOAD_DIR):
        for fname in files:
            if fname.lower().endswith(".csv"):
                csv_path = os.path.join(root, fname)
                try:
                    with open(csv_path, "r") as f:
                        reader = csv.DictReader(f)
                        fieldnames = reader.fieldnames or []
                        label_col = None
                        id_col = None
                        for col in fieldnames:
                            if col.lower() in ("label", "amd", "category", "class", "target"):
                                label_col = col
                            if col.lower() in ("id", "image", "filename", "image_id", "imgname"):
                                id_col = col
                        if label_col and id_col:
                            print(f"Found candidate CSV: {csv_path} (label column: '{label_col}', id column: '{id_col}')")
                            for row in reader:
                                val = str(row.get(label_col, "")).strip().lower()
                                if val in ("1", "amd", "true", "yes"):
                                    candidates.append((root, row[id_col].strip()))
                except Exception as e:
                    print(f"  Could not parse {csv_path}: {e}")

    return candidates


def build_existing_hashes():
    hashes = set()
    if not os.path.isdir(AMD_DEST_DIR):
        return hashes
    for fname in os.listdir(AMD_DEST_DIR):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            try:
                img = Image.open(os.path.join(AMD_DEST_DIR, fname))
                hashes.add(str(imagehash.phash(img)))
            except Exception:
                continue
    return hashes


def main():
    try:
        import imagehash  # noqa
    except ImportError:
        print("Missing dependency. Run: pip install imagehash --break-system-packages")
        return

    download_dataset()
    inspect_download()

    print("\nSearching for AMD-positive labels (CSV-based)...")
    candidates = find_amd_positive_images()

    if not candidates:
        print("\nCould not automatically identify AMD-positive images via CSV.")
        print("Look at the folder structure printed above and check for:")
        print("  - A folder literally named 'AMD' (as opposed to 'Non-AMD' or 'Normal')")
        print("  - A differently-named label column in any CSV file found")
        print(f"\nInspect it yourself at: {DOWNLOAD_DIR}")
        print("Then tell me what you find and I'll adjust this script.")
        return

    print(f"Found {len(candidates)} candidate AMD-positive entries.")

    os.makedirs(AMD_DEST_DIR, exist_ok=True)
    existing_hashes = build_existing_hashes()
    print(f"Existing real+synthetic images already in AMD folder: {len(existing_hashes)}")

    n_added = 0
    n_duplicate = 0
    n_missing = 0

    for source_dir, image_id in candidates:
        if n_added >= N_TO_ADD:
            break

        found = False
        for ext in IMAGE_EXTENSIONS:
            src_path = os.path.join(source_dir, image_id + ext)
            if not os.path.exists(src_path):
                # Try walking the whole download dir for this filename as a fallback
                for root, _, files in os.walk(DOWNLOAD_DIR):
                    if image_id + ext in files:
                        src_path = os.path.join(root, image_id + ext)
                        break
            if os.path.exists(src_path):
                found = True
                try:
                    img = Image.open(src_path).convert("RGB")
                    img_hash = str(imagehash.phash(img))
                except Exception:
                    continue

                if img_hash in existing_hashes:
                    n_duplicate += 1
                    continue

                existing_hashes.add(img_hash)
                dst_fname = f"adam_amd_{n_added:04d}.jpg"
                img.save(os.path.join(AMD_DEST_DIR, dst_fname), "JPEG")
                n_added += 1
                break

        if not found:
            n_missing += 1

    print(f"\nAdded: {n_added} new real AMD images")
    print(f"Skipped as duplicates: {n_duplicate}")
    print(f"Files listed but not found on disk: {n_missing}")
    print(f"\nYour AMD folder should now have roughly {300 + n_added} images "
          f"(300 existing + {n_added} newly added).")
    print(f"You can delete the {DOWNLOAD_DIR} folder afterward to save disk space.")


if __name__ == "__main__":
    main()