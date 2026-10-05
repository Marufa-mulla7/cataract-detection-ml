"""
Download and integrate real images from the eye_diseases_classification
Kaggle dataset (gunavenkatdoddi/eye-diseases-classification) to grow
Normal, Cataract, and Glaucoma with genuinely new real photos.

Requires: your Kaggle API credentials already set up (kaggle.json), same
as your project's other add_*.py scripts used.

IMPORTANT - de-duplication: this dataset partly sources from ODIR-5K, which
your project already used to boost Cataract/Glaucoma. To avoid silently
double-counting the same photo under two different filenames, this script
computes a perceptual hash of every existing image in your dataset, then
skips any new image whose hash matches something you already have.

This ADDS DIRECTLY into your existing data/cataract/{normal,cataract,glaucoma}/
folders (does not build a separate copy) - back these up first if you want
an undo option, same as with extract_aptos_normal.py.
"""

import os
import shutil
import subprocess
import zipfile

from PIL import Image
import imagehash

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "_eye_diseases_download")
KAGGLE_DATASET = "gunavenkatdoddi/eye-diseases-classification"

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract")

# How many NEW real images to add per class (after de-duplication)
TARGETS_TO_ADD = {
    "normal": 600,     # 300 existing -> aim for 900
    "cataract": 518,   # 382 existing -> aim for 900
    "glaucoma": 560,   # 340 existing -> aim for 900
}

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
        print("\nCheck that kaggle.json is set up correctly (same as your other add_*.py scripts).")
        return

    for fname in os.listdir(DOWNLOAD_DIR):
        if fname.endswith(".zip"):
            zip_path = os.path.join(DOWNLOAD_DIR, fname)
            print(f"Extracting {fname}...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(DOWNLOAD_DIR)


def find_class_folders():
    """
    This dataset's exact internal folder layout isn't confirmed - most
    Kaggle image-classification datasets use class-named subfolders
    (e.g. a 'cataract' folder, a 'normal' folder, etc). This searches
    for folders whose name contains each target class name, anywhere
    in the extracted download.
    """
    found = {}
    for root, dirs, files in os.walk(DOWNLOAD_DIR):
        folder_name = os.path.basename(root).lower()
        for class_name in TARGETS_TO_ADD:
            search_terms = {
                "normal": ["normal"],
                "cataract": ["cataract"],
                "glaucoma": ["glaucoma"],
            }[class_name]
            if any(term in folder_name for term in search_terms):
                image_files = [f for f in files if f.lower().endswith(IMAGE_EXTENSIONS)]
                if image_files:
                    found.setdefault(class_name, []).append((root, image_files))
    return found


def build_existing_hashes(class_name):
    """Hash every image already in this class, so we can detect duplicates."""
    class_dir = os.path.join(DATA_ROOT, class_name)
    hashes = set()
    if not os.path.isdir(class_dir):
        return hashes
    for fname in os.listdir(class_dir):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            try:
                img = Image.open(os.path.join(class_dir, fname))
                hashes.add(str(imagehash.phash(img)))
            except Exception:
                continue
    return hashes


def main():
    print("Checking for required 'imagehash' package...")
    try:
        import imagehash  # noqa
    except ImportError:
        print("Missing dependency. Run: pip install imagehash --break-system-packages")
        return

    download_dataset()

    print("\nSearching downloaded files for class folders...")
    class_folders = find_class_folders()

    for class_name in TARGETS_TO_ADD:
        if class_name not in class_folders:
            print(f"\nWARNING: could not automatically find a '{class_name}' folder in the download.")
            print("Open this folder yourself to check its actual structure:")
            print(f"  {DOWNLOAD_DIR}")
            continue

        print(f"\n--- {class_name} ---")
        existing_hashes = build_existing_hashes(class_name)
        print(f"  Existing real images already in your dataset: {len(existing_hashes)}")

        target_to_add = TARGETS_TO_ADD[class_name]
        dest_dir = os.path.join(DATA_ROOT, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        n_added = 0
        n_duplicate = 0
        n_error = 0

        for source_dir, image_files in class_folders[class_name]:
            for fname in image_files:
                if n_added >= target_to_add:
                    break
                src_path = os.path.join(source_dir, fname)
                try:
                    img = Image.open(src_path).convert("RGB")
                    img_hash = str(imagehash.phash(img))
                except Exception:
                    n_error += 1
                    continue

                if img_hash in existing_hashes:
                    n_duplicate += 1
                    continue

                existing_hashes.add(img_hash)
                dst_fname = f"eyedis_{class_name}_{n_added:04d}.jpg"
                img.save(os.path.join(dest_dir, dst_fname), "JPEG")
                n_added += 1

        print(f"  Added: {n_added} new real images")
        print(f"  Skipped as duplicates: {n_duplicate}")
        if n_error > 0:
            print(f"  Unreadable files skipped: {n_error}")
        if n_added < target_to_add:
            print(f"  NOTE: wanted {target_to_add} but only found {n_added} usable unique images.")

    print("\nDone. Check data/cataract/{normal,cataract,glaucoma}/ for the new counts.")
    print(f"You can delete the {DOWNLOAD_DIR} folder afterward to save disk space.")


if __name__ == "__main__":
    main()