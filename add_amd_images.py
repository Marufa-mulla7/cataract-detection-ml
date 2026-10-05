import os
import shutil
import subprocess
import sys
import zipfile

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "_rfmid_download")
DEST_DIR = os.path.join(PROJECT_ROOT, "data", "cataract", "amd")
DATASET = "andrewmvd/retinal-disease-classification"


def ensure_kaggle_credentials():
    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    kaggle_json_target = os.path.join(kaggle_dir, "kaggle.json")
    if os.path.exists(kaggle_json_target):
        print("kaggle.json already configured.")
        return
    local_kaggle_json = os.path.join(PROJECT_ROOT, "kaggle.json")
    if os.path.exists(local_kaggle_json):
        os.makedirs(kaggle_dir, exist_ok=True)
        shutil.copy(local_kaggle_json, kaggle_json_target)
        print(f"Copied kaggle.json to {kaggle_json_target}")


def download_dataset():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    existing_zips = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".zip")]
    if existing_zips:
        print(f"Zip already present: {existing_zips[0]}, skipping download step.")
        return os.path.join(DOWNLOAD_DIR, existing_zips[0])
    print(f"Downloading {DATASET} from Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", DOWNLOAD_DIR],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    downloaded_files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith(".zip")]
    return os.path.join(DOWNLOAD_DIR, downloaded_files[0])


def extract_dataset(zip_path):
    extract_dir = os.path.join(DOWNLOAD_DIR, "extracted")
    if os.path.isdir(extract_dir) and os.listdir(extract_dir):
        print("Already extracted, skipping.")
        return extract_dir
    print("Extracting zip file...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("Extraction complete.")
    return extract_dir


def find_training_csv_and_images(extract_dir):
    csv_path = None
    images_dir = None
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.lower().endswith(".csv"):
                candidate = os.path.join(root, f)
                try:
                    df_head = pd.read_csv(candidate, nrows=1)
                    if "ARMD" in df_head.columns and "DR" in df_head.columns:
                        if csv_path is None or "train" in f.lower():
                            csv_path = candidate
                            images_dir = root
                except Exception:
                    continue
    if csv_path is None:
        print("ERROR: could not find training labels CSV. Contents:")
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                print(os.path.join(root, f))
        sys.exit(1)
    parent_dir = os.path.dirname(csv_path)
    for root, dirs, files in os.walk(parent_dir):
        image_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if image_files:
            images_dir = root
            break
    return csv_path, images_dir


def extract_amd_images(df, images_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    selected = df[(df["ARMD"] == 1) & (df["DR"] == 0)]
    print(f"Total images in labels file: {len(df)}")
    print(f"AMD-flagged images (ARMD=1): {(df['ARMD'] == 1).sum()}")
    print(f"Selected (AMD, non-DR): {len(selected)}")
    copied = 0
    missing = 0
    for _, row in selected.iterrows():
        image_id = row["ID"]
        src_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(images_dir, f"{image_id}{ext}")
            if os.path.exists(candidate):
                src_path = candidate
                break
        if src_path is None:
            missing += 1
            continue
        ext = os.path.splitext(src_path)[1]
        dst_filename = f"rfmid_{image_id}{ext}"
        dst_path = os.path.join(dest_dir, dst_filename)
        if os.path.exists(dst_path):
            continue
        shutil.copy(src_path, dst_path)
        copied += 1
    print(f"\nCopied {copied} new amd images to {dest_dir}")
    if missing:
        print(f"Warning: {missing} referenced images were not found on disk.")
    return copied


def main():
    ensure_kaggle_credentials()
    zip_path = download_dataset()
    extract_dir = extract_dataset(zip_path)
    csv_path, images_dir = find_training_csv_and_images(extract_dir)
    print(f"Using CSV: {csv_path}")
    print(f"Using images folder: {images_dir}")
    df = pd.read_csv(csv_path)
    copied = extract_amd_images(df, images_dir, DEST_DIR)
    print("\nDone.")
    print(f"  Added {copied} amd images -> {DEST_DIR}")


if __name__ == "__main__":
    main()
