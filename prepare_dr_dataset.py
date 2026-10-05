import os
import shutil
import subprocess
import sys
import zipfile

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "_aptos_download")
DEST_DIR = os.path.join(PROJECT_ROOT, "data", "cataract", "diabetic_retinopathy")
COMPETITION = "aptos2019-blindness-detection"


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
    else:
        print("ERROR: kaggle.json not found in project root.")
        sys.exit(1)


def download_dataset():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    zip_path = os.path.join(DOWNLOAD_DIR, f"{COMPETITION}.zip")
    if os.path.exists(zip_path):
        print("Zip already downloaded, skipping download step.")
        return zip_path
    print(f"Downloading {COMPETITION} from Kaggle (this may take a while)...")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", DOWNLOAD_DIR],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    print("Download complete.")
    return zip_path


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


def collapse_dr_positive_images(extract_dir):
    csv_path = os.path.join(extract_dir, "train.csv")
    images_dir = os.path.join(extract_dir, "train_images")
    if not os.path.exists(csv_path):
        print(f"ERROR: could not find train.csv at {csv_path}")
        sys.exit(1)
    if not os.path.isdir(images_dir):
        print(f"ERROR: could not find train_images/ at {images_dir}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    dr_positive = df[df["diagnosis"] > 0]
    print(f"Total images in dataset: {len(df)}")
    print(f"DR-positive images (severity 1-4): {len(dr_positive)}")
    print(f"Skipping {len(df) - len(dr_positive)} 'No DR' images.")
    os.makedirs(DEST_DIR, exist_ok=True)
    copied = 0
    missing = 0
    for _, row in dr_positive.iterrows():
        image_id = row["id_code"]
        src_path = os.path.join(images_dir, f"{image_id}.png")
        if not os.path.exists(src_path):
            missing += 1
            continue
        dst_path = os.path.join(DEST_DIR, f"{image_id}.png")
        shutil.copy(src_path, dst_path)
        copied += 1
    print(f"\nCopied {copied} DR-positive images to: {DEST_DIR}")
    if missing:
        print(f"Warning: {missing} referenced images were not found on disk.")


def main():
    ensure_kaggle_credentials()
    zip_path = download_dataset()
    extract_dir = extract_dataset(zip_path)
    collapse_dr_positive_images(extract_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
