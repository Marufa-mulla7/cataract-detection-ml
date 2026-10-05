import os
import shutil
import subprocess
import sys
import zipfile

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "_odir_download")
CATARACT_DEST = os.path.join(PROJECT_ROOT, "data", "cataract", "cataract")
GLAUCOMA_DEST = os.path.join(PROJECT_ROOT, "data", "cataract", "glaucoma")
DATASET = "andrewmvd/ocular-disease-recognition-odir5k"

OTHER_DISEASE_KEYWORDS = [
    "diabetic retinopathy", "diabetes", "hypertension", "myopia",
    "macular degeneration", "amd", "drusen", "retinopathy",
    "vessel tortuosity", "epiretinal membrane",
]


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


def find_csv_and_images(extract_dir):
    csv_path = None
    images_dir = None
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.lower() == "full_df.csv":
                csv_path = os.path.join(root, f)
        for d in dirs:
            if d.lower() in ("preprocessed_images", "preprocessed_image"):
                images_dir = os.path.join(root, d)
    if csv_path is None:
        print("ERROR: could not find full_df.csv. Contents found:")
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                print(os.path.join(root, f))
        sys.exit(1)
    if images_dir is None:
        print("ERROR: could not find preprocessed_images folder.")
        sys.exit(1)
    return csv_path, images_dir


def has_only_keyword(text, target_keyword):
    text_lower = text.lower()
    if target_keyword not in text_lower:
        return False
    for other in OTHER_DISEASE_KEYWORDS:
        if other == target_keyword:
            continue
        if other in text_lower:
            return False
    return True


def extract_condition_images(df, images_dir, keyword, dest_dir, label):
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    skipped_missing = 0
    for _, row in df.iterrows():
        for side in ["Left", "Right"]:
            keywords_col = f"{side}-Diagnostic Keywords"
            fundus_col = f"{side}-Fundus"
            if keywords_col not in row or fundus_col not in row:
                continue
            keywords_text = str(row[keywords_col])
            fundus_filename = str(row[fundus_col])
            if not has_only_keyword(keywords_text, keyword):
                continue
            src_path = os.path.join(images_dir, fundus_filename)
            if not os.path.exists(src_path):
                skipped_missing += 1
                continue
            dst_filename = f"odir_{fundus_filename}"
            dst_path = os.path.join(dest_dir, dst_filename)
            if os.path.exists(dst_path):
                continue
            shutil.copy(src_path, dst_path)
            copied += 1
    print(f"\n{label}: copied {copied} new images to {dest_dir}")
    if skipped_missing:
        print(f"{label}: {skipped_missing} referenced images were not found on disk.")
    return copied


def main():
    ensure_kaggle_credentials()
    zip_path = download_dataset()
    extract_dir = extract_dataset(zip_path)
    csv_path, images_dir = find_csv_and_images(extract_dir)
    print(f"Using CSV: {csv_path}")
    print(f"Using images folder: {images_dir}")
    df = pd.read_csv(csv_path)
    print(f"Total patient records: {len(df)}")
    cataract_count = extract_condition_images(df, images_dir, "cataract", CATARACT_DEST, "Cataract")
    glaucoma_count = extract_condition_images(df, images_dir, "glaucoma", GLAUCOMA_DEST, "Glaucoma")
    print("\nDone.")
    print(f"  Added {cataract_count} cataract images -> {CATARACT_DEST}")
    print(f"  Added {glaucoma_count} glaucoma images -> {GLAUCOMA_DEST}")


if __name__ == "__main__":
    main()
