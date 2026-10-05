"""
Copy real AMD-positive images from the already-downloaded ADAM dataset.

Confirmed folder structure (from actually looking at the extracted files):
  _adam_download/ADAM/Training400/AMD/A0001.jpg, A0002.jpg, ...

This is simpler than originally expected - no CSV parsing needed, just a
straightforward folder of real AMD-positive fundus photos. This version
replaces the earlier CSV-guessing logic in download_amd_boost.py.

Adds directly into data/cataract/amd/ (back it up first if you haven't -
Copy-Item -Path "data\\cataract\\amd" -Destination "data\\cataract\\amd_backup_300" -Recurse
"""

import os

from PIL import Image
import imagehash

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
AMD_SOURCE_DIR = os.path.join(PROJECT_ROOT, "_adam_download", "ADAM", "Training400", "AMD")

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract")
AMD_DEST_DIR = os.path.join(DATA_ROOT, "amd")

N_TO_ADD = 100
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_images(folder):
    return [
        f for f in os.listdir(folder)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]


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
    if not os.path.isdir(AMD_SOURCE_DIR):
        print(f"Could not find: {AMD_SOURCE_DIR}")
        print("Check that _adam_download/ADAM/Training400/AMD/ actually exists.")
        return

    source_images = list_images(AMD_SOURCE_DIR)
    print(f"Found {len(source_images)} real AMD-positive images in the ADAM dataset.")

    os.makedirs(AMD_DEST_DIR, exist_ok=True)
    existing_hashes = build_existing_hashes()
    print(f"Existing images already in your AMD folder: {len(existing_hashes)}")

    n_added = 0
    n_duplicate = 0
    n_error = 0

    for fname in source_images:
        if n_added >= N_TO_ADD:
            break

        src_path = os.path.join(AMD_SOURCE_DIR, fname)
        try:
            img = Image.open(src_path).convert("RGB")
            img_hash = str(imagehash.phash(img))
        except Exception as e:
            n_error += 1
            continue

        if img_hash in existing_hashes:
            n_duplicate += 1
            continue

        existing_hashes.add(img_hash)
        dst_fname = f"adam_amd_{n_added:04d}.jpg"
        img.save(os.path.join(AMD_DEST_DIR, dst_fname), "JPEG")
        n_added += 1

    print(f"\nAdded: {n_added} new real AMD images")
    print(f"Skipped as duplicates: {n_duplicate}")
    if n_error > 0:
        print(f"Unreadable files skipped: {n_error}")

    final_count = len(list_images(AMD_DEST_DIR))
    print(f"\nYour AMD folder now has: {final_count} images total.")
    print(f"You can delete the _adam_download folder afterward to save disk space.")


if __name__ == "__main__":
    main()