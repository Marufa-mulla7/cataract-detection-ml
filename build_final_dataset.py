import os
import csv
import random
import hashlib
import shutil

from PIL import Image
from torchvision import transforms


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SOURCE_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract")
DEST_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract_final")

CLASS_NAMES = [
    "normal",
    "cataract",
    "glaucoma",
    "amd",
    "diabetic_retinopathy",
]

TARGETS = {
    "normal": 900,
    "cataract": 900,
    "glaucoma": 900,
    "amd": 900,
    "diabetic_retinopathy": 1000,
}

# Real validation/test counts.
# AMD has only ~188 independent source cases, so it receives
# 28 real validation + 28 real test cases.
VAL_COUNTS = {
    "normal": 135,
    "cataract": 135,
    "glaucoma": 135,
    "amd": 28,
    "diabetic_retinopathy": 150,
}

TEST_COUNTS = {
    "normal": 135,
    "cataract": 135,
    "glaucoma": 135,
    "amd": 28,
    "diabetic_retinopathy": 150,
}

SEED = 42

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

AUGMENT_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.10,
    ),
])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_real_images(class_dir):
    """
    Return source images only.

    Existing files beginning with aug_ are intentionally excluded because
    they are derived images from earlier dataset versions.
    """
    files = []

    if not os.path.isdir(class_dir):
        raise RuntimeError(f"Missing class directory: {class_dir}")

    for fname in os.listdir(class_dir):
        path = os.path.join(class_dir, fname)

        if not os.path.isfile(path):
            continue

        if not fname.lower().endswith(IMAGE_EXTENSIONS):
            continue

        if fname.lower().startswith("aug_"):
            continue

        files.append(fname)

    return sorted(files)


def prepare_sources(class_name):
    """
    Build a list of unique real source images using SHA-256.

    Exact duplicate files are treated as the same underlying source so
    they cannot accidentally appear in different splits.
    """
    class_dir = os.path.join(SOURCE_ROOT, class_name)
    filenames = list_real_images(class_dir)

    unique = []
    seen_hashes = set()

    for fname in filenames:
        path = os.path.join(class_dir, fname)

        try:
            digest = sha256_file(path)
        except Exception as e:
            print(f"WARNING: cannot hash {path}: {e}")
            continue

        if digest in seen_hashes:
            print(f"  Duplicate ignored: {fname}")
            continue

        seen_hashes.add(digest)

        unique.append({
            "filename": fname,
            "path": path,
            "sha256": digest,
        })

    return unique


def copy_real(source, class_name, split, writer):
    split_dir = os.path.join(DEST_ROOT, split, class_name)
    os.makedirs(split_dir, exist_ok=True)

    destination = os.path.join(split_dir, source["filename"])

    shutil.copy2(source["path"], destination)

    writer.writerow({
        "class": class_name,
        "split": split,
        "filename": source["filename"],
        "sample_type": "real",
        "parent_source": source["filename"],
        "source_sha256": source["sha256"],
        "output_sha256": sha256_file(destination),
    })


def create_augmented(source, class_name, index, writer):
    split_dir = os.path.join(DEST_ROOT, "train", class_name)
    os.makedirs(split_dir, exist_ok=True)

    base_name = os.path.splitext(source["filename"])[0]
    filename = f"aug_{index:04d}_{base_name}.jpg"
    destination = os.path.join(split_dir, filename)

    try:
        with Image.open(source["path"]) as img:
            img = img.convert("RGB")
            augmented = AUGMENT_TRANSFORM(img)
            augmented.save(destination, "JPEG", quality=95)

    except Exception as e:
        print(f"WARNING: augmentation failed for {source['filename']}: {e}")
        return False

    writer.writerow({
        "class": class_name,
        "split": "train",
        "filename": filename,
        "sample_type": "augmented",
        "parent_source": source["filename"],
        "source_sha256": source["sha256"],
        "output_sha256": sha256_file(destination),
    })

    return True


def main():
    random.seed(SEED)

    print("=" * 70)
    print("BUILDING FINAL LEAKAGE-FREE DATASET")
    print("=" * 70)
    print(f"Source      : {SOURCE_ROOT}")
    print(f"Destination : {DEST_ROOT}")
    print(f"Seed        : {SEED}")
    print()

    if os.path.exists(DEST_ROOT):
        raise RuntimeError(
            f"Destination already exists:\n{DEST_ROOT}\n"
            "Delete/rename it only after we inspect it. "
            "This script refuses to overwrite an existing dataset."
        )

    os.makedirs(DEST_ROOT)

    for split in ["train", "val", "test"]:
        for class_name in CLASS_NAMES:
            os.makedirs(
                os.path.join(DEST_ROOT, split, class_name),
                exist_ok=True,
            )

    manifest_path = os.path.join(DEST_ROOT, "manifest.csv")

    summary = {}

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as manifest_file:

        writer = csv.DictWriter(
            manifest_file,
            fieldnames=[
                "class",
                "split",
                "filename",
                "sample_type",
                "parent_source",
                "source_sha256",
                "output_sha256",
            ],
        )

        writer.writeheader()

        for class_name in CLASS_NAMES:

            print("-" * 70)
            print(f"CLASS: {class_name}")

            sources = prepare_sources(class_name)
            random.shuffle(sources)

            n_sources = len(sources)
            val_count = VAL_COUNTS[class_name]
            test_count = TEST_COUNTS[class_name]
            target = TARGETS[class_name]

            if n_sources < val_count + test_count:
                raise RuntimeError(
                    f"{class_name}: only {n_sources} unique real sources, "
                    f"but {val_count + test_count} are required for "
                    "validation/test."
                )

            train_real_sources = n_sources - val_count - test_count
            train_target = target - val_count - test_count
            synthetic_needed = train_target - train_real_sources

            if synthetic_needed < 0:
                raise RuntimeError(
                    f"{class_name}: {n_sources} real sources exceed the "
                    f"{target}-image target after reserving val/test."
                )

            val_sources = sources[:val_count]
            test_sources = sources[val_count:val_count + test_count]
            train_sources = sources[val_count + test_count:]

            print(f"Unique real sources : {n_sources}")
            print(f"Train real          : {len(train_sources)}")
            print(f"Validation real     : {len(val_sources)}")
            print(f"Test real           : {len(test_sources)}")
            print(f"Training augmented  : {synthetic_needed}")
            print(f"Final target        : {target}")

            # Copy real sources into their predetermined split.
            for source in train_sources:
                copy_real(source, class_name, "train", writer)

            for source in val_sources:
                copy_real(source, class_name, "val", writer)

            for source in test_sources:
                copy_real(source, class_name, "test", writer)

            # Generate synthetic samples ONLY from training sources.
            generated = 0
            attempts = 0
            max_attempts = max(1000, synthetic_needed * 10)

            while generated < synthetic_needed:

                if attempts >= max_attempts:
                    raise RuntimeError(
                        f"{class_name}: could not generate enough "
                        "training augmentations."
                    )

                parent = random.choice(train_sources)

                success = create_augmented(
                    parent,
                    class_name,
                    generated,
                    writer,
                )

                attempts += 1

                if success:
                    generated += 1

            summary[class_name] = {
                "target": target,
                "real_total": n_sources,
                "train_real": len(train_sources),
                "train_augmented": generated,
                "val_real": len(val_sources),
                "test_real": len(test_sources),
            }

    print()
    print("=" * 70)
    print("FINAL DATASET SUMMARY")
    print("=" * 70)

    grand_total = 0

    for class_name in CLASS_NAMES:
        s = summary[class_name]

        train_total = s["train_real"] + s["train_augmented"]
        val_total = s["val_real"]
        test_total = s["test_real"]
        total = train_total + val_total + test_total

        grand_total += total

        print(
            f"{class_name:22s} "
            f"train={train_total:4d} "
            f"val={val_total:3d} "
            f"test={test_total:3d} "
            f"TOTAL={total:4d}"
        )

    print("-" * 70)
    print(f"{'TOTAL':22s} {grand_total:4d} images")
    print()
    print(f"Dataset created at:")
    print(DEST_ROOT)
    print()
    print(f"Manifest:")
    print(manifest_path)
    print()
    print("IMPORTANT:")
    print("- Validation/test contain REAL images only.")
    print("- Augmented images exist only in training.")
    print("- Parent sources are recorded in manifest.csv.")
    print("- Original data/cataract/ was not modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
