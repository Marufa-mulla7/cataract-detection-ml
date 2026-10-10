"""
Cross-class duplicate check.

The earlier de-duplication scripts only checked for duplicates WITHIN the
same class (e.g., a new Cataract candidate against existing Cataract
images). This checks something more important: whether the SAME or a
near-identical image accidentally ended up in TWO DIFFERENT classes -
a real risk given how many different source datasets were combined
(original base, ODIR-5K, eye_diseases_classification, ADAM, APTOS).

If the same photo is labeled "cataract" in one folder and "glaucoma" in
another, that's a genuine data integrity problem worth catching before
training - it directly confuses the model with contradictory labels for
the same input, and could leak information between train/test splits if
the two copies land in different splits.

This does NOT modify any files - it only reports what it finds.
"""

import os

from PIL import Image
import imagehash

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract")

CLASS_NAMES = ["normal", "cataract", "glaucoma", "amd", "diabetic_retinopathy"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Perceptual hashes that differ by this many bits or fewer are considered
# "the same image". Fundus/eye photos generally share a similar visual
# structure (circular shape, dark background, similar color palette),
# so a loose tolerance produces false positives - genuine duplicates
# (the same photo, just recompressed or lightly resized) almost always
# land at distance 0-2. This is intentionally strict.
HAMMING_THRESHOLD = 2


def hash_class_folder(class_name):
    folder = os.path.join(DATA_ROOT, class_name)
    hashes = {}  # hash_string -> filename
    if not os.path.isdir(folder):
        return hashes
    for fname in os.listdir(folder):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            try:
                img = Image.open(os.path.join(folder, fname))
                h = imagehash.phash(img)
                hashes[fname] = h
            except Exception:
                continue
    return hashes


def main():
    print("Hashing every image in every class (this may take a few minutes)...\n")

    class_hashes = {}
    for class_name in CLASS_NAMES:
        hashes = hash_class_folder(class_name)
        class_hashes[class_name] = hashes
        print(f"  {class_name}: {len(hashes)} images hashed")

    print(f"\nChecking for cross-class matches (Hamming distance <= {HAMMING_THRESHOLD})...\n")

    total_matches = 0
    match_summary = {}

    class_list = list(CLASS_NAMES)
    for i in range(len(class_list)):
        for j in range(i + 1, len(class_list)):
            class_a = class_list[i]
            class_b = class_list[j]

            matches = []
            for fname_a, hash_a in class_hashes[class_a].items():
                for fname_b, hash_b in class_hashes[class_b].items():
                    if hash_a - hash_b <= HAMMING_THRESHOLD:
                        matches.append((fname_a, fname_b, hash_a - hash_b))

            if matches:
                match_summary[(class_a, class_b)] = matches
                total_matches += len(matches)

    print(f"{'=' * 60}")
    print("CROSS-CLASS DUPLICATE CHECK RESULTS")
    print(f"{'=' * 60}")

    if total_matches == 0:
        print("No cross-class duplicates found. Every class's images appear")
        print("to be genuinely distinct from every other class. Safe to proceed.")
    else:
        print(f"Found {total_matches} potential cross-class match(es):\n")
        for (class_a, class_b), matches in match_summary.items():
            print(f"  {class_a} <-> {class_b}: {len(matches)} match(es)")
            for fname_a, fname_b, distance in matches[:5]:  # show first 5 per pair
                print(f"    {fname_a}  ~=  {fname_b}  (distance: {distance})")
            if len(matches) > 5:
                print(f"    ... and {len(matches) - 5} more")
        print("\nRECOMMENDATION: open a few of these flagged file pairs and look at them")
        print("side by side. If they're genuinely the same photo, one copy should be")
        print("removed before training - having it in two classes teaches the model")
        print("contradictory labels for the same input.")


if __name__ == "__main__":
    main()