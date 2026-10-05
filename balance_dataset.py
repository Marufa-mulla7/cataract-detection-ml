import os
import random

from PIL import Image, ImageEnhance

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "cataract")

TARGET_COUNT = 300
RANDOM_SEED = 42
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")


def list_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(VALID_EXTENSIONS)]


def augment_image(image):
    img = image.copy()
    angle = random.uniform(-20, 20)
    img = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    brightness_factor = random.uniform(0.85, 1.15)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)
    contrast_factor = random.uniform(0.85, 1.15)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)
    return img


def upsample_class(class_dir, class_name, current_files, target):
    random.seed(RANDOM_SEED)
    needed = target - len(current_files)
    generated = 0
    attempt = 0
    while generated < needed:
        source_fname = current_files[attempt % len(current_files)]
        source_path = os.path.join(class_dir, source_fname)
        try:
            with Image.open(source_path).convert("RGB") as img:
                augmented = augment_image(img)
                base_name = os.path.splitext(source_fname)[0]
                new_fname = f"aug_{generated}_{base_name}.jpg"
                new_path = os.path.join(class_dir, new_fname)
                augmented.save(new_path, "JPEG", quality=95)
                generated += 1
        except Exception as e:
            print(f"    Warning: failed to augment {source_fname}: {e}")
        attempt += 1
    print(f"  {class_name}: generated {generated} synthetic augmented images -> now {target}")


def balance_class(class_name):
    class_dir = os.path.join(DATA_ROOT, class_name)
    if not os.path.isdir(class_dir):
        print(f"  {class_name}: folder not found, skipping.")
        return
    current_files = list_images(class_dir)
    current_count = len(current_files)
    print(f"\n{class_name}: currently {current_count} images")
    if current_count < TARGET_COUNT:
        upsample_class(class_dir, class_name, current_files, TARGET_COUNT)
    else:
        print(f"  {class_name}: already at or above target ({TARGET_COUNT}), leaving untouched.")


def main():
    print(f"Topping up any class below {TARGET_COUNT} images in {DATA_ROOT}.")
    print(f"Classes at or above target are left untouched. Nothing is deleted.\n")
    class_names = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    for class_name in sorted(class_names):
        balance_class(class_name)
    print("\nDone. Final counts:")
    for class_name in sorted(class_names):
        class_dir = os.path.join(DATA_ROOT, class_name)
        if os.path.isdir(class_dir):
            print(f"  {class_name}: {len(list_images(class_dir))}")


if __name__ == "__main__":
    main()
