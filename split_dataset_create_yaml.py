from pathlib import Path
import random
import shutil

# Update this path to your dataset's archive folder.
DATASET_DIR = Path(r"C:/Users/ADMIN/OneDrive - Softlabs Technologies and Development Pvt Ltd/Documents/skin_disease/TechieProject/archive")

TRAIN_SOURCE = DATASET_DIR / "train"
TEST_SOURCE = DATASET_DIR / "test"
VAL_DIR = DATASET_DIR / "val"

VAL_RATIO = 0.20
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def images_in(folder: Path):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

def main():
    if not TRAIN_SOURCE.is_dir():
        raise FileNotFoundError(f"Training folder not found: {TRAIN_SOURCE}")
    if not TEST_SOURCE.is_dir():
        raise FileNotFoundError(f"Test folder not found: {TEST_SOURCE}")

    # This script moves validation images out of train, so it will not overwrite an existing val.
    if VAL_DIR.exists() and any(VAL_DIR.iterdir()):
        raise FileExistsError(
            f"{VAL_DIR} already exists and is not empty. Back it up/remove it, "
            "or change VAL_DIR before running this script."
        )

    class_dirs = sorted(p for p in TRAIN_SOURCE.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class folders found inside {TRAIN_SOURCE}")

    random.seed(SEED)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for class_dir in class_dirs:
        class_name = class_dir.name
        class_images = images_in(class_dir)
        if len(class_images) < 2:
            print(f"Skipping {class_name}: only {len(class_images)} image(s)")
            continue

        random.shuffle(class_images)
        val_count = max(1, round(len(class_images) * VAL_RATIO))
        val_images = class_images[:val_count]

        val_class_dir = VAL_DIR / class_name
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for src in val_images:
            # Preserve any nested relative path within the class folder.
            relative = src.relative_to(class_dir)
            dest = val_class_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))

        remaining = len(images_in(class_dir))
        summary.append((class_name, remaining, len(val_images)))

    # Use relative paths so the dataset can be moved with its YAML file.
    class_names = [p.name for p in class_dirs]

    yaml_path = DATASET_DIR / "data.yaml"
    # This YAML is a class-name/path manifest. For YOLO classification, train
    # directly from the folder path; standard YOLO detection YAML also needs labels.
    lines = [
        f"path: {DATASET_DIR.resolve().as_posix()!r}",
        "train: train",
        "val: val",
        "test: test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        safe_name = name.replace("'", "''")
        lines.append(f"  {i}: '{safe_name}'")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nSplit complete.")
    print(f"Dataset: {DATASET_DIR}")
    print(f"YAML:    {yaml_path}")
    print("\nPer-class image counts (train / val):")
    for name, train_n, val_n in summary:
        print(f"{name}: {train_n} / {val_n}")
    print("\nTest images were not changed.")

if __name__ == "__main__":
    main()
