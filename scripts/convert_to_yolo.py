from pathlib import Path
import shutil
from tqdm import tqdm

# =====================================================
# Paths
# =====================================================

ROOT = Path(__file__).resolve().parent.parent

RAW_IMAGES = ROOT / "dataset" / "raw" / "images"
RAW_LABELS = ROOT / "dataset" / "raw" / "labels"

YOLO_IMAGES = ROOT / "dataset" / "detection_dataset" / "images"
YOLO_LABELS = ROOT / "dataset" / "detection_dataset" / "labels"

# =====================================================
# Create folders
# =====================================================

for split in ["train", "val"]:

    (YOLO_IMAGES / split).mkdir(parents=True, exist_ok=True)
    (YOLO_LABELS / split).mkdir(parents=True, exist_ok=True)

# =====================================================
# Copy Function
# =====================================================

def copy_split(split):

    copied = 0
    missing = 0

    for cls in ["Fake", "Real"]:

        image_folder = RAW_IMAGES / split / cls
        label_folder = RAW_LABELS / split / cls

        for image in tqdm(list(image_folder.glob("*.*")),
                          desc=f"{split}-{cls}"):

            stem = image.stem

            label = label_folder / f"{stem}.txt"

            # create unique filename

            new_name = f"{cls}_{image.name}"

            shutil.copy2(
                image,
                YOLO_IMAGES / split / new_name
            )

            if label.exists():

                shutil.copy2(
                    label,
                    YOLO_LABELS / split / f"{cls}_{label.name}"
                )

                copied += 1

            else:

                missing += 1

    return copied, missing

# =====================================================
# Run
# =====================================================

train_ok, train_missing = copy_split("train")
val_ok, val_missing = copy_split("val")

print("\n==============================")
print("Dataset Conversion Complete")
print("==============================")

print(f"Train labels copied : {train_ok}")
print(f"Train missing labels: {train_missing}")

print(f"Val labels copied   : {val_ok}")
print(f"Val missing labels  : {val_missing}")