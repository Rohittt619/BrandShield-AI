from pathlib import Path
import shutil

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent

RAW = ROOT / "dataset" / "raw"

YOLO = ROOT / "dataset" / "detection_dataset"

CLASSIFIER = ROOT / "dataset" / "classification_dataset"

# -----------------------------
# Create folders
# -----------------------------

for folder in [

    YOLO / "images" / "train",
    YOLO / "images" / "val",

    YOLO / "labels" / "train",
    YOLO / "labels" / "val",

    CLASSIFIER / "train" / "Fake",
    CLASSIFIER / "train" / "Real",

    CLASSIFIER / "val" / "Fake",
    CLASSIFIER / "val" / "Real",

]:
    folder.mkdir(parents=True, exist_ok=True)

print("Folders Created ✔")