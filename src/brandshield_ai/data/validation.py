from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ("Fake", "Real")
SPLITS = ("train", "val")


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_classification_dataset(images_root: Path) -> dict:
    report = {"root": str(images_root), "splits": {}, "errors": []}
    for split in SPLITS:
        report["splits"][split] = {}
        for class_name in CLASS_NAMES:
            folder = images_root / split / class_name
            count = count_images(folder)
            report["splits"][split][class_name] = count
            if count == 0:
                report["errors"].append(f"Missing images for {split}/{class_name}: {folder}")
    return report


def detect_duplicate_label_images(raw_root: Path) -> list[str]:
    issues = []
    images_root = raw_root / "images"
    labels_root = raw_root / "labels"
    if not labels_root.exists():
        return issues
    for split in SPLITS:
        for class_name in CLASS_NAMES:
            label_dir = labels_root / split / class_name
            image_dir = images_root / split / class_name
            if not label_dir.exists():
                continue
            jpg_labels = [p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
            txt_labels = [p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
            if jpg_labels and not txt_labels:
                issues.append(
                    f"{label_dir} contains image files, not YOLO .txt labels. "
                    f"This looks duplicated from {image_dir}."
                )
    return issues
