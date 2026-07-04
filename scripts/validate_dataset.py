from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brandshield_ai.data.validation import detect_duplicate_label_images, validate_classification_dataset

raw_root = ROOT / "dataset" / "raw"
images_root = raw_root / "images"

report = validate_classification_dataset(images_root)
print("BrandShield-AI Dataset Validation")
print("=================================")
for split, classes in report["splits"].items():
    for class_name, count in classes.items():
        print(f"{split}/{class_name}: {count} images")

issues = report["errors"] + detect_duplicate_label_images(raw_root)
if issues:
    print("\nIssues:")
    for issue in issues:
        print(f"- {issue}")
    raise SystemExit(1)

print("\nDataset looks ready for Phase 1 classification training.")
