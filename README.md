# BrandShield-AI

BrandShield-AI is a professional upgrade of the original graduation project, **Fake Logo Detection Using Python**. The project focuses on brand authenticity verification using computer vision and deep learning.

## Project Direction

The original project combined YOLOv5 logo localization, CNN-based Real/Fake classification, webcam inference, sound alerts, and basic brand mapping. During migration, the available dataset was verified and found to contain Real/Fake image folders, but no YOLO bounding-box annotation files. Because of that, BrandShield-AI is structured in phases:

- **Phase 1:** Train and deploy a strong Real/Fake logo classifier using the existing dataset.
- **Phase 2:** Add YOLO-based logo detection after proper bounding-box labels or a real detection dataset are available.
- **Phase 3:** Add Grad-CAM explainability, PDF reports, Streamlit UI, and API deployment.

## Current Repository Structure

```text
BrandShield-AI/
  src/brandshield_ai/        Core Python package
  scripts/                   CLI scripts for dataset checks, training, inference
  dataset/                   Local datasets, ignored by Git
  models/                    Local trained models, ignored by Git
  outputs/                   Local predictions/reports, ignored by Git
  docs/                      Project notes and documentation
  tests/                     Automated tests
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Layout For Phase 1

Place the Real/Fake dataset like this:

```text
dataset/raw/images/train/Fake/*.jpg
dataset/raw/images/train/Real/*.jpg
dataset/raw/images/val/Fake/*.jpg
dataset/raw/images/val/Real/*.jpg
```

Then validate it:

```bash
python scripts/validate_dataset.py
```

## Train Classifier

```bash
python scripts/train_classifier.py --data dataset/raw/images --model-out models/brandshield_classifier.h5 --epochs 20 --fine-tune-epochs 5
```

## Run Inference

```bash
python scripts/inference.py --model models/brandshield_classifier.h5 --image path\to\logo.jpg
```

## Important Dataset Note

The previous `labels` folders contained `.jpg` files duplicated from the image folders, not YOLO `.txt` labels. This means the current data can train a classifier, but it cannot train YOLO detection until bounding-box annotations are created.

## Evaluate Classifier\n\n```bash\npython scripts/evaluate_classifier.py --model models/brandshield_classifier.h5 --data dataset/raw/images/val\n```\n\nThis writes metrics, predictions, and a confusion matrix under `outputs/evaluation/`.\n\nTune the classifier threshold after evaluation:\n\n```bash\npython scripts/tune_threshold.py --predictions outputs/evaluation/predictions.csv --metric balanced_accuracy\n```\n\n## Roadmap

- Dataset validation and migration tooling
- CNN/EfficientNet classifier training
- Confusion matrix and metrics report
- Single-image inference
- Streamlit dashboard
- Grad-CAM visual explanations
- PDF authenticity report generation
- YOLO detector integration with real bounding-box annotations
