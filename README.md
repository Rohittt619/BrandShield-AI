# BrandShield-AI

BrandShield-AI is a professional upgrade of the original graduation project, Fake Logo Detection Using Python. The project focuses on brand authenticity verification using computer vision and deep learning.

## Project Direction

The original project combined YOLOv5 logo localization, CNN-based Real/Fake classification, webcam inference, sound alerts, and basic brand mapping. During migration, the available dataset was verified and found to contain Real/Fake image folders, but no YOLO bounding-box annotation files. Because of that, BrandShield-AI is structured in phases:

**Findings during migration:**
* The old project folder contained a copied YOLOv5 repository plus generated/runtime files, and several key files (`data.yaml`, `label_names.txt`, `detect_and_classify.py`) had accidentally been saved as folders instead of files.
* The dataset has Real/Fake image splits: 439 Fake train, 220 Real train, 110 Fake val, 55 Real val.
* The `labels` folders contained `.jpg` images, not YOLO `.txt` annotation files — meaning the dataset supports classification but not detector training as-is.

This is why the upgrade was split into phases rather than porting the old two-stage pipeline directly:

* **Phase 1:** Train and deploy a strong Real/Fake logo classifier using the existing dataset.
* **Phase 2:** Add YOLO-based logo detection after proper bounding-box labels or a real detection dataset are available.
* **Phase 3:** Add Grad-CAM explainability, PDF reports, Streamlit UI, and API deployment.

## Current Repository Structure

```
BrandShield-AI/
  src/brandshield_ai/        Core Python package
  scripts/                   CLI scripts for dataset checks, training, inference, and explainability
  dataset/                   Local datasets, ignored by Git
  models/                    Local trained models, ignored by Git
  outputs/                   Local predictions/reports, ignored by Git
  docs/                      Project notes and documentation
  tests/                     Automated tests
```

## Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Layout For Phase 1

Place the Real/Fake dataset like this:

```
dataset/raw/images/train/Fake/*.jpg
dataset/raw/images/train/Real/*.jpg
dataset/raw/images/val/Fake/*.jpg
dataset/raw/images/val/Real/*.jpg
```

Then validate it:

```
python scripts/validate_dataset.py
```

## Train Classifier

```
python scripts/train_classifier.py --data dataset/raw/images --model-out models/brandshield_classifier.h5 --epochs 20 --fine-tune-epochs 5
```

## Run Inference

```
python scripts/inference.py --model models/brandshield_classifier.h5 --image path\to\logo.jpg
```

## Important Dataset Note

The previous `labels` folders contained `.jpg` files duplicated from the image folders, not YOLO `.txt` labels. This means the current data can train a classifier, but it cannot train YOLO detection until bounding-box annotations are created.

## Evaluate Classifier

```
python scripts/evaluate_classifier.py --model models/brandshield_classifier.h5 --data dataset/raw/images/val
```

This writes metrics, predictions, and a confusion matrix under `outputs/evaluation/`.

Tune the classifier threshold after evaluation:

```
python scripts/tune_threshold.py --predictions outputs/evaluation/predictions.csv --metric balanced_accuracy
```

## Explainability (Grad-CAM)

Generate a Grad-CAM heatmap showing which regions of an image drove the model's Real/Fake prediction:

```
python scripts/gradcam.py --model models/brandshield_classifier.h5 --image path\to\logo.jpg
```

This saves an overlay image to `outputs/evaluation/gradcam_overlay.jpg` by default (use `--output` to change the path).

## Generate Evaluation Report

Build a Markdown report from the real evaluation outputs (metrics + confusion matrix):

```
python scripts/generate_report.py --model models/brandshield_classifier.h5
```

## Results

Fill this in with your actual numbers from `outputs/evaluation/metrics.txt` after running `evaluate_classifier.py` — accuracy, precision/recall, and confusion matrix on the validation set. Do not use placeholder or invented figures here.

## Roadmap

* Dataset validation and migration tooling
* CNN/EfficientNet classifier training
* Confusion matrix and metrics report
* Single-image inference
* Grad-CAM visual explanations
* Streamlit dashboard
* PDF authenticity report generation
* YOLO detector integration with real bounding-box annotations

## Author

**Rohit Rathod**
- GitHub: [@Rohittt619](https://github.com/Rohittt619)
- LinkedIn: [rohit-rathod-19442a228](https://www.linkedin.com/in/rohit-rathod-19442a228)

Feel free to connect, raise an issue, or open a PR if you'd like to contribute or discuss the project.
