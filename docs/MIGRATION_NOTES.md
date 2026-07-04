# Migration Notes: FakeLogoDetection to BrandShield-AI

## Old Project

The graduation project, `FakeLogoDetection`, used a two-stage idea:

1. YOLOv5 detects/localizes the logo.
2. A Keras CNN classifier predicts Real or Fake.

It also included webcam inference, sound feedback, and basic brand label mapping.

## Findings During Upgrade

- The old folder contains a copied YOLOv5 repository and generated/runtime files.
- Several important files were accidentally saved as folders, for example `data.yaml`, `label_names.txt`, and `detect_and_classify.py`.
- The dataset has Real/Fake image splits: 439 Fake train, 220 Real train, 110 Fake val, 55 Real val.
- The `labels` folders contain `.jpg` images, not YOLO `.txt` annotation files.
- Therefore, the current dataset supports image classification, but not YOLO detector training.

## BrandShield-AI Upgrade Strategy

Phase 1 builds a clean Real/Fake classifier with metrics and inference.
Phase 2 adds YOLO detection only after proper bounding-box labels are created or imported.
Phase 3 adds explainability, reporting, UI, and deployment.
