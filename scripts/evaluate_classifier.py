import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


CLASS_NAMES = ["Fake", "Real"]


def build_dataset(data_root: Path, image_size: tuple[int, int], batch_size: int) -> tf.data.Dataset:
    return tf.keras.utils.image_dataset_from_directory(
        data_root,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )


def save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("BrandShield-AI Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_predictions_csv(paths: list[str], y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "actual", "predicted", "real_score", "confidence"])
        for path, actual, score, predicted in zip(paths, y_true, scores, y_pred):
            confidence = score if predicted == 1 else 1 - score
            writer.writerow([
                path,
                CLASS_NAMES[int(actual)],
                CLASS_NAMES[int(predicted)],
                f"{score:.6f}",
                f"{confidence:.6f}",
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the BrandShield-AI Real/Fake classifier.")
    parser.add_argument("--model", default="models/brandshield_classifier.h5")
    parser.add_argument("--data", default="dataset/raw/images/val", help="Validation folder with Fake/Real subfolders.")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    model_path = Path(args.model)
    data_root = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Validation data not found: {data_root}")

    model = tf.keras.models.load_model(model_path)
    dataset = build_dataset(data_root, (args.image_size, args.image_size), args.batch_size)

    y_true = np.concatenate([labels.numpy().reshape(-1) for _, labels in dataset]).astype(int)
    scores = model.predict(dataset, verbose=1).reshape(-1)
    y_pred = (scores >= args.threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
        average="binary",
        pos_label=1,
    )
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    print("BrandShield-AI Classifier Evaluation")
    print("====================================")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nConfusion Matrix [Fake, Real]:")
    print(matrix)
    print("\nClassification Report:")
    print(report)

    save_confusion_matrix(matrix, output_dir / "confusion_matrix.png")
    save_predictions_csv(dataset.file_paths, y_true, scores, y_pred, output_dir / "predictions.csv")

    metrics_path = output_dir / "metrics.txt"
    metrics_path.write_text(
        "BrandShield-AI Classifier Evaluation\n"
        "====================================\n"
        f"Model: {model_path}\n"
        f"Data: {data_root}\n"
        f"Threshold: {args.threshold:.2f}\n"
        f"Accuracy : {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall   : {recall:.4f}\n"
        f"F1-score : {f1:.4f}\n\n"
        "Confusion Matrix [Fake, Real]:\n"
        f"{matrix}\n\n"
        "Classification Report:\n"
        f"{report}\n",
        encoding="utf-8",
    )

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved confusion matrix to: {output_dir / 'confusion_matrix.png'}")
    print(f"Saved predictions to: {output_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
