import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


CLASS_NAMES = ["Fake", "Real"]


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    y_true = []
    scores = []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            y_true.append(CLASS_NAMES.index(row["actual"]))
            scores.append(float(row["real_score"]))
    return np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find the best Real/Fake decision threshold.")
    parser.add_argument("--predictions", default="outputs/evaluation/predictions.csv")
    parser.add_argument("--metric", choices=["f1", "balanced_accuracy"], default="f1")
    parser.add_argument("--output", default="outputs/evaluation/threshold_sweep.csv")
    args = parser.parse_args()

    y_true, scores = load_predictions(Path(args.predictions))
    rows = []
    best = None

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (scores >= threshold).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
        fake_recall = matrix[0, 0] / matrix[0].sum() if matrix[0].sum() else 0.0
        real_recall = matrix[1, 1] / matrix[1].sum() if matrix[1].sum() else 0.0
        balanced_accuracy = (fake_recall + real_recall) / 2
        score = f1 if args.metric == "f1" else balanced_accuracy
        row = {
            "threshold": threshold,
            "accuracy": accuracy,
            "precision_real": precision,
            "recall_real": recall,
            "f1_real": f1,
            "fake_recall": fake_recall,
            "real_recall": real_recall,
            "balanced_accuracy": balanced_accuracy,
            "tn_fake": matrix[0, 0],
            "fp_fake_as_real": matrix[0, 1],
            "fn_real_as_fake": matrix[1, 0],
            "tp_real": matrix[1, 1],
        }
        rows.append(row)
        if best is None or score > best[0]:
            best = (score, row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_row = best[1]
    print("Best Threshold")
    print("==============")
    print(f"Metric optimized : {args.metric}")
    print(f"Threshold        : {best_row['threshold']:.2f}")
    print(f"Accuracy         : {best_row['accuracy']:.4f}")
    print(f"Real precision   : {best_row['precision_real']:.4f}")
    print(f"Real recall      : {best_row['recall_real']:.4f}")
    print(f"Real F1          : {best_row['f1_real']:.4f}")
    print(f"Fake recall      : {best_row['fake_recall']:.4f}")
    print(f"Balanced accuracy: {best_row['balanced_accuracy']:.4f}")
    print("\nConfusion Matrix [Fake, Real]:")
    print(f"[[{best_row['tn_fake']} {best_row['fp_fake_as_real']}]")
    print(f" [{best_row['fn_real_as_fake']} {best_row['tp_real']}]]")
    print(f"\nSaved sweep to: {output_path}")


if __name__ == "__main__":
    main()
