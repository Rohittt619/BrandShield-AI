
from pathlib import Path
from datetime import datetime
 
OUTPUTS_DIR = Path("outputs/evaluation")
REPORT_PATH = OUTPUTS_DIR / "report.md"
 
 
def generate_report() -> None:
    metrics_path = OUTPUTS_DIR / "metrics.txt"
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
 
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found — run evaluate_classifier.py first."
        )
 
    metrics_text = metrics_path.read_text()
 
    lines = [
        "# BrandShield-AI Evaluation Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Metrics",
        "```",
        metrics_text.strip(),
        "```",
    ]
 
    if cm_path.exists():
        lines += ["", "## Confusion Matrix", f"![Confusion Matrix]({cm_path.name})"]
    else:
        lines += ["", "_No confusion matrix found — run evaluate_classifier.py to generate one._"]
 
    REPORT_PATH.write_text("\n".join(lines))
    print(f"Report written to {REPORT_PATH}")
 
 
if __name__ == "__main__":
    generate_report()