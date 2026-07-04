from pathlib import Path
 
# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "raw" / "images"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "evaluation"
 
DEFAULT_MODEL_PATH = MODELS_DIR / "brandshield_classifier.h5"
 
# ---- Image / data settings ----
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
 
# Class labels — order must match your training generator's class_indices
CLASS_NAMES = ["Fake", "Real"]
 
# ---- Training ----
WARMUP_EPOCHS = 5
FINE_TUNE_EPOCHS = 20
FINE_TUNE_UNFROZEN_LAYERS = 30
LEARNING_RATE = 1e-4
 
# ---- Decision thresholds ----
# From your tune_threshold.py sweeps — update if you re-run the sweep
DEFAULT_THRESHOLD = 0.50
BALANCED_ACCURACY_THRESHOLD = 0.58
BEST_F1_THRESHOLD = 0.64