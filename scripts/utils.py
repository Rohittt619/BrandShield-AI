import logging
import random
from pathlib import Path
 
import numpy as np
import tensorflow as tf
 
 
def set_seed(seed: int = 42) -> None:
    """Make runs reproducible across numpy, random, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
 
 
def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that prints to stdout with timestamps."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
 
 
def count_images(directory) -> dict:
    """
    Count images per class subfolder.
    e.g. count_images('dataset/raw/images/train') -> {'Fake': 439, 'Real': 220}
    Mirrors what validate_dataset.py already prints, but reusable elsewhere.
    """
    directory = Path(directory)
    counts = {}
    for class_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        counts[class_dir.name] = sum(
            1 for f in class_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
    return counts