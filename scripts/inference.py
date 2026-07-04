import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def load_image(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(image_size)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BrandShield classifier inference on one image.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    image = load_image(Path(args.image), (args.image_size, args.image_size))
    score = float(model.predict(image, verbose=0)[0][0])
    label = "Real" if score >= 0.5 else "Fake"
    confidence = score if label == "Real" else 1 - score
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
