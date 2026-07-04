import argparse
from pathlib import Path

import tensorflow as tf


CLASS_NAMES = ["Fake", "Real"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(data_root: Path) -> dict[int, int]:
    counts = {}
    train_root = data_root / "train"
    for index, class_name in enumerate(CLASS_NAMES):
        class_dir = train_root / class_name
        counts[index] = len([
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ])
    return counts


def compute_class_weights(counts: dict[int, int], real_weight_scale: float) -> dict[int, float]:
    total = sum(counts.values())
    classes = len(counts)
    weights = {class_id: total / (classes * count) for class_id, count in counts.items() if count > 0}
    if 1 in weights:
        weights[1] *= real_weight_scale
    return weights


def build_model(image_size: tuple[int, int], base_trainable: bool = False, trainable_layers: int = 30) -> tf.keras.Model:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size[0], image_size[1], 3),
    )
    base.trainable = base_trainable
    if base_trainable:
        for layer in base.layers[:-trainable_layers]:
            layer.trainable = False

    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.0)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4 if not base_trainable else 1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BrandShield Real/Fake logo classifier.")
    parser.add_argument("--data", default="dataset/raw/images", help="Folder with train/val/Fake/Real layout.")
    parser.add_argument("--model-out", default="models/brandshield_classifier.h5")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--real-weight-scale", type=float, default=1.0, help="Extra multiplier for the minority Real class weight.")
    args = parser.parse_args()

    data_root = Path(args.data)
    image_size = (args.image_size, args.image_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "train",
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_root / "val",
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=image_size,
        batch_size=args.batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    counts = count_images(data_root)
    class_weights = compute_class_weights(counts, args.real_weight_scale)
    print(f"Training class counts: {counts}")
    print(f"Class weights: {class_weights}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.4, patience=3, min_lr=1e-7),
    ]

    model = build_model(image_size, base_trainable=False)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    if args.fine_tune_epochs > 0:
        print("Fine-tuning top MobileNetV2 layers...")
        fine_tune_model = build_model(image_size, base_trainable=True, trainable_layers=30)
        fine_tune_model.set_weights(model.get_weights())
        model = fine_tune_model
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
        )

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_out)
    print(f"Saved model to {model_out}")


if __name__ == "__main__":
    main()
