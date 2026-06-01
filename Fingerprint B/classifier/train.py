import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def _discover_images(data_dir: Path, class_names: List[str]) -> Tuple[List[str], List[int]]:
    image_paths: List[str] = []
    labels: List[int] = []
    for idx, name in enumerate(class_names):
        class_dir = data_dir / name
        if not class_dir.exists():
            continue
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            for p in class_dir.rglob(ext):
                image_paths.append(os.fspath(p))
                labels.append(idx)
    return image_paths, labels


def _load_batch(paths: List[str], size: Tuple[int, int]) -> np.ndarray:
    batch = []
    for p in paths:
        img = Image.open(p).convert("L").resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=-1)
        batch.append(arr)
    return np.stack(batch, axis=0)


@dataclass
class TrainConfig:
    data_dir: str = "data/classifier_dataset"
    output_dir: str = "classifier"
    classes: Tuple[str, ...] = ("Arch", "Loop", "Whorl")
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 16
    epochs: int = 10
    val_split: float = 0.2
    seed: int = 42


def build_model(input_shape: Tuple[int, int, int], num_classes: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(cfg: TrainConfig):
    from sklearn.model_selection import train_test_split
    from tensorflow import keras

    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths, labels = _discover_images(data_dir, list(cfg.classes))
    if len(image_paths) == 0:
        raise RuntimeError(
            f"No images found under {data_dir}. Expected subfolders: {', '.join(cfg.classes)}"
        )

    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=cfg.val_split, random_state=cfg.seed, stratify=labels
    )

    def generator(paths, labels, batch_size):
        size = cfg.image_size
        idxs = np.arange(len(paths))
        while True:
            np.random.shuffle(idxs)
            for start in range(0, len(paths), batch_size):
                end = min(start + batch_size, len(paths))
                batch_idx = idxs[start:end]
                batch_paths = [paths[i] for i in batch_idx]
                batch_labels = np.array([labels[i] for i in batch_idx], dtype=np.int64)
                batch_x = _load_batch(batch_paths, size)
                yield batch_x, batch_labels

    steps_per_epoch = max(1, len(X_train) // cfg.batch_size)
    val_steps = max(1, len(X_val) // cfg.batch_size)

    model = build_model((cfg.image_size[0], cfg.image_size[1], 1), len(cfg.classes))

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        generator(X_train, y_train, cfg.batch_size),
        validation_data=generator(X_val, y_val, cfg.batch_size),
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        epochs=cfg.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    model_path = Path(cfg.output_dir) / "model.h5"
    label_map_path = Path(cfg.output_dir) / "label_map.json"
    model.save(os.fspath(model_path))
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(list(cfg.classes), f, indent=2)

    # Optional: save simple metrics
    metrics_path = Path(cfg.output_dir) / "metrics.json"
    final_train_acc = float(history.history.get("accuracy", [0])[-1])
    final_val_acc = float(history.history.get("val_accuracy", [0])[-1])
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"train_accuracy": final_train_acc, "val_accuracy": final_val_acc}, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved label map to {label_map_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train fingerprint pattern classifier (Arch/Loop/Whorl)")
    parser.add_argument("--data-dir", default=r"C:\Users\Youss\Desktop\College\M.A.H.E\Images\FingerprintClasses", help="Dataset root with class subfolders (Arch/Loop/Whorl)")
    parser.add_argument("--output-dir", default="classifier", help="Directory to save model.h5 and label_map.json")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", default="224,224", help="HxW, e.g. 224,224")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--classes", default="Arch,Loop,Whorl", help="Comma-separated class names")

    args = parser.parse_args()

    h, w = [int(x) for x in args.image_size.split(",")]
    classes = tuple([c.strip() for c in args.classes.split(",") if c.strip()])

    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        classes=classes,
        image_size=(h, w),
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split,
    )
    train(cfg)


