import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _discover_images(data_dir: Path, class_names: Tuple[str, ...]) -> Tuple[List[str], List[int]]:
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


def _load_to_vector(path: str, size: Tuple[int, int]) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()


def main():
    parser = argparse.ArgumentParser(description="Train scikit-learn classifier (PCA+SVM) for Arch/Loop/Whorl")
    parser.add_argument("--data-dir", default=r"C:\\Users\\Youss\\Desktop\\College\\M.A.H.E\\Images\\FingerprintClasses")
    parser.add_argument("--output-dir", default="classifier")
    parser.add_argument("--classes", default="Arch,Loop,Whorl")
    parser.add_argument("--image-size", default="128,128")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=128)
    args = parser.parse_args()

    classes = tuple([c.strip() for c in args.classes.split(",") if c.strip()])
    h, w = [int(x) for x in args.image_size.split(",")]
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_paths, y = _discover_images(data_dir, classes)
    if not X_paths:
        raise RuntimeError(f"No images found under {data_dir}")

    X = np.stack([_load_to_vector(p, (h, w)) for p in X_paths], axis=0)
    y = np.array(y, dtype=np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=min(args.pca_components, X_train.shape[1]))),
        ("svm", SVC(kernel="rbf", probability=True, C=5.0, gamma="scale")),
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    cm = confusion_matrix(y_val, y_pred).tolist()

    model_path = output_dir / "model_sklearn.joblib"
    label_map_path = output_dir / "label_map.json"
    metrics_path = output_dir / "metrics_sklearn.json"

    joblib.dump(pipe, os.fspath(model_path))
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(list(classes), f, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"val_accuracy": acc, "confusion_matrix": cm}, f, indent=2)

    print({
        "model": os.fspath(model_path),
        "labels": os.fspath(label_map_path),
        "metrics": os.fspath(metrics_path),
        "val_accuracy": acc,
    })


if __name__ == "__main__":
    main()


