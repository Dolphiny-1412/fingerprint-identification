import csv
import json
from pathlib import Path
from typing import List, Tuple, Dict
import random
import numpy as np
import cv2

# Configuration
DATA_ROOT = Path.cwd() / "training data" / "MyCustom"
R_FOLDERS = [f"R{i}" for i in range(1, 11)]
LABEL_CSV_GLOB = "R*_Labels.csv"
OUTPUT_DIR = Path.cwd() / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "fingerprint_svm.yml"
CONFIG_PATH = OUTPUT_DIR / "fingerprint_config.json"

# Feature extraction params
IMG_SIZE = (128, 128)  # width, height
NUM_BINS = 9
GRID_YX = (4, 4)  # (rows, cols)
RANDOM_SEED = 42
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

# Map raw labels to target classes (3 classes)
# Left Loop + Right Loop -> Loop; Tented Arch -> Arch
LABEL_MAP = {
    "Arch": "Arch",
    "Tented Arch": "Arch",
    "Left Loop": "Loop",
    "Right Loop": "Loop",
    "Whorl": "Whorl",
}

CLASS_TO_IDX = {"Arch": 0, "Loop": 1, "Whorl": 2}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


def read_all_labels(data_root: Path) -> Dict[str, str]:
    """Return mapping from image file name to simplified label (Arch/Loop/Whorl)."""
    mapping = {}
    for csv_path in data_root.glob(LABEL_CSV_GLOB):
        with csv_path.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_label = row.get("label", "").strip()
                img_name = row.get("image_name", "").strip()
                if not img_name:
                    continue
                simplified = LABEL_MAP.get(raw_label)
                if simplified in CLASS_TO_IDX:
                    mapping[img_name] = simplified
    return mapping


def build_dataset(data_root: Path) -> List[Tuple[Path, int]]:
    """Build list of (image_path, class_idx) only for files that exist."""
    labels = read_all_labels(data_root)
    samples: List[Tuple[Path, int]] = []
    # Search images within R1..R10
    for r in R_FOLDERS:
        folder = data_root / r
        if not folder.is_dir():
            continue
        for img_path in folder.glob("*.BMP"):
            name = img_path.name
            if name in labels:
                cls = labels[name]
                samples.append((img_path, CLASS_TO_IDX[cls]))
    return samples


def extract_hog_like(img_gray: np.ndarray, num_bins: int = NUM_BINS, grid_yx: Tuple[int, int] = GRID_YX) -> np.ndarray:
    """Simple HOG-like descriptor: gradients + cell histograms.
    Returns 1D feature vector.
    """
    # Resize to fixed size
    img = cv2.resize(img_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    # Mild blur to reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Gradients
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Use unsigned angles [0, 180)
    ang = ang % 180.0
    h, w = img.shape

    cells_y, cells_x = grid_yx
    cell_h = h // cells_y
    cell_w = w // cells_x
    bin_width = 180.0 / num_bins

    feats: List[float] = []
    for cy in range(cells_y):
        for cx in range(cells_x):
            y0, y1 = cy * cell_h, (cy + 1) * cell_h
            x0, x1 = cx * cell_w, (cx + 1) * cell_w
            a = ang[y0:y1, x0:x1].ravel()
            m = mag[y0:y1, x0:x1].ravel()
            bins = np.floor(a / bin_width).astype(np.int32)
            bins = np.clip(bins, 0, num_bins - 1)
            hist = np.bincount(bins, weights=m, minlength=num_bins).astype(np.float32)
            feats.extend(hist.tolist())

    v = np.asarray(feats, dtype=np.float32)
    # L2 normalize
    norm = np.linalg.norm(v) + 1e-6
    v /= norm
    return v


def compute_features(samples: List[Tuple[Path, int]]) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    for path, cls in samples:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feat = extract_hog_like(img)
        X.append(feat)
        y.append(cls)
    return np.vstack(X).astype(np.float32), np.asarray(y, dtype=np.int32)


def stratified_split(samples: List[Tuple[Path, int]], train_ratio: float = TRAIN_SPLIT, seed: int = RANDOM_SEED):
    random.seed(seed)
    by_class: Dict[int, List[Tuple[Path, int]]] = {}
    for s in samples:
        by_class.setdefault(s[1], []).append(s)
    train, val = [], []
    for cls, lst in by_class.items():
        random.shuffle(lst)
        n_train = int(len(lst) * train_ratio)
        train.extend(lst[:n_train])
        val.extend(lst[n_train:])
    random.shuffle(train)
    random.shuffle(val)
    return train, val


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    Xn = (X - mean) / std
    return Xn.astype(np.float32), {"mean": mean.tolist(), "std": std.tolist()}


def standardize_apply(X: np.ndarray, stats: Dict[str, List[float]]) -> np.ndarray:
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    return ((X - mean) / std).astype(np.float32)


def train_svm_classifier(X_train: np.ndarray, y_train: np.ndarray):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(2.0)
    gamma = 1.0 / X_train.shape[1]
    svm.setGamma(gamma)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3))

    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    return svm


def evaluate(svm, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    _, preds = svm.predict(X)
    preds = preds.ravel().astype(np.int32)
    acc = float((preds == y).mean())
    # Confusion matrix 3x3
    cm = np.zeros((3, 3), dtype=np.int32)
    for t, p in zip(y, preds):
        cm[t, p] += 1
    return acc, cm


def main():
    print("Building dataset from:", DATA_ROOT)
    samples = build_dataset(DATA_ROOT)
    print(f"Total labeled images: {len(samples)}")
    if len(samples) < 50:
        print("Warning: Very few samples detected; training quality may be poor.")

    train, val = stratified_split(samples, TRAIN_SPLIT, RANDOM_SEED)
    print(f"Train: {len(train)} | Val: {len(val)}")

    X_train, y_train = compute_features(train)
    X_val, y_val = compute_features(val)
    print("Feature shapes:", X_train.shape, X_val.shape)

    X_train_n, stats = standardize_fit(X_train)
    X_val_n = standardize_apply(X_val, stats)

    svm = train_svm_classifier(X_train_n, y_train)

    train_acc, train_cm = evaluate(svm, X_train_n, y_train)
    val_acc, val_cm = evaluate(svm, X_val_n, y_val)

    print(f"Train accuracy: {train_acc:.3f}")
    print("Train confusion matrix (rows=true, cols=pred):")
    print(train_cm)
    print(f"Val accuracy: {val_acc:.3f}")
    print("Val confusion matrix (rows=true, cols=pred):")
    print(val_cm)

    # Save model and config
    svm.save(str(MODEL_PATH))
    config = {
        "img_size": IMG_SIZE,
        "num_bins": NUM_BINS,
        "grid_yx": GRID_YX,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "standardize": stats,
    }
    with CONFIG_PATH.open('w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print("Saved model to:", MODEL_PATH)
    print("Saved config to:", CONFIG_PATH)


if __name__ == "__main__":
    main()

