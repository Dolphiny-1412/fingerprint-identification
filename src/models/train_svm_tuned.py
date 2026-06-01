import csv
import json
from pathlib import Path
from typing import List, Tuple, Dict
import random
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PNGs
import matplotlib.pyplot as plt

# Configuration
DATA_ROOT = Path.cwd() / "training data" / "MyCustom"
R_FOLDERS = [f"R{i}" for i in range(1, 11)]
LABEL_CSV_GLOB = "R*_Labels.csv"
OUTPUT_DIR = Path.cwd() / "models"
REPORTS_DIR = Path.cwd() / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "fingerprint_svm_tuned.yml"
CONFIG_PATH = OUTPUT_DIR / "fingerprint_config_tuned.json"
SEARCH_CSV = REPORTS_DIR / "svm_gridsearch_results.csv"
TRAIN_CM_CSV = REPORTS_DIR / "confusion_matrix_train.csv"
VAL_CM_CSV = REPORTS_DIR / "confusion_matrix_val.csv"
TRAIN_CM_PNG = REPORTS_DIR / "confusion_matrix_train.png"
VAL_CM_PNG = REPORTS_DIR / "confusion_matrix_val.png"

# Feature extraction params
IMG_SIZE = (128, 128)  # width, height
NUM_BINS = 9
GRID_YX = (4, 4)  # (rows, cols)
RANDOM_SEED = 42
TRAIN_SPLIT = 0.8

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

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def read_all_labels(data_root: Path) -> Dict[str, str]:
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
    labels = read_all_labels(data_root)
    samples: List[Tuple[Path, int]] = []
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
    img = cv2.resize(img_gray, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
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
    v /= (np.linalg.norm(v) + 1e-6)
    return v


def augment_image(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    # random small rotation
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # random translation
    tx = int(random.uniform(-5, 5))
    ty = int(random.uniform(-5, 5))
    T = np.float32([[1, 0, tx], [0, 1, ty]])
    img = cv2.warpAffine(img, T, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # random contrast/brightness
    alpha = random.uniform(0.9, 1.1)
    beta = random.uniform(-10, 10)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


def compute_features(samples: List[Tuple[Path, int]]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for path, cls in samples:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        feat = extract_hog_like(img)
        X.append(feat)
        y.append(cls)
    return np.vstack(X).astype(np.float32), np.asarray(y, dtype=np.int32)


def compute_features_with_augmentation(train_samples: List[Tuple[Path, int]], target_per_class: int) -> Tuple[np.ndarray, np.ndarray]:
    # Load images grouped by class
    by_class: Dict[int, List[np.ndarray]] = {}
    for path, cls in train_samples:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        by_class.setdefault(cls, []).append(img)

    X_list, y_list = [], []
    for cls, imgs in by_class.items():
        # Compute features for originals
        for img in imgs:
            X_list.append(extract_hog_like(img))
            y_list.append(cls)
        # Augment until reaching target_per_class
        needed = max(0, target_per_class - len(imgs))
        for i in range(needed):
            base = imgs[i % len(imgs)]
            aug = augment_image(base)
            X_list.append(extract_hog_like(aug))
            y_list.append(cls)
    X = np.vstack(X_list).astype(np.float32)
    y = np.asarray(y_list, dtype=np.int32)
    return X, y


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


def train_eval_svm(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, C: float, gamma: float):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(float(C))
    svm.setGamma(float(gamma))
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3))
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    def acc(X, y):
        _, p = svm.predict(X)
        p = p.ravel().astype(np.int32)
        return float((p == y).mean()), p
    train_acc, _ = acc(X_train, y_train)
    val_acc, val_preds = acc(X_val, y_val)
    return svm, train_acc, val_acc, val_preds


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_cm_csv(cm: np.ndarray, labels: List[str], path: Path):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + labels)
        for i, row in enumerate(cm):
            writer.writerow([labels[i]] + list(map(int, row)))


def save_cm_png(cm: np.ndarray, labels: List[str], path: Path):
    plt.figure(figsize=(5, 4), dpi=150)
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    print("Building dataset from:", DATA_ROOT)
    samples = build_dataset(DATA_ROOT)
    print(f"Total labeled images: {len(samples)}")

    train_samples, val_samples = stratified_split(samples, TRAIN_SPLIT, RANDOM_SEED)
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)}")

    # Determine class counts to balance via augmentation
    cls_counts: Dict[int, int] = {}
    for _p, c in train_samples:
        cls_counts[c] = cls_counts.get(c, 0) + 1
    target_per_class = max(cls_counts.values())
    print("Train per-class counts:", {IDX_TO_CLASS[k]: v for k, v in cls_counts.items()}, "-> target", target_per_class)

    # Build features: augmented for train, original for val
    print("Computing training features with augmentation...")
    X_train, y_train = compute_features_with_augmentation(train_samples, target_per_class)
    print("Computing validation features...")
    X_val, y_val = compute_features(val_samples)
    print("Feature shapes:", X_train.shape, X_val.shape)

    # Standardize
    X_train_n, stats = standardize_fit(X_train)
    X_val_n = standardize_apply(X_val, stats)

    # Grid search for SVM hyperparameters
    d = X_train_n.shape[1]
    gamma_base = 1.0 / d
    C_values = [0.5, 1, 2, 4, 8, 16]
    gamma_multipliers = [0.25, 0.5, 1, 2, 4]

    best = {"C": None, "gamma": None, "train_acc": -1, "val_acc": -1}
    rows = [("C", "gamma", "train_acc", "val_acc")]
    print("Starting grid search...")
    for C in C_values:
        for m in gamma_multipliers:
            gamma = m * gamma_base
            svm, tr_acc, va_acc, val_preds = train_eval_svm(X_train_n, y_train, X_val_n, y_val, C, gamma)
            rows.append((C, gamma, tr_acc, va_acc))
            if va_acc > best["val_acc"]:
                best.update({"C": C, "gamma": gamma, "train_acc": tr_acc, "val_acc": va_acc, "svm": svm, "val_preds": val_preds})
            print(f"C={C:>4}, gamma~={gamma:.5f} -> train={tr_acc:.3f}, val={va_acc:.3f}")

    # Save search results
    with SEARCH_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)
    print("Saved grid search results to:", SEARCH_CSV)

    # Evaluate best model on train and val
    svm = best["svm"]
    # Recompute preds for train/val
    _, train_preds = svm.predict(X_train_n)
    train_preds = train_preds.ravel().astype(np.int32)
    val_preds = best["val_preds"].ravel().astype(np.int32)
    train_acc = float((train_preds == y_train).mean())
    val_acc = float((val_preds == y_val).mean())

    print(f"Best params -> C={best['C']}, gamma={best['gamma']:.6f}")
    print(f"Final Train acc: {train_acc:.3f}")
    print(f"Final Val acc: {val_acc:.3f}")

    # Confusion matrices
    labels = [IDX_TO_CLASS[i] for i in range(3)]
    train_cm = confusion_matrix(y_train, train_preds, 3)
    val_cm = confusion_matrix(y_val, val_preds, 3)

    save_cm_csv(train_cm, labels, TRAIN_CM_CSV)
    save_cm_csv(val_cm, labels, VAL_CM_CSV)
    save_cm_png(train_cm, labels, TRAIN_CM_PNG)
    save_cm_png(val_cm, labels, VAL_CM_PNG)
    print("Saved confusion matrices to:", TRAIN_CM_CSV, VAL_CM_CSV, TRAIN_CM_PNG, VAL_CM_PNG)

    # Save model + config
    svm.save(str(MODEL_PATH))
    config = {
        "img_size": IMG_SIZE,
        "num_bins": NUM_BINS,
        "grid_yx": GRID_YX,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": IDX_TO_CLASS,
        "standardize": stats,
        "best": {"C": best["C"], "gamma": best["gamma"]},
    }
    with CONFIG_PATH.open('w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print("Saved model to:", MODEL_PATH)
    print("Saved config to:", CONFIG_PATH)


if __name__ == "__main__":
    main()

