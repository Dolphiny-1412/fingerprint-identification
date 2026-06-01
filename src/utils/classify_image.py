import json
from pathlib import Path
import sys
import numpy as np
import cv2

CONFIG_PATH = Path.cwd() / "models" / "fingerprint_config_tuned.json"
MODEL_PATH = Path.cwd() / "models" / "fingerprint_svm_tuned.yml"


def extract_hog_like(img_gray, img_size, num_bins, grid_yx):
    img = cv2.resize(img_gray, tuple(img_size), interpolation=cv2.INTER_AREA)
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
    feats = []
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python classify_image.py <image_path>")
        sys.exit(1)
    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    with CONFIG_PATH.open('r', encoding='utf-8') as f:
        cfg = json.load(f)
    idx_to_class = {int(k): v for k, v in cfg["idx_to_class"].items()}

    svm = cv2.ml.SVM_load(str(MODEL_PATH))

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to read image.")
        sys.exit(1)

    feat = extract_hog_like(img, cfg["img_size"], cfg["num_bins"], tuple(cfg["grid_yx"]))
    mean = np.array(cfg["standardize"]["mean"], dtype=np.float32)
    std = np.array(cfg["standardize"]["std"], dtype=np.float32)
    feat = (feat - mean) / std
    feat = feat.astype(np.float32).reshape(1, -1)

    _ret, pred = svm.predict(feat)
    pred = int(pred.ravel()[0])
    print(idx_to_class[pred])


if __name__ == "__main__":
    main()

