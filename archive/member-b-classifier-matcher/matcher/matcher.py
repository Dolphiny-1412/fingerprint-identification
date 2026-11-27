import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


DB_DIRNAME = "data/fingerprints_db"


@dataclass
class MatchConfig:
    images_root: str = "Images"  # default under workspace root
    db_dir: str = DB_DIRNAME
    image_globs: Tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    orb_nfeatures: int = 1500
    ratio_test: float = 0.75
    min_inliers_for_accept: int = 25
    min_keypoints_quality: int = 120
    laplacian_var_spoof_threshold: float = 15.0


def _load_grayscale(path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if target_size is not None:
        img = img.resize(target_size)
    return np.asarray(img, dtype=np.uint8)


def _compute_orb(image_gray: np.ndarray, nfeatures: int) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    orb = cv2.ORB_create(nfeatures=nfeatures)
    keypoints, descriptors = orb.detectAndCompute(image_gray, None)
    return keypoints, descriptors


def _laplacian_var(image_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(image_gray, cv2.CV_64F).var())


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iter_images(root: Path, patterns: Tuple[str, ...]):
    for pat in patterns:
        yield from root.rglob(pat)


def build_database(cfg: MatchConfig) -> int:
    images_root = Path(cfg.images_root)
    db_dir = Path(cfg.db_dir)
    _ensure_dir(db_dir)

    index: List[Dict[str, str]] = []
    count = 0
    for img_path in _iter_images(images_root, cfg.image_globs):
        try:
            gray = _load_grayscale(os.fspath(img_path))
        except Exception:
            continue
        kps, desc = _compute_orb(gray, cfg.orb_nfeatures)
        if desc is None or len(kps) == 0:
            continue

        # Use stable id from relative path
        rel = os.path.relpath(os.fspath(img_path), os.fspath(images_root))
        entry_id = rel.replace(os.sep, "__")

        # Save descriptor npz per entry
        entry_file = db_dir / f"{entry_id}.npz"
        np.savez_compressed(os.fspath(entry_file), descriptors=desc)

        index.append({"id": entry_id, "rel_path": rel, "npz": entry_file.name})
        count += 1

    with open(db_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    return count


def _load_db(db_dir: Path) -> List[Dict[str, str]]:
    index_path = db_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Database index not found at {index_path}. Run build_database first.")
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _match_descriptors(query_desc: np.ndarray, cand_desc: np.ndarray, ratio: float) -> int:
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(query_desc, cand_desc, k=2)
    good = 0
    for m in matches:
        if len(m) < 2:
            continue
        if m[0].distance < ratio * m[1].distance:
            good += 1
    return good


def match_image(image_path: str, cfg: MatchConfig) -> Dict[str, object]:
    db_dir = Path(cfg.db_dir)
    images_root = Path(cfg.images_root)

    # Load query
    gray = _load_grayscale(image_path)
    kps, desc = _compute_orb(gray, cfg.orb_nfeatures)
    if desc is None or desc.size == 0:
        return {
            "match_score": 0.0,
            "matched_id": None,
            "access_granted": False,
            "spoof_flag": True,
            "quality_score": 0.0,
        }

    # Spoof heuristics
    keypoint_quality = len(kps)
    blur_var = _laplacian_var(gray)
    spoof_flag = False
    quality_score = min(1.0, keypoint_quality / max(1.0, float(cfg.min_keypoints_quality)))
    if keypoint_quality < cfg.min_keypoints_quality or blur_var < cfg.laplacian_var_spoof_threshold:
        spoof_flag = True

    # Search DB
    index = _load_db(db_dir)
    best_good = -1
    best_entry: Optional[Dict[str, str]] = None

    for entry in index:
        npz_path = db_dir / entry["npz"]
        if not npz_path.exists():
            continue
        with np.load(os.fspath(npz_path)) as npz:
            cand_desc = npz["descriptors"]
        good = _match_descriptors(desc, cand_desc, cfg.ratio_test)
        if good > best_good:
            best_good = good
            best_entry = entry

    # Normalize score by query keypoints to keep 0..1 approx scale
    match_score = 0.0
    matched_id = None
    access_granted = False
    if best_entry is not None and len(kps) > 0:
        match_score = float(best_good) / float(len(kps))
        matched_id = best_entry["id"]
        access_granted = (best_good >= cfg.min_inliers_for_accept) and (not spoof_flag)

    return {
        "match_score": round(match_score, 6),
        "matched_id": matched_id,
        "access_granted": bool(access_granted),
        "spoof_flag": bool(spoof_flag),
        "quality_score": round(float(quality_score), 6),
        "query_keypoints": int(len(kps)),
        "blur_variance": round(float(blur_var), 4),
        "db_size": int(len(index)),
        "images_root": os.fspath(images_root),
    }


