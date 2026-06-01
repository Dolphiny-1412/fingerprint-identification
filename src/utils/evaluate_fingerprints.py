#!/usr/bin/env python3
"""
Batch evaluate fingerprint classifier on Images/fingerprints across all databases (DB1_B..DB4_B).
Outputs:
- results/fingerprint_predictions.csv
- results/fingerprint_summary.json
Prints a concise summary to stdout.
"""
import os
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure we can import from project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
if (PROJECT_ROOT / "classifier.py").exists():
    sys.path.insert(0, str(PROJECT_ROOT))

from classifier import classify_fingerprint  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def iter_image_paths(root: Path) -> List[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def main(images_root: str = "Images/fingerprints") -> int:
    start = time.time()
    root = Path(images_root)
    if not root.exists():
        print(f"ERROR: images root not found: {root}")
        return 2

    # Prepare output
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "fingerprint_predictions.csv"
    summary_path = results_dir / "fingerprint_summary.json"

    # Discover DB folders (first-level dirs under images_root)
    db_dirs = [d for d in root.iterdir() if d.is_dir()]
    db_dirs.sort()

    rows: List[Dict] = []
    overall_counts: Dict[str, int] = {"Loop": 0, "Whorl": 0, "Arch": 0, "Unknown": 0}
    per_db_counts: Dict[str, Dict[str, int]] = {}

    total_images = 0

    for db in db_dirs:
        per_db_counts[db.name] = {"Loop": 0, "Whorl": 0, "Arch": 0, "Unknown": 0}
        for img_path in iter_image_paths(db):
            total_images += 1
            try:
                res = classify_fingerprint(str(img_path))
            except Exception as e:
                res = {"label": "Unknown", "confidence": 0.0, "error": str(e)}

            label = res.get("label", "Unknown") or "Unknown"
            conf = float(res.get("confidence", 0.0) or 0.0)
            probs = res.get("probabilities", {})

            # Update counts
            if label not in overall_counts:
                overall_counts[label] = 0
            overall_counts[label] += 1
            if label not in per_db_counts[db.name]:
                per_db_counts[db.name][label] = 0
            per_db_counts[db.name][label] += 1

            rows.append({
                "path": str(img_path),
                "db": db.name,
                "label": label,
                "confidence": conf,
                "prob_Loop": float(probs.get("Loop", 0.0)),
                "prob_Whorl": float(probs.get("Whorl", 0.0)),
                "prob_Arch": float(probs.get("Arch", 0.0)),
                "error": res.get("error", "")
            })

    # Write CSV
    fieldnames = [
        "path", "db", "label", "confidence",
        "prob_Loop", "prob_Whorl", "prob_Arch", "error"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Build summary
    def pct(x: int) -> float:
        return 0.0 if total_images == 0 else x / total_images

    summary = {
        "images_root": str(root),
        "total_images": total_images,
        "overall_counts": overall_counts,
        "overall_distribution": {k: pct(v) for k, v in overall_counts.items()},
        "per_db_counts": per_db_counts,
        "per_db_distribution": {
            db: {
                k: (0.0 if sum(per_db_counts[db].values()) == 0 else v / sum(per_db_counts[db].values()))
                for k, v in per_db_counts[db].items()
            } for db in per_db_counts
        },
        "runtime_sec": round(time.time() - start, 2)
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print concise summary
    print("\nEvaluation complete")
    print(f"Total images: {total_images}")
    print("Overall distribution:")
    for k in ["Loop", "Whorl", "Arch", "Unknown"]:
        v = overall_counts.get(k, 0)
        share = 0.0 if total_images == 0 else v / total_images
        print(f"  {k:7s}: {v:5d} ({share:5.1%})")
    print(f"CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

