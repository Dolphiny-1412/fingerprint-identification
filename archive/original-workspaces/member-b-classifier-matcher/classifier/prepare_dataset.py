import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple, Dict


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def _iter_images(root: Path) -> Iterable[Path]:
    for ext in IMAGE_EXTS:
        yield from root.rglob(f"*{ext}")


def _matches_any_keyword(path: Path, keywords: List[str]) -> bool:
    parts = [p.lower() for p in path.parts]
    for kw in keywords:
        kw = kw.lower()
        if kw in parts or any(kw in p for p in parts):
            return True
    return False


def collect_by_class(images_root: Path, classes: Tuple[str, ...], keyword_map: Dict[str, List[str]]) -> dict:
    buckets = {c: [] for c in classes}
    for p in _iter_images(images_root):
        parent = p.parent
        for c in classes:
            keywords = keyword_map.get(c, [c])
            if _matches_any_keyword(parent, keywords):
                buckets[c].append(p)
                break
    return buckets


def copy_balanced_subset(buckets: dict, out_root: Path, per_class: int) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    copied = {}
    for c, items in buckets.items():
        cls_dir = out_root / c
        cls_dir.mkdir(parents=True, exist_ok=True)
        random.shuffle(items)
        take = items[:per_class] if per_class > 0 else items
        copied[c] = []
        for src in take:
            dst = cls_dir / src.name
            try:
                shutil.copy2(os.fspath(src), os.fspath(dst))
                copied[c].append(dst)
            except Exception:
                continue
    return copied


def main():
    parser = argparse.ArgumentParser(description="Prepare balanced fingerprint dataset from Images")
    parser.add_argument("--images-root", default=r"C:\\Users\\Youss\\Desktop\\College\\M.A.H.E\\Images")
    parser.add_argument("--out-dir", default=r"C:\\Users\\Youss\\Desktop\\College\\M.A.H.E\\Images\\FingerprintClasses")
    parser.add_argument("--classes", default="Arch,Loop,Whorl")
    parser.add_argument(
        "--map",
        default="Arch:arch,tented;Loop:loop;Whorl:whorl",
        help="Class to keywords mapping, e.g. 'Arch:arch,tented;Loop:loop;Whorl:whorl'",
    )
    parser.add_argument("--per-class", type=int, default=200, help="Max samples per class to copy (0 = all)")
    args = parser.parse_args()

    classes = tuple([c.strip() for c in args.classes.split(",") if c.strip()])
    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir)

    # Parse mapping
    keyword_map: Dict[str, List[str]] = {c: [c] for c in classes}
    if args.map:
        for pair in args.map.split(";"):
            if ":" in pair:
                k, v = pair.split(":", 1)
                k = k.strip()
                if k in keyword_map:
                    keyword_map[k] = [s.strip() for s in v.split(",") if s.strip()]

    buckets = collect_by_class(images_root, classes, keyword_map)
    copied = copy_balanced_subset(buckets, out_dir, args.per_class)

    summary = {k: len(v) for k, v in copied.items()}
    print({"output_dir": os.fspath(out_dir), "copied_counts": summary})


if __name__ == "__main__":
    main()


