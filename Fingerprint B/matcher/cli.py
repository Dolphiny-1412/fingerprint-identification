import argparse
from pathlib import Path

from .matcher import MatchConfig, build_database, match_image


def main():
    parser = argparse.ArgumentParser(description="Fingerprint matcher utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build ORB descriptor database from an images root")
    p_build.add_argument("--images-root", default=str(Path("Images")), help="Root folder with fingerprint images")
    p_build.add_argument("--db-dir", default="data/fingerprints_db", help="Output database directory")

    p_match = sub.add_parser("match", help="Match a query image against the database")
    p_match.add_argument("image", help="Path to query fingerprint image")
    p_match.add_argument("--images-root", default=str(Path("Images")))
    p_match.add_argument("--db-dir", default="data/fingerprints_db")

    args = parser.parse_args()

    if args.cmd == "build":
        cfg = MatchConfig(images_root=args.images_root, db_dir=args.db_dir)
        count = build_database(cfg)
        print(f"Indexed {count} images into {args.db_dir}")
    elif args.cmd == "match":
        cfg = MatchConfig(images_root=args.images_root, db_dir=args.db_dir)
        print(match_image(args.image, cfg))


if __name__ == "__main__":
    main()


