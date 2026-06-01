#!/usr/bin/env python3
"""
Train the classifier on a real labeled dataset organized as:
  <data_dir>/{Loop,Whorl,Arch}/*.png|jpg|jpeg|tif
Usage:
  python train_real.py --data-dir data/train
"""
import argparse
from classifier import get_classifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Path with subfolders Loop/Whorl/Arch")
    args = parser.parse_args()

    clf = get_classifier()
    results = clf.train(data_dir=args.data_dir)

    print("Training results:")
    print({k: v for k, v in results.items() if k in ("best_model", "train_accuracy", "test_accuracy", "macro_f1", "balanced_accuracy")})
    print("Model saved in models/ and plots in reports/ if enabled.")


if __name__ == "__main__":
    main()

