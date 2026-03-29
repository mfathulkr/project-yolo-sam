from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pool_segmentation_compare.config import load_config, resolve_path
from pool_segmentation_compare.data.isaid import download_isaid_file, download_isaid_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official iSAID splits from Google Drive.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--skip-rgb", action="store_true")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--skip-test-info", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    raw_root = resolve_path(config["paths"]["raw_dataset_dir"])
    urls = config["dataset"]["urls"]

    if not args.skip_train:
        train_dir = raw_root / "train"
        print(f"Downloading iSAID train annotations/masks -> {train_dir}")
        download_isaid_folder(urls["isaid_train"], train_dir)
        if not args.skip_rgb:
            print(f"Downloading DOTA train RGB images -> {train_dir / 'images'}")
            download_isaid_folder(urls["dota_train_images"], train_dir / "images")

    if not args.skip_val:
        val_dir = raw_root / "val"
        print(f"Downloading iSAID val annotations/masks -> {val_dir}")
        download_isaid_folder(urls["isaid_val"], val_dir)
        if not args.skip_rgb:
            print(f"Downloading DOTA val RGB images -> {val_dir / 'images'}")
            download_isaid_folder(urls["dota_val_images"], val_dir / "images")

    if args.include_test:
        test_dir = raw_root / "test"
        print(f"Downloading iSAID test images -> {test_dir}")
        download_isaid_folder(urls["test"], test_dir)

    if args.include_test and not args.skip_test_info:
        test_info_path = raw_root / "test_info.json"
        print(f"Downloading iSAID test info -> {test_info_path}")
        download_isaid_file(urls["test_info"], test_info_path)


if __name__ == "__main__":
    main()
