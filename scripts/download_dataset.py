from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam3_bbox_study.config import load_config, resolve_path
from sam3_bbox_study.data.landcover import (
    download_landcover_archive,
    expected_archive_size_bytes,
    extract_landcover_archive,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract the official LandCover.ai archive.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--skip-extract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    raw_root = resolve_path(config["paths"]["raw_dataset_dir"])
    dataset_cfg = config["dataset"]
    archive_name = dataset_cfg["archive_name"]
    url = dataset_cfg["urls"]["landcover_v1_zip"]
    archive_path = raw_root / archive_name

    print(f"Resolving archive size for {url}")
    expected_size = expected_archive_size_bytes(url)
    if expected_size is not None:
        print(f"Expected archive size: {expected_size} bytes")

    print(f"Downloading LandCover.ai -> {archive_path}")
    download_landcover_archive(url=url, archive_path=archive_path, expected_size_bytes=expected_size)

    if args.skip_extract:
        print("Skipping extraction as requested.")
        return

    print(f"Extracting archive -> {raw_root}")
    extract_landcover_archive(archive_path=archive_path, raw_root=raw_root)
    print("LandCover.ai raw dataset is ready.")


if __name__ == "__main__":
    main()
