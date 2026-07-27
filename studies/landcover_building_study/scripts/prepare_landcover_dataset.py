from __future__ import annotations

import argparse
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.config import load_config, resolve_path
from landcover_building.data import prepare_landcover_single_class_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a LandCover.ai building-only subset for YOLO + mask IoU.")
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "configs" / "yolo_sam3.yaml")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--class-value", type=int, default=None)
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--image-format", choices=["jpg", "png"], default=None)
    parser.add_argument("--min-component-area", type=int, default=None)
    parser.add_argument("--train-negative-ratio", type=float, default=None)
    parser.add_argument("--sampling-seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config["dataset"]

    category_name = args.category or dataset_cfg["target_category"]
    category_value = args.class_value if args.class_value is not None else int(dataset_cfg["target_category_value"])
    tile_size = args.tile_size or int(dataset_cfg["tile_size"])
    image_format = args.image_format or str(dataset_cfg["tile_image_format"])
    min_component_area = (
        args.min_component_area
        if args.min_component_area is not None
        else int(dataset_cfg["min_component_area"])
    )
    train_negative_ratio = (
        args.train_negative_ratio
        if args.train_negative_ratio is not None
        else dataset_cfg.get("train_negative_ratio")
    )
    sampling_seed = args.sampling_seed if args.sampling_seed is not None else int(dataset_cfg.get("sampling_seed", 42))

    summary = prepare_landcover_single_class_dataset(
        raw_root=resolve_path(config["paths"]["raw_dataset_dir"]),
        output_root=resolve_path(config["paths"]["prepared_dataset_dir"]),
        category_name=category_name,
        category_value=category_value,
        tile_size=tile_size,
        image_format=image_format,
        min_component_area=min_component_area,
        train_split=str(dataset_cfg["train_split"]),
        val_split=str(dataset_cfg["eval_split"]),
        test_split=str(dataset_cfg.get("test_split")) if dataset_cfg.get("test_split") else None,
        train_negative_ratio=float(train_negative_ratio) if train_negative_ratio is not None else None,
        sampling_seed=sampling_seed,
    )

    print("Prepared LandCover.ai subset")
    for split, split_summary in summary.items():
        print(f"{split}: {split_summary}")
    print(f"dataset_root: {resolve_path(config['paths']['prepared_dataset_dir'])}")


if __name__ == "__main__":
    main()
