from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam3_bbox_study.config import load_config, resolve_path
from sam3_bbox_study.data.isaid import prepare_isaid_vehicle_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a tiled iSAID aerial vehicle dataset for YOLO + SAM3.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "isaid_vehicle_yolo26x.yaml")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--include-edge-tiles", action="store_true", default=None)
    parser.add_argument("--no-include-edge-tiles", action="store_false", dest="include_edge_tiles")
    parser.add_argument("--train-negative-ratio", type=float, default=None)
    parser.add_argument("--eval-max-per-stratum", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    eval_cfg = config["evaluation"]

    tile_size = args.tile_size if args.tile_size is not None else int(dataset_cfg["tile_size"])
    stride = args.stride if args.stride is not None else int(dataset_cfg.get("stride", tile_size))
    include_edge_tiles = (
        bool(args.include_edge_tiles)
        if args.include_edge_tiles is not None
        else bool(dataset_cfg.get("include_edge_tiles", False))
    )
    train_negative_ratio = (
        args.train_negative_ratio
        if args.train_negative_ratio is not None
        else dataset_cfg.get("train_negative_ratio")
    )
    eval_max_per_stratum = (
        args.eval_max_per_stratum
        if args.eval_max_per_stratum is not None
        else eval_cfg.get("max_per_stratum")
    )

    summaries = prepare_isaid_vehicle_dataset(
        raw_root=resolve_path(config["paths"]["raw_dataset_dir"]),
        output_root=resolve_path(config["paths"]["prepared_dataset_dir"]),
        target_category_names=set(dataset_cfg["target_categories"]),
        merged_category_name=str(dataset_cfg["merged_category"]),
        train_split=str(dataset_cfg["train_split"]),
        val_split=str(dataset_cfg["val_split"]),
        eval_split=str(dataset_cfg["eval_split"]),
        tile_size=tile_size,
        stride=stride,
        include_edge_tiles=include_edge_tiles,
        image_format=str(dataset_cfg["tile_image_format"]),
        min_instance_area=int(dataset_cfg["min_instance_area"]),
        train_negative_ratio=float(train_negative_ratio) if train_negative_ratio is not None else None,
        sampling_seed=int(dataset_cfg.get("sampling_seed", 42)),
        min_eval_objects=int(eval_cfg["min_objects_per_image"]),
        max_eval_objects=(
            int(eval_cfg["max_objects_per_image"])
            if eval_cfg.get("max_objects_per_image") is not None
            else None
        ),
        eval_overlap_iou_threshold=float(eval_cfg.get("overlap_iou_threshold", 0.0)),
        eval_no_overlap_iou_max=(
            float(eval_cfg["no_overlap_iou_max"])
            if eval_cfg.get("no_overlap_iou_max") is not None
            else None
        ),
        eval_overlap_iou_min=(
            float(eval_cfg["overlap_iou_min"])
            if eval_cfg.get("overlap_iou_min") is not None
            else None
        ),
        eval_area_threshold=eval_cfg.get("area_threshold", "median"),
        eval_stratify_by=str(eval_cfg.get("stratify_by", "mask_area")),
        eval_low_object_count_max=(
            int(eval_cfg["low_object_count_max"])
            if eval_cfg.get("low_object_count_max") is not None
            else None
        ),
        eval_high_object_count_min=(
            int(eval_cfg["high_object_count_min"])
            if eval_cfg.get("high_object_count_min") is not None
            else None
        ),
        eval_max_per_stratum=int(eval_max_per_stratum) if eval_max_per_stratum is not None else None,
        eval_balance_to_smallest_stratum=bool(eval_cfg.get("balance_to_smallest_stratum", True)),
    )

    print("Prepared iSAID aerial vehicle dataset")
    for split, summary in summaries.items():
        print(
            f"{split}: images={summary.images}, positive={summary.positive_images}, "
            f"negative={summary.negative_images}, annotations={summary.annotations}"
        )
    print(f"dataset_root: {resolve_path(config['paths']['prepared_dataset_dir'])}")


if __name__ == "__main__":
    main()
