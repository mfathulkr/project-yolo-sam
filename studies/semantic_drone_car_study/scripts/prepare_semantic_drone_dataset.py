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
from semantic_drone_car.data import prepare_semantic_drone_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Semantic Drone Dataset car tiles for YOLO + SAM3.")
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "configs" / "yolo26x.yaml")
    parser.add_argument("--tile-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--eval-max-per-stratum", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    eval_cfg = config["evaluation"]

    tile_size = args.tile_size if args.tile_size is not None else int(dataset_cfg["tile_size"])
    stride = args.stride if args.stride is not None else int(dataset_cfg.get("stride", tile_size))
    val_fraction = args.val_fraction if args.val_fraction is not None else float(dataset_cfg["val_fraction"])
    eval_max_per_stratum = (
        args.eval_max_per_stratum
        if args.eval_max_per_stratum is not None
        else eval_cfg.get("max_per_stratum")
    )

    summaries = prepare_semantic_drone_dataset(
        raw_root=resolve_path(config["paths"]["raw_dataset_dir"]),
        output_root=resolve_path(config["paths"]["prepared_dataset_dir"]),
        image_dir=dataset_cfg.get("image_dir"),
        mask_dir=dataset_cfg.get("mask_dir"),
        target_categories=list(dataset_cfg["target_categories"]),
        target_category_ids=dataset_cfg.get("target_category_ids"),
        target_rgb_colors=dataset_cfg.get("target_rgb_colors"),
        tile_size=tile_size,
        stride=stride,
        include_edge_tiles=bool(dataset_cfg.get("include_edge_tiles", True)),
        image_format=str(dataset_cfg["tile_image_format"]),
        min_component_area=int(dataset_cfg["min_component_area"]),
        val_fraction=val_fraction,
        train_split=str(dataset_cfg["train_split"]),
        val_split=str(dataset_cfg["val_split"]),
        eval_split=str(dataset_cfg["eval_split"]),
        train_negative_ratio=(
            float(dataset_cfg["train_negative_ratio"])
            if dataset_cfg.get("train_negative_ratio") is not None
            else None
        ),
        sampling_seed=int(dataset_cfg.get("sampling_seed", 42)),
        min_eval_objects=int(eval_cfg["min_objects_per_image"]),
        max_eval_objects=(
            int(eval_cfg["max_objects_per_image"])
            if eval_cfg.get("max_objects_per_image") is not None
            else None
        ),
        eval_overlap_iou_threshold=float(eval_cfg.get("overlap_iou_threshold", 0.0)),
        eval_area_threshold=eval_cfg.get("area_threshold", "median"),
        eval_max_per_stratum=int(eval_max_per_stratum) if eval_max_per_stratum is not None else None,
        eval_balance_to_smallest_stratum=bool(eval_cfg.get("balance_to_smallest_stratum", True)),
    )

    print("Prepared Semantic Drone car dataset")
    for split, summary in summaries.items():
        print(
            f"{split}: images={summary.images}, positive={summary.positive_images}, "
            f"negative={summary.negative_images}, annotations={summary.annotations}"
        )
    print(f"dataset_root: {resolve_path(config['paths']['prepared_dataset_dir'])}")


if __name__ == "__main__":
    main()
