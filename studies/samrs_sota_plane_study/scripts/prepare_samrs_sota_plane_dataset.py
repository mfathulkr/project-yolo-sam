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
from yolo_sam.data.samrs import prepare_samrs_sota_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SAMRS SOTA plane data for YOLO + SAM experiments.")
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "configs" / "yolo26x.yaml")
    parser.add_argument("--eval-max-per-stratum", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    eval_cfg = config["evaluation"]

    eval_max_per_stratum = (
        args.eval_max_per_stratum
        if args.eval_max_per_stratum is not None
        else eval_cfg.get("max_per_stratum")
    )

    summaries = prepare_samrs_sota_dataset(
        raw_root=resolve_path(config["paths"]["raw_dataset_dir"]),
        output_root=resolve_path(config["paths"]["prepared_dataset_dir"]),
        target_category_names=set(dataset_cfg["target_categories"]),
        merged_category_name=str(dataset_cfg["merged_category"]),
        train_split=str(dataset_cfg["train_split"]),
        val_split=str(dataset_cfg["val_split"]),
        eval_split=str(dataset_cfg["eval_split"]),
        mask_subdir=str(dataset_cfg.get("mask_subdir", "rhbox_segs_init/ins")),
        image_format=str(dataset_cfg["tile_image_format"]),
        min_instance_area=int(dataset_cfg["min_instance_area"]),
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

    print("Prepared SAMRS SOTA plane dataset")
    for split, summary in summaries.items():
        print(
            f"{split}: images={summary.images}, positive={summary.positive_images}, "
            f"negative={summary.negative_images}, annotations={summary.annotations}"
        )
    print(f"dataset_root: {resolve_path(config['paths']['prepared_dataset_dir'])}")


if __name__ == "__main__":
    main()
