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
from yolo_sam.data.isaid import create_balanced_eval_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild only the balanced iSAID eval split.")
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "configs" / "yolo26x_cpu_eval.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    eval_cfg = config["evaluation"]
    prepared_root = resolve_path(config["paths"]["prepared_dataset_dir"])

    summary = create_balanced_eval_split(
        dataset_root=prepared_root,
        source_split=str(dataset_cfg["val_split"]),
        eval_split=str(dataset_cfg["eval_split"]),
        min_objects=int(eval_cfg["min_objects_per_image"]),
        max_objects=(
            int(eval_cfg["max_objects_per_image"])
            if eval_cfg.get("max_objects_per_image") is not None
            else None
        ),
        overlap_iou_threshold=float(eval_cfg.get("overlap_iou_threshold", 0.0)),
        no_overlap_iou_max=(
            float(eval_cfg["no_overlap_iou_max"])
            if eval_cfg.get("no_overlap_iou_max") is not None
            else None
        ),
        overlap_iou_min=(
            float(eval_cfg["overlap_iou_min"])
            if eval_cfg.get("overlap_iou_min") is not None
            else None
        ),
        area_threshold=eval_cfg.get("area_threshold", "median"),
        stratify_by=str(eval_cfg.get("stratify_by", "mask_area")),
        low_object_count_max=(
            int(eval_cfg["low_object_count_max"])
            if eval_cfg.get("low_object_count_max") is not None
            else None
        ),
        high_object_count_min=(
            int(eval_cfg["high_object_count_min"])
            if eval_cfg.get("high_object_count_min") is not None
            else None
        ),
        max_per_stratum=(
            int(eval_cfg["max_per_stratum"])
            if eval_cfg.get("max_per_stratum") is not None
            else None
        ),
        balance_to_smallest_stratum=bool(eval_cfg.get("balance_to_smallest_stratum", True)),
        sampling_seed=int(dataset_cfg.get("sampling_seed", 42)),
    )

    print(
        f"eval: images={summary.images}, positive={summary.positive_images}, "
        f"negative={summary.negative_images}, annotations={summary.annotations}"
    )
    print(f"metadata: {prepared_root / dataset_cfg['eval_split'] / 'metadata.csv'}")


if __name__ == "__main__":
    main()
