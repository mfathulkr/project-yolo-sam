from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam3_bbox_study.config import load_config, resolve_path
from sam3_bbox_study.evaluation.triplet import evaluate_sam3_triplet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM3 text-only, YOLO+SAM3, and GT bbox+SAM3.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split = config["dataset"]["eval_split"]

    df = evaluate_sam3_triplet(
        coco_dir=resolve_path(config["paths"]["prepared_dataset_dir"]) / split,
        text_only_dir=resolve_path(config["paths"]["sam3_text_output_dir"]),
        yolo_sam3_dir=resolve_path(config["paths"]["yolo_sam3_output_dir"]),
        gt_sam3_dir=resolve_path(config["paths"]["gt_box_sam3_output_dir"]),
        metrics_dir=resolve_path(config["paths"]["sam3_triplet_metrics_dir"]),
        visualizations_dir=resolve_path(config["paths"]["sam3_triplet_visualizations_dir"]),
        max_visualizations=config["evaluation"]["max_visualizations"],
        positive_only=config["evaluation"]["positive_only"],
    )

    print(df.head())
    if not df.empty:
        print(f"Mean IoU SAM3 text-only: {df['iou_text_sam3'].mean():.4f}")
        print(f"Mean IoU YOLO + SAM3: {df['iou_yolo_sam3'].mean():.4f}")
        print(f"Mean IoU GT bbox + SAM3: {df['iou_gt_box_sam3'].mean():.4f}")


if __name__ == "__main__":
    main()
