from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pool_segmentation_compare.config import load_config, resolve_path
from pool_segmentation_compare.evaluation.runner import evaluate_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate both segmentation pipelines with IoU.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split = config["dataset"]["eval_split"]

    df = evaluate_experiment(
        coco_dir=resolve_path(config["paths"]["prepared_dataset_dir"]) / split,
        pipeline_a_dir=resolve_path(config["paths"]["pipeline_a_output_dir"]),
        pipeline_b_dir=resolve_path(config["paths"]["pipeline_b_output_dir"]),
        metrics_dir=resolve_path(config["paths"]["metrics_dir"]),
        visualizations_dir=resolve_path(config["paths"]["visualizations_dir"]),
        max_visualizations=config["evaluation"]["max_visualizations"],
        positive_only=config["evaluation"]["positive_only"],
    )

    print(df.head())
    if not df.empty:
        print(f"Mean IoU Pipeline A: {df['iou_pipeline_a'].mean():.4f}")
        print(f"Mean IoU Pipeline B: {df['iou_pipeline_b'].mean():.4f}")


if __name__ == "__main__":
    main()
