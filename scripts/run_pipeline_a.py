from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pool_segmentation_compare.config import load_config, resolve_path
from pool_segmentation_compare.pipelines.pipeline_a import run_yolo_sam_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO -> SAM2 pipeline on the evaluation split.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--weights", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    split = config["dataset"]["eval_split"]
    images_dir = resolve_path(config["paths"]["prepared_dataset_dir"]) / split / "images"
    output_dir = resolve_path(config["paths"]["pipeline_a_output_dir"])

    run_yolo_sam_pipeline(
        images_dir=images_dir,
        output_dir=output_dir,
        yolo_weights=args.weights or config["yolo"]["trained_weights"],
        sam_checkpoint=config["sam2"]["checkpoint"],
        conf_threshold=config["yolo"]["conf"],
        image_size=config["yolo"]["imgsz"],
        yolo_device=config["yolo"]["device"],
        sam_device=config["sam2"]["device"],
    )


if __name__ == "__main__":
    main()
