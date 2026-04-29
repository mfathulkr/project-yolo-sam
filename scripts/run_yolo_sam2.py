from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam3_bbox_study.config import load_config, resolve_path
from sam3_bbox_study.pipelines.yolo_sam2 import run_yolo_sam2_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO bbox + SAM2 segmentation on the evaluation split.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "isaid_vehicle_yolo26x_cpu_eval.yaml")
    parser.add_argument("--weights", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)

    split = config["dataset"]["eval_split"]
    images_dir = resolve_path(config["paths"]["prepared_dataset_dir"]) / split / "images"
    output_dir = resolve_path(config["paths"]["yolo_sam2_output_dir"])
    sam2_cfg = config["sam2"]

    run_yolo_sam2_pipeline(
        images_dir=images_dir,
        output_dir=output_dir,
        yolo_weights=args.weights or config["yolo"]["trained_weights"],
        conf_threshold=config["yolo"]["conf"],
        image_size=config["yolo"]["imgsz"],
        yolo_device=config["yolo"]["device"],
        max_det=config["yolo"].get("max_det"),
        model_id=sam2_cfg["model_id"],
        sam_device=sam2_cfg["device"],
        torch_dtype=sam2_cfg["torch_dtype"],
        mask_threshold=sam2_cfg["mask_threshold"],
        box_batch_size=int(sam2_cfg.get("box_batch_size", 16)),
        hf_token=os.getenv("HF_TOKEN"),
    )


if __name__ == "__main__":
    main()
