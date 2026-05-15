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
from sam3_bbox_study.pipelines.hybrid_sam3 import run_sam3_hybrid_gt_pipeline, run_sam3_hybrid_yolo_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SAM3 with text plus GT or YOLO bounding-box prompts.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "isaid_vehicle_yolo26x_cpu_eval.yaml")
    parser.add_argument("--box-source", choices=["gt", "yolo"], required=True)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--sam-device", type=str, default=None)
    parser.add_argument("--torch-dtype", type=str, default=None)
    parser.add_argument("--yolo-device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)

    split = config["dataset"]["eval_split"]
    prepared_dir = resolve_path(config["paths"]["prepared_dataset_dir"])
    images_dir = prepared_dir / split / "images"
    coco_dir = prepared_dir / split
    sam3_cfg = config["sam3"]
    prompt = args.prompt or sam3_cfg["prompt"]
    common = {
        "model_dir": resolve_path(sam3_cfg["local_model_dir"]),
        "prompt": prompt,
        "sam_device": args.sam_device or sam3_cfg["device"],
        "torch_dtype": args.torch_dtype or sam3_cfg["torch_dtype"],
        "output_prob_thresh": sam3_cfg["output_prob_thresh"],
        "mask_threshold": sam3_cfg["mask_threshold"],
        "hf_token": os.getenv("HF_TOKEN"),
    }

    if args.box_source == "gt":
        run_sam3_hybrid_gt_pipeline(
            images_dir=images_dir,
            coco_dir=coco_dir,
            output_dir=resolve_path(config["paths"]["sam3_hybrid_gt_output_dir"]),
            **common,
        )
    else:
        run_sam3_hybrid_yolo_pipeline(
            images_dir=images_dir,
            output_dir=resolve_path(config["paths"]["sam3_hybrid_yolo_output_dir"]),
            yolo_weights=args.weights or config["yolo"]["trained_weights"],
            conf_threshold=config["yolo"]["conf"],
            image_size=config["yolo"]["imgsz"],
            yolo_device=args.yolo_device or config["yolo"]["device"],
            max_det=config["yolo"].get("max_det"),
            **common,
        )


if __name__ == "__main__":
    main()
