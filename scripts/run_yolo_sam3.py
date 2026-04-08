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

from pool_segmentation_compare.config import load_config, resolve_path
from pool_segmentation_compare.pipelines.pipeline_c import run_yolo_sam3_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO bbox + SAM3 segmentation on the evaluation split.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)

    split = config["dataset"]["eval_split"]
    images_dir = resolve_path(config["paths"]["prepared_dataset_dir"]) / split / "images"
    output_dir = resolve_path(config["paths"]["yolo_sam3_output_dir"])
    prompt = args.prompt or config["sam3"]["prompt"]

    run_yolo_sam3_pipeline(
        images_dir=images_dir,
        output_dir=output_dir,
        yolo_weights=args.weights or config["yolo"]["trained_weights"],
        conf_threshold=config["yolo"]["conf"],
        image_size=config["yolo"]["imgsz"],
        yolo_device=config["yolo"]["device"],
        model_dir=resolve_path(config["sam3"]["local_model_dir"]),
        prompt=prompt,
        sam_device=config["sam3"]["device"],
        torch_dtype=config["sam3"]["torch_dtype"],
        output_prob_thresh=config["sam3"]["output_prob_thresh"],
        mask_threshold=config["sam3"]["mask_threshold"],
        hf_token=os.getenv("HF_TOKEN"),
    )


if __name__ == "__main__":
    main()
