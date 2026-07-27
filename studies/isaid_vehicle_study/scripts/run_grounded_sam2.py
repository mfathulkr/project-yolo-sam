from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.config import load_config, resolve_path
from isaid_vehicle.pipelines.grounded_sam2 import run_grounded_sam2_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GroundingDINO text boxes + SAM2 segmentation on eval split.")
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "configs" / "yolo26x_cpu_eval.yaml")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)
    split = config["dataset"]["eval_split"]
    images_dir = resolve_path(config["paths"]["prepared_dataset_dir"]) / split / "images"
    output_dir = resolve_path(config["paths"]["grounded_sam2_output_dir"])
    grounded_cfg = config["grounded_sam2"]
    sam2_cfg = config["sam2"]

    run_grounded_sam2_pipeline(
        images_dir=images_dir,
        output_dir=output_dir,
        detector_model_id=grounded_cfg["detector_model_id"],
        detector_device=grounded_cfg["device"],
        detector_torch_dtype=grounded_cfg["torch_dtype"],
        text_prompt=grounded_cfg["prompt"],
        threshold=float(grounded_cfg["threshold"]),
        text_threshold=float(grounded_cfg["text_threshold"]),
        nms_iou=float(grounded_cfg["nms_iou"]),
        max_boxes=int(grounded_cfg["max_boxes"]),
        sam2_model_id=sam2_cfg["model_id"],
        sam2_device=sam2_cfg["device"],
        sam2_torch_dtype=sam2_cfg["torch_dtype"],
        mask_threshold=float(sam2_cfg["mask_threshold"]),
        box_batch_size=int(sam2_cfg.get("box_batch_size", 16)),
        hf_token=os.getenv("HF_TOKEN"),
    )


if __name__ == "__main__":
    main()
