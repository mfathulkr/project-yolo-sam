from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yolo_sam.config import load_config, resolve_path
from yolo_sam.pipelines.gt_box_sam1 import run_gt_box_sam1_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ground-truth bbox + SAM1 segmentation on the evaluation split.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sam-device", type=str, default=None)
    parser.add_argument("--torch-dtype", type=str, default=None)
    parser.add_argument("--box-batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config = load_config(args.config)

    split = config["dataset"]["eval_split"]
    prepared_dir = resolve_path(config["paths"]["prepared_dataset_dir"])
    sam1_cfg = config["sam1"]

    run_gt_box_sam1_pipeline(
        images_dir=prepared_dir / split / "images",
        coco_dir=prepared_dir / split,
        output_dir=resolve_path(config["paths"]["sam1_gt_box_output_dir"]),
        model_id=sam1_cfg["model_id"],
        sam_device=args.sam_device or sam1_cfg["device"],
        torch_dtype=args.torch_dtype or sam1_cfg["torch_dtype"],
        mask_threshold=sam1_cfg["mask_threshold"],
        box_batch_size=int(args.box_batch_size or sam1_cfg.get("box_batch_size", 16)),
        hf_token=os.getenv("HF_TOKEN"),
    )


if __name__ == "__main__":
    main()
