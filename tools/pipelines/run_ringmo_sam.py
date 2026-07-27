from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yolo_sam.config import load_config, resolve_path
from yolo_sam.pipelines.ringmo_sam import run_ringmo_sam_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RingMo-SAM with GT and YOLO box constraints.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split = config["dataset"]["eval_split"]
    prepared_split_dir = resolve_path(config["paths"]["prepared_dataset_dir"]) / split
    ringmo_cfg = config["ringmo_sam"]

    run_ringmo_sam_pipeline(
        images_dir=prepared_split_dir / "images",
        coco_dir=prepared_split_dir,
        gt_output_dir=resolve_path(config["paths"]["gt_box_ringmo_sam_output_dir"]),
        yolo_output_dir=resolve_path(config["paths"]["yolo_ringmo_sam_output_dir"]),
        yolo_raw_dir=resolve_path(config["paths"]["yolo_sam2_output_dir"]) / "raw",
        model_root=resolve_path(ringmo_cfg["model_root"]),
        config_path=resolve_path(ringmo_cfg["config_path"]),
        checkpoint_path=resolve_path(ringmo_cfg["checkpoint_path"]),
        device=str(ringmo_cfg.get("device", "cpu")),
        normalize=bool(ringmo_cfg.get("normalize", True)),
        class_ids=[int(value) for value in ringmo_cfg.get("class_ids", ringmo_cfg.get("vehicle_class_ids", [5]))],
    )


if __name__ == "__main__":
    main()
