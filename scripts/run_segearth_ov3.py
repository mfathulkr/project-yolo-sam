from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam3_bbox_study.config import load_config, resolve_path
from sam3_bbox_study.pipelines.segearth_ov3 import run_segearth_ov3_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SegEarth-OV3-style SAM3 open-vocabulary segmentation.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "isaid_vehicle_yolo26x_cpu_eval.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for smoke runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split = config["dataset"]["eval_split"]
    images_dir = resolve_path(config["paths"]["prepared_dataset_dir"]) / split / "images"
    output_dir = resolve_path(config["paths"]["segearth_ov3_output_dir"])
    segearth_cfg = config["segearth_ov3"]

    run_segearth_ov3_pipeline(
        images_dir=images_dir,
        output_dir=output_dir,
        external_root=resolve_path(segearth_cfg["external_root"]),
        checkpoint_path=resolve_path(segearth_cfg["checkpoint_path"]),
        bpe_path=resolve_path(segearth_cfg["bpe_path"]),
        prompts=[str(prompt) for prompt in segearth_cfg["prompts"]],
        device=segearth_cfg.get("device", "cpu"),
        confidence_threshold=float(segearth_cfg.get("confidence_threshold", 0.4)),
        prob_threshold=float(segearth_cfg.get("prob_threshold", 0.5)),
        resolution=int(segearth_cfg.get("resolution", 1008)),
        use_semantic_head=bool(segearth_cfg.get("use_semantic_head", True)),
        use_instance_head=bool(segearth_cfg.get("use_instance_head", True)),
        use_presence_score=bool(segearth_cfg.get("use_presence_score", True)),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
