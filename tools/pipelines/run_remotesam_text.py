from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yolo_sam.config import load_config, resolve_path
from yolo_sam.pipelines.remotesam_text import run_remotesam_text_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RemoteSAM text/referring semantic segmentation on eval split.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for smoke runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split = config["dataset"]["eval_split"]
    images_dir = resolve_path(config["paths"]["prepared_dataset_dir"]) / split / "images"
    output_dir = resolve_path(config["paths"]["remotesam_text_output_dir"])
    remotesam_cfg = config["remotesam_text"]

    run_remotesam_text_pipeline(
        images_dir=images_dir,
        output_dir=output_dir,
        external_root=resolve_path(remotesam_cfg["external_root"]),
        checkpoint_path=resolve_path(remotesam_cfg["checkpoint_path"]),
        prompts=[str(prompt) for prompt in remotesam_cfg["prompts"]],
        device=remotesam_cfg.get("device", "cpu"),
        prob_threshold=float(remotesam_cfg.get("prob_threshold", 0.1)),
        use_epoc=bool(remotesam_cfg.get("use_epoc", False)),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
