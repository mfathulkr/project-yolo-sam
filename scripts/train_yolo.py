from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pool_segmentation_compare.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on the prepared LandCover.ai building dataset.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    yolo_cfg = config["yolo"]
    data_yaml = resolve_path(config["paths"]["prepared_dataset_dir"]) / "data.yaml"
    project_dir = resolve_path(config["paths"]["yolo_train_run_dir"])
    weights = args.weights or yolo_cfg["base_weights"]

    if args.resume:
        last_checkpoint = project_dir / "train" / "weights" / "last.pt"
        if not last_checkpoint.exists():
            raise SystemExit(f"Resume icin checkpoint bulunamadi: {last_checkpoint}")

        model = YOLO(str(last_checkpoint))
        model.train(resume=True)
        return

    model = YOLO(weights)
    model.train(
        data=str(data_yaml),
        epochs=yolo_cfg["epochs"],
        imgsz=yolo_cfg["imgsz"],
        batch=yolo_cfg["batch"],
        conf=yolo_cfg["conf"],
        workers=yolo_cfg.get("workers", 1),
        project=str(project_dir),
        name="train",
        exist_ok=True,
        device=yolo_cfg["device"],
    )


if __name__ == "__main__":
    main()
