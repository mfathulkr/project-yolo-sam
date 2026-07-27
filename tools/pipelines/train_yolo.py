from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yolo_sam.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO on the prepared LandCover.ai building dataset.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--name", type=str, default="train")
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
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs if args.epochs is not None else yolo_cfg["epochs"],
        "imgsz": args.imgsz if args.imgsz is not None else yolo_cfg["imgsz"],
        "batch": args.batch if args.batch is not None else yolo_cfg["batch"],
        "workers": args.workers if args.workers is not None else yolo_cfg.get("workers", 1),
        "project": str(project_dir),
        "name": args.name,
        "exist_ok": True,
        "device": args.device if args.device is not None else yolo_cfg["device"],
    }
    if "train_conf" in yolo_cfg:
        train_kwargs["conf"] = yolo_cfg["train_conf"]
    for optional_key in [
        "patience",
        "cache",
        "close_mosaic",
        "multi_scale",
        "optimizer",
        "lr0",
        "lrf",
        "max_det",
    ]:
        if optional_key in yolo_cfg:
            train_kwargs[optional_key] = yolo_cfg[optional_key]

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
