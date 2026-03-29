from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pool_segmentation_compare.config import load_config, resolve_path
from pool_segmentation_compare.data.isaid import prepare_single_class_split, write_yolo_data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a swimming-pool-only iSAID subset for YOLO+IoU evaluation.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--link-method", choices=["hardlink", "copy"], default="hardlink")
    parser.add_argument("--positive-only-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = config["dataset"]

    category_name = args.category or dataset_cfg["target_category"]
    keep_all_images = not args.positive_only_images and bool(dataset_cfg["keep_all_images"])
    raw_root = resolve_path(config["paths"]["raw_dataset_dir"])
    prepared_root = resolve_path(config["paths"]["prepared_dataset_dir"])

    train_split = dataset_cfg["train_split"]
    eval_split = dataset_cfg["eval_split"]

    train_summary = prepare_single_class_split(
        split=train_split,
        raw_split_root=raw_root / train_split,
        output_split_root=prepared_root / train_split,
        category_name=category_name,
        keep_all_images=keep_all_images,
        link_method=args.link_method,
    )
    eval_summary = prepare_single_class_split(
        split=eval_split,
        raw_split_root=raw_root / eval_split,
        output_split_root=prepared_root / eval_split,
        category_name=category_name,
        keep_all_images=keep_all_images,
        link_method=args.link_method,
    )

    write_yolo_data_yaml(prepared_root, train_split=train_split, val_split=eval_split, class_name=category_name)

    print("Prepared iSAID subset")
    print(f"train: {train_summary}")
    print(f"{eval_split}: {eval_summary}")
    print(f"dataset_root: {prepared_root}")


if __name__ == "__main__":
    main()
