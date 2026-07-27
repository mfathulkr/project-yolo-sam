from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prepared SAMRS SOTA plane data and eval strata.")
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "configs" / "yolo26x.yaml")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def summarize_split(dataset_root: Path, split: str) -> list[str]:
    metadata_path = dataset_root / split / "metadata.csv"
    if not metadata_path.exists():
        return [f"## {split}", "", f"Missing metadata: `{metadata_path}`", ""]

    metadata = pd.read_csv(metadata_path)
    positive = metadata[metadata["num_objects"] > 0].copy()
    lines = [f"## {split}", ""]
    lines.append(f"- Images: {len(metadata)}")
    lines.append(f"- Positive images: {len(positive)}")
    lines.append(f"- Instances: {int(metadata['num_objects'].sum()) if 'num_objects' in metadata else 0}")
    if not positive.empty:
        lines.append(f"- Mean objects/image: {positive['num_objects'].mean():.2f}")
        lines.append(f"- Median mask area ratio: {positive['mask_area_ratio'].median():.6f}")
        lines.append(f"- Mean bbox fill ratio: {positive['mean_bbox_fill_ratio'].mean():.4f}")
        lines.append(f"- Median bbox fill ratio: {positive['median_bbox_fill_ratio'].median():.4f}")
        lines.append(f"- Images with bbox overlap: {int((positive['max_pair_bbox_iou'] > 0).sum())}")
        lines.append(f"- Images with bbox IoU >= 0.05: {int((positive['max_pair_bbox_iou'] >= 0.05).sum())}")
    lines.append("")
    return lines


def eval_strata_table(dataset_root: Path, split: str) -> list[str]:
    metadata_path = dataset_root / split / "metadata.csv"
    if not metadata_path.exists():
        return []
    metadata = pd.read_csv(metadata_path)
    if "stratum" not in metadata:
        return []

    grouped = (
        metadata.groupby("stratum", as_index=False)
        .agg(
            images=("file_name", "nunique"),
            instances=("num_objects", "sum"),
            mean_objects=("num_objects", "mean"),
            mean_mask_area_ratio=("mask_area_ratio", "mean"),
            mean_bbox_fill_ratio=("mean_bbox_fill_ratio", "mean"),
            mean_max_pair_bbox_iou=("max_pair_bbox_iou", "mean"),
        )
        .sort_values("stratum")
    )
    lines = ["## Eval Strata", ""]
    lines.append("| Stratum | Images | Instances | Mean Objects | Mean Mask Area | Mean BBox Fill | Mean Max BBox IoU |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in grouped.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["stratum"]),
                    str(int(row["images"])),
                    str(int(row["instances"])),
                    f"{float(row['mean_objects']):.2f}",
                    f"{float(row['mean_mask_area_ratio']):.6f}",
                    f"{float(row['mean_bbox_fill_ratio']):.4f}",
                    f"{float(row['mean_max_pair_bbox_iou']):.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_root = resolve_path(config["paths"]["prepared_dataset_dir"])
    output = args.output or dataset_root / "AUDIT.md"

    lines = [
        "# SAMRS SOTA Plane Dataset Audit",
        "",
        "This audit verifies the prepared dataset used for the SOTA plane YOLO + SAM experiment.",
        "SAMRS masks are pseudo-labels generated from SAM over remote-sensing detection boxes, not manually drawn masks.",
        "",
    ]
    for split in [config["dataset"]["train_split"], config["dataset"]["val_split"], config["dataset"]["eval_split"]]:
        lines.extend(summarize_split(dataset_root, str(split)))
    lines.extend(eval_strata_table(dataset_root, str(config["dataset"]["eval_split"])))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
