from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam3_bbox_study.config import load_config, resolve_path
from sam3_bbox_study.data.coco_masks import load_ground_truth_masks
from sam3_bbox_study.evaluation.metrics import compute_mask_metrics
from sam3_bbox_study.evaluation.visualization import save_sam3_triplet_comparison_figure
from sam3_bbox_study.io_utils import ensure_dir, load_binary_mask


PIPELINES = [
    ("sam3_text", "SAM3 text-only", "iou_text_sam3", "sam3_text_output_dir"),
    ("yolo_sam3", "YOLO + SAM3", "iou_yolo_sam3", "yolo_sam3_output_dir"),
    ("gt_box_sam3", "GT bbox + SAM3", "iou_gt_box_sam3", "gt_box_sam3_output_dir"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAM3 triplet metrics by bbox-overlap and mask-area strata.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "isaid_vehicle_yolo26x.yaml")
    return parser.parse_args()


def bootstrap_ci(values: pd.Series, seed: int = 0, n_boot: int = 3000) -> tuple[float, float]:
    if values.empty:
        return 0.0, 0.0
    samples = values.sample(n=len(values) * n_boot, replace=True, random_state=seed).to_numpy()
    means = samples.reshape(n_boot, len(values)).mean(axis=1)
    low, high = pd.Series(means).quantile([0.025, 0.975]).tolist()
    return float(low), float(high)


def summarize_long_metrics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_columns = ["iou", "dice", "precision", "recall", "pred_area_ratio"]
    summary_overall = (
        metrics.groupby(["pipeline", "pipeline_label"], as_index=False)
        .agg(
            images=("image", "nunique"),
            mean_iou=("iou", "mean"),
            median_iou=("iou", "median"),
            mean_dice=("dice", "mean"),
            mean_precision=("precision", "mean"),
            mean_recall=("recall", "mean"),
            mean_pred_area_ratio=("pred_area_ratio", "mean"),
            zero_iou_images=("iou", lambda values: int((values == 0).sum())),
        )
        .sort_values("pipeline")
    )
    summary_by_stratum = (
        metrics.groupby(["stratum", "overlap_group", "area_group", "pipeline", "pipeline_label"], as_index=False)
        .agg(
            images=("image", "nunique"),
            mean_objects=("num_objects", "mean"),
            mean_mask_area_ratio=("mask_area_ratio", "mean"),
            **{f"mean_{column}": (column, "mean") for column in metric_columns},
            zero_iou_images=("iou", lambda values: int((values == 0).sum())),
        )
        .sort_values(["stratum", "pipeline"])
    )

    wide = metrics.pivot_table(index=["image", "stratum"], columns="pipeline", values="iou", aggfunc="first").reset_index()
    pair_rows: list[dict[str, object]] = []
    comparisons = [
        ("yolo_sam3", "sam3_text", "YOLO + SAM3 - SAM3 text-only"),
        ("gt_box_sam3", "yolo_sam3", "GT bbox + SAM3 - YOLO + SAM3"),
        ("gt_box_sam3", "sam3_text", "GT bbox + SAM3 - SAM3 text-only"),
    ]
    for stratum, group in wide.groupby("stratum"):
        for left, right, label in comparisons:
            if left not in group or right not in group:
                continue
            diff = group[left] - group[right]
            low, high = bootstrap_ci(diff)
            pair_rows.append(
                {
                    "stratum": stratum,
                    "comparison": label,
                    "images": len(diff),
                    "mean_difference": float(diff.mean()),
                    "ci95_low": low,
                    "ci95_high": high,
                    "left_wins": int((diff > 0).sum()),
                    "ties": int((diff == 0).sum()),
                    "right_wins": int((diff < 0).sum()),
                }
            )
    pairwise = pd.DataFrame(pair_rows)
    return summary_overall, summary_by_stratum, pairwise


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split = config["dataset"]["eval_split"]

    prepared_split_dir = resolve_path(config["paths"]["prepared_dataset_dir"]) / split
    metadata = pd.read_csv(prepared_split_dir / "metadata.csv")
    gt_masks = load_ground_truth_masks(prepared_split_dir)

    rows: list[dict[str, object]] = []
    for _, metadata_row in metadata.iterrows():
        image_name = str(metadata_row["file_name"])
        gt_mask = gt_masks[image_name]
        if config["evaluation"].get("positive_only", True) and not gt_mask.any():
            continue

        for pipeline, pipeline_label, _, output_key in PIPELINES:
            mask_dir = resolve_path(config["paths"][output_key]) / "masks"
            pred_mask = load_binary_mask(mask_dir / f"{Path(image_name).stem}.png", gt_mask.shape)
            metric_values = compute_mask_metrics(pred_mask, gt_mask)
            rows.append(
                {
                    "image": image_name,
                    "pipeline": pipeline,
                    "pipeline_label": pipeline_label,
                    "stratum": metadata_row.get("stratum", ""),
                    "overlap_group": metadata_row.get("overlap_group", ""),
                    "area_group": metadata_row.get("area_group", ""),
                    "num_objects": int(metadata_row["num_objects"]),
                    "mask_area_ratio": float(metadata_row["mask_area_ratio"]),
                    "max_pair_bbox_iou": float(metadata_row["max_pair_bbox_iou"]),
                    **metric_values,
                }
            )

    metrics = pd.DataFrame(rows)
    metrics_dir = ensure_dir(resolve_path(config["paths"]["sam3_triplet_metrics_dir"]))
    metrics.to_csv(metrics_dir / "per_image_stratified_metrics.csv", index=False)

    summary_overall, summary_by_stratum, pairwise = summarize_long_metrics(metrics)
    summary_overall.to_csv(metrics_dir / "summary_overall_stratified.csv", index=False)
    summary_by_stratum.to_csv(metrics_dir / "summary_by_stratum.csv", index=False)
    pairwise.to_csv(metrics_dir / "pairwise_iou_by_stratum.csv", index=False)

    visualizations_dir = ensure_dir(resolve_path(config["paths"]["sam3_triplet_visualizations_dir"]))
    max_per_stratum = int(config["evaluation"].get("max_visualizations_per_stratum", 3))
    if max_per_stratum > 0:
        selected_images = (
            metadata.sort_values(["stratum", "file_name"])
            .groupby("stratum")
            .head(max_per_stratum)["file_name"]
            .tolist()
        )
        for image_name in selected_images:
            gt_mask = gt_masks[image_name]
            stem = Path(image_name).stem
            save_sam3_triplet_comparison_figure(
                image_path=prepared_split_dir / "images" / image_name,
                gt_mask=gt_mask,
                text_only_mask=load_binary_mask(
                    resolve_path(config["paths"]["sam3_text_output_dir"]) / "masks" / f"{stem}.png",
                    gt_mask.shape,
                ),
                yolo_sam3_mask=load_binary_mask(
                    resolve_path(config["paths"]["yolo_sam3_output_dir"]) / "masks" / f"{stem}.png",
                    gt_mask.shape,
                ),
                gt_box_sam3_mask=load_binary_mask(
                    resolve_path(config["paths"]["gt_box_sam3_output_dir"]) / "masks" / f"{stem}.png",
                    gt_mask.shape,
                ),
                output_path=visualizations_dir / f"{stem}.png",
            )

    print(summary_overall)
    print(summary_by_stratum)


if __name__ == "__main__":
    main()
