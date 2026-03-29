from __future__ import annotations

from pathlib import Path

import pandas as pd

from pool_segmentation_compare.data.coco_masks import load_ground_truth_masks
from pool_segmentation_compare.evaluation.metrics import compute_iou
from pool_segmentation_compare.evaluation.visualization import save_comparison_figure
from pool_segmentation_compare.io_utils import ensure_dir, load_binary_mask


def evaluate_experiment(
    coco_dir: Path,
    pipeline_a_dir: Path,
    pipeline_b_dir: Path,
    metrics_dir: Path,
    visualizations_dir: Path,
    max_visualizations: int,
    positive_only: bool,
) -> pd.DataFrame:
    gt_masks = load_ground_truth_masks(coco_dir)
    rows: list[dict[str, object]] = []
    visuals_written = 0

    for file_name, gt_mask in gt_masks.items():
        has_ground_truth = bool(gt_mask.any())
        if positive_only and not has_ground_truth:
            continue

        stem = Path(file_name).stem
        pred_a = load_binary_mask(pipeline_a_dir / "masks" / f"{stem}.png", gt_mask.shape)
        pred_b = load_binary_mask(pipeline_b_dir / "masks" / f"{stem}.png", gt_mask.shape)

        row = {
            "image": file_name,
            "has_ground_truth": has_ground_truth,
            "iou_pipeline_a": compute_iou(pred_a, gt_mask),
            "iou_pipeline_b": compute_iou(pred_b, gt_mask),
        }
        rows.append(row)

        if visuals_written < max_visualizations:
            image_path = coco_dir / "images" / file_name
            if not image_path.exists():
                image_path = coco_dir / file_name
            save_comparison_figure(
                image_path=image_path,
                gt_mask=gt_mask,
                pipeline_a_mask=pred_a,
                pipeline_b_mask=pred_b,
                output_path=visualizations_dir / f"{stem}.png",
            )
            visuals_written += 1

    df = pd.DataFrame(rows)
    ensure_dir(metrics_dir)
    df.to_csv(metrics_dir / "per_image_iou.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "mean_iou_pipeline_a": df["iou_pipeline_a"].mean() if not df.empty else 0.0,
                "mean_iou_pipeline_b": df["iou_pipeline_b"].mean() if not df.empty else 0.0,
                "num_images": len(df),
                "positive_only": positive_only,
            }
        ]
    )
    summary.to_csv(metrics_dir / "summary.csv", index=False)
    return df
