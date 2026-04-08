from __future__ import annotations

from pathlib import Path

import pandas as pd

from sam3_bbox_study.data.coco_masks import load_ground_truth_masks
from sam3_bbox_study.evaluation.metrics import compute_iou
from sam3_bbox_study.evaluation.visualization import save_sam3_triplet_comparison_figure
from sam3_bbox_study.io_utils import ensure_dir, load_binary_mask


def evaluate_sam3_triplet(
    coco_dir: Path,
    text_only_dir: Path,
    yolo_sam3_dir: Path,
    gt_sam3_dir: Path,
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
        pred_text = load_binary_mask(text_only_dir / "masks" / f"{stem}.png", gt_mask.shape)
        pred_yolo = load_binary_mask(yolo_sam3_dir / "masks" / f"{stem}.png", gt_mask.shape)
        pred_gt = load_binary_mask(gt_sam3_dir / "masks" / f"{stem}.png", gt_mask.shape)

        rows.append(
            {
                "image": file_name,
                "has_ground_truth": has_ground_truth,
                "iou_text_sam3": compute_iou(pred_text, gt_mask),
                "iou_yolo_sam3": compute_iou(pred_yolo, gt_mask),
                "iou_gt_box_sam3": compute_iou(pred_gt, gt_mask),
            }
        )

        if visuals_written < max_visualizations:
            image_path = coco_dir / "images" / file_name
            if not image_path.exists():
                image_path = coco_dir / file_name
            save_sam3_triplet_comparison_figure(
                image_path=image_path,
                gt_mask=gt_mask,
                text_only_mask=pred_text,
                yolo_sam3_mask=pred_yolo,
                gt_box_sam3_mask=pred_gt,
                output_path=visualizations_dir / f"{stem}.png",
            )
            visuals_written += 1

    df = pd.DataFrame(rows)
    ensure_dir(metrics_dir)
    df.to_csv(metrics_dir / "per_image_iou_sam3_triplet.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "mean_iou_text_sam3": df["iou_text_sam3"].mean() if not df.empty else 0.0,
                "mean_iou_yolo_sam3": df["iou_yolo_sam3"].mean() if not df.empty else 0.0,
                "mean_iou_gt_box_sam3": df["iou_gt_box_sam3"].mean() if not df.empty else 0.0,
                "num_images": len(df),
                "positive_only": positive_only,
            }
        ]
    )
    summary.to_csv(metrics_dir / "summary_sam3_triplet.csv", index=False)
    return df
