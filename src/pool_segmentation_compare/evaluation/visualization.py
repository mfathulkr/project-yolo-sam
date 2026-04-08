from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    overlay = image.copy()
    overlay[mask.astype(bool)] = color
    return cv2.addWeighted(image, 0.7, overlay, 0.3, 0.0)


def save_sam3_triplet_comparison_figure(
    image_path: Path,
    gt_mask: np.ndarray,
    text_only_mask: np.ndarray,
    yolo_sam3_mask: np.ndarray,
    gt_box_sam3_mask: np.ndarray,
    output_path: Path,
) -> None:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[1].imshow(overlay_mask(image_rgb, gt_mask, (0, 255, 0)))
    axes[1].set_title("Ground Truth")
    axes[2].imshow(overlay_mask(image_rgb, text_only_mask, (0, 0, 255)))
    axes[2].set_title("SAM3 Text")
    axes[3].imshow(overlay_mask(image_rgb, yolo_sam3_mask, (255, 255, 0)))
    axes[3].set_title("YOLO + SAM3")
    axes[4].imshow(overlay_mask(image_rgb, gt_box_sam3_mask, (255, 0, 255)))
    axes[4].set_title("GT Box + SAM3")

    for axis in axes:
        axis.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
