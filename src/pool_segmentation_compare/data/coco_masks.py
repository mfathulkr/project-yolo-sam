from __future__ import annotations

from pathlib import Path

import numpy as np
from pycocotools.coco import COCO


def coco_annotation_path(coco_dir: Path) -> Path:
    return coco_dir / "_annotations.coco.json"


def load_ground_truth_masks(coco_dir: Path) -> dict[str, np.ndarray]:
    annotation_file = coco_annotation_path(coco_dir)
    coco = COCO(str(annotation_file))
    masks_by_file_name: dict[str, np.ndarray] = {}

    for image in coco.loadImgs(coco.getImgIds()):
        height = image["height"]
        width = image["width"]
        merged = np.zeros((height, width), dtype=bool)
        ann_ids = coco.getAnnIds(imgIds=[image["id"]])
        annotations = coco.loadAnns(ann_ids)

        for ann in annotations:
            if not ann.get("segmentation"):
                continue
            ann_mask = coco.annToMask(ann).astype(bool)
            merged |= ann_mask

        masks_by_file_name[image["file_name"]] = merged

    return masks_by_file_name
