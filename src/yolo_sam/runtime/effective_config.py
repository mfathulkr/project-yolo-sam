from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


BBOX_SEGMENTATION_CONFIG_SCHEMA = "bbox_segmentation_effective_v1"


def bbox_segmentation_effective_config(
    *,
    study_id: str,
    image_size: int,
    dataset: Mapping[str, Any],
    model: str,
    model_config: Mapping[str, Any],
    bbox_source: str,
    split: str,
    detector_seed: int | None = None,
    match_iou: float | None = None,
    selected_confidence_threshold: float | None = None,
) -> dict[str, Any]:
    """Return only the configuration that can affect one mask-inference stage."""
    if bbox_source not in {"gt_bbox", "yolo_bbox"}:
        raise ValueError(f"Unsupported bbox source: {bbox_source}")
    if bbox_source == "yolo_bbox" and detector_seed is None:
        raise ValueError("YOLO-bbox effective config requires detector_seed")

    dataset_fields = (
        "dataset_id",
        "version",
        "profile_id",
        "reference_type",
        "target_category",
        "area_threshold",
    )
    payload: dict[str, Any] = {
        "schema": BBOX_SEGMENTATION_CONFIG_SCHEMA,
        "study_id": study_id,
        "image_size": int(image_size),
        "dataset": {key: dataset.get(key) for key in dataset_fields},
        "stage": (
            "gt_bbox_segmentation"
            if bbox_source == "gt_bbox"
            else "yolo_bbox_segmentation"
        ),
        "bbox_source": bbox_source,
        "split": split,
        "model": model,
        "model_config": dict(model_config),
    }
    if bbox_source == "yolo_bbox":
        payload["detector_seed"] = int(detector_seed)
        payload["match_iou"] = float(match_iou) if match_iou is not None else None
        payload["selected_confidence_threshold"] = (
            float(selected_confidence_threshold)
            if selected_confidence_threshold is not None
            else None
        )
    return payload


def effective_config_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_effective_config_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
