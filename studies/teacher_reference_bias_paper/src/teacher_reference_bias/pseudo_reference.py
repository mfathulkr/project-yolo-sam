from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pycocotools import mask as mask_utils


def read_prediction_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_sam1_pseudo_reference_rows(
    predictions: list[dict[str, Any]],
    *,
    expected_model_id: str,
    expected_model_version: str,
) -> list[dict[str, Any]]:
    if not predictions:
        raise ValueError("SAM1 pseudo-reference source is empty")
    instance_ids = [str(row["instance_id"]) for row in predictions]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("SAM1 pseudo-reference source has duplicate instance IDs")

    rows = []
    for prediction in predictions:
        model_id = str(prediction["model_id"])
        model_version = str(prediction["model_version"])
        if (
            model_id != expected_model_id
            or model_version != expected_model_version
        ):
            raise ValueError(
                "Pseudo-reference teacher identity mismatch: expected "
                f"{expected_model_id}@{expected_model_version}, received "
                f"{model_id}@{model_version}"
            )
        if str(prediction["prompt_type"]) != "gt_bbox":
            raise ValueError(
                "Controlled pseudo references must be generated with GT bbox prompts"
            )
        if "predicted_mask_rle" not in prediction:
            raise ValueError("Prediction has no predicted_mask_rle")
        encoded = prediction["predicted_mask_rle"]
        counts = encoded["counts"]
        normalized = {
            "size": [int(value) for value in encoded["size"]],
            "counts": counts.encode("ascii") if isinstance(counts, str) else counts,
        }
        mask_pixels = int(mask_utils.area(normalized))
        is_empty = mask_pixels == 0
        status = str(prediction.get("status", ""))
        if status not in {"ok", "empty_mask"}:
            raise ValueError(f"Invalid pseudo-reference source status: {status}")
        if (status == "empty_mask") != is_empty:
            raise ValueError("Prediction status does not match encoded mask area")
        rows.append(
            {
                "instance_id": str(prediction["instance_id"]),
                "image_id": str(prediction["image_id"]),
                "source_scene_id": str(prediction["source_scene_id"]),
                "stratum": str(prediction["stratum"]),
                "mask_rle": prediction["predicted_mask_rle"],
                "reference_type": "pseudo_sam1",
                "teacher_model_id": model_id,
                "teacher_model_version": model_version,
                "teacher_prompt_type": str(prediction["prompt_type"]),
                "teacher_prompt_source": str(prediction["prompt_source"]),
                "teacher_run_id": str(prediction["run_id"]),
                "teacher_prediction_status": status,
                "reference_mask_pixels": mask_pixels,
                "reference_is_empty": is_empty,
            }
        )
    return rows
