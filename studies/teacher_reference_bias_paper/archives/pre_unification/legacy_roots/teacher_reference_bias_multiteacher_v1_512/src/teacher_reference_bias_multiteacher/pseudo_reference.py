from __future__ import annotations

from typing import Any

from pycocotools import mask as mask_utils

from .rle_metrics import normalized_rle


EXPECTED_MODEL_IDS = {
    "sam1": "facebook/sam-vit-huge",
    "sam2": "facebook/sam2.1-hiera-large",
    "sam3": "sam3-local",
}


def build_pseudo_reference_rows(
    predictions: list[dict[str, Any]],
    *,
    teacher: str,
) -> list[dict[str, Any]]:
    if teacher not in EXPECTED_MODEL_IDS:
        raise ValueError(f"Desteklenmeyen öğretmen: {teacher}")
    if not predictions:
        raise ValueError("Pseudo referans kaynağı boş")

    instance_ids = [str(row["instance_id"]) for row in predictions]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("Pseudo referans kaynağında yinelenen instance ID var")

    expected_model_id = EXPECTED_MODEL_IDS[teacher]
    versions = {str(row["model_version"]) for row in predictions}
    if len(versions) != 1:
        raise ValueError(f"{teacher} kaynağında birden fazla model sürümü var")
    model_version = next(iter(versions))

    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        if str(prediction["model_id"]) != expected_model_id:
            raise ValueError(
                f"Öğretmen kimliği uyuşmuyor: {prediction['model_id']} != "
                f"{expected_model_id}"
            )
        if str(prediction["prompt_type"]) != "gt_bbox":
            raise ValueError("Pseudo referans yalnız GT bbox isteminden üretilebilir")
        if str(prediction["status"]) not in {"ok", "empty_mask"}:
            raise ValueError(
                "Pseudo referans kaynağında geçersiz tahmin durumu var: "
                f"{prediction['status']}"
            )
        if "predicted_mask_rle" not in prediction:
            raise ValueError("Pseudo referans kaynağında maske yok")
        mask_pixels = int(
            mask_utils.area(normalized_rle(prediction["predicted_mask_rle"]))
        )
        is_empty = mask_pixels == 0
        if (str(prediction["status"]) == "empty_mask") != is_empty:
            raise ValueError(
                f"{prediction['instance_id']}: status ile gerçek maske alanı uyuşmuyor"
            )
        rows.append(
            {
                "instance_id": str(prediction["instance_id"]),
                "image_id": str(prediction["image_id"]),
                "source_scene_id": str(prediction["source_scene_id"]),
                "stratum": str(prediction["stratum"]),
                "mask_rle": prediction["predicted_mask_rle"],
                "reference_type": f"pseudo_{teacher}",
                "teacher_model": teacher,
                "teacher_model_id": expected_model_id,
                "teacher_model_version": model_version,
                "teacher_prompt_type": "gt_bbox",
                "teacher_prompt_source": str(prediction["prompt_source"]),
                "teacher_run_id": str(prediction["run_id"]),
                "teacher_prediction_status": str(prediction["status"]),
                "reference_mask_pixels": mask_pixels,
                "reference_is_empty": is_empty,
            }
        )
    return rows


def validate_pseudo_reference_identity(
    predictions: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    teacher: str,
) -> None:
    """Prove that a pseudo-reference is an exact frozen teacher prediction."""
    expected_rows = build_pseudo_reference_rows(predictions, teacher=teacher)
    expected_by_instance = {
        str(row["instance_id"]): row for row in expected_rows
    }
    actual_by_instance = {
        str(row["instance_id"]): row for row in references
    }
    if len(actual_by_instance) != len(references):
        raise ValueError("Pseudo referansta yinelenen instance ID var")
    if set(actual_by_instance) != set(expected_by_instance):
        raise ValueError("Pseudo referans ve kaynak tahmin instance kümeleri farklı")

    exact_fields = (
        "image_id",
        "source_scene_id",
        "stratum",
        "reference_type",
        "teacher_model",
        "teacher_model_id",
        "teacher_model_version",
        "teacher_prompt_type",
        "teacher_prompt_source",
        "teacher_run_id",
        "teacher_prediction_status",
        "reference_mask_pixels",
        "reference_is_empty",
    )
    mismatches: list[str] = []
    for instance_id, expected in expected_by_instance.items():
        actual = actual_by_instance[instance_id]
        expected_rle = normalized_rle(expected["mask_rle"])
        actual_rle = normalized_rle(actual["mask_rle"])
        same_rle = (
            tuple(expected_rle["size"]) == tuple(actual_rle["size"])
            and expected_rle["counts"] == actual_rle["counts"]
        )
        if not same_rle or any(
            actual.get(field) != expected[field] for field in exact_fields
        ):
            mismatches.append(instance_id)
    if mismatches:
        raise ValueError(
            f"{len(mismatches)} pseudo referans kaynak tahminle birebir aynı değil; "
            f"örnekler={mismatches[:5]}"
        )
