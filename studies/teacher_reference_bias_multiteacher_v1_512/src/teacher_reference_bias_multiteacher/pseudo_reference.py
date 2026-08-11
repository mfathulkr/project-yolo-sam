from __future__ import annotations

from typing import Any


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
            }
        )
    return rows
