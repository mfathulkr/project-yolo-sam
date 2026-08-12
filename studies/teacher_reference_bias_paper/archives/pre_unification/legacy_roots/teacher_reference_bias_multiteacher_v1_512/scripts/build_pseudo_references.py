from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias_multiteacher.io import (  # noqa: E402
    read_jsonl,
    portable_path,
    sha256_file,
    write_json,
    write_jsonl,
)
from teacher_reference_bias_multiteacher.paths import (  # noqa: E402
    DATASETS,
    prediction_path,
    reference_path,
)
from teacher_reference_bias_multiteacher.pseudo_reference import (  # noqa: E402
    build_pseudo_reference_rows,
)
from yolo_sam.runtime.manifest import validate_completed_run_manifest  # noqa: E402


def validate_source_manifest(
    predictions_path: Path,
    *,
    teacher: str,
) -> tuple[Path, dict[str, object]]:
    manifest_path = predictions_path.with_name("manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Kaynak prediction manifesti yok: {manifest_path}")
    manifest = validate_completed_run_manifest(manifest_path)
    if manifest.get("stage") != "gt_bbox_segmentation":
        raise ValueError(f"Beklenmeyen kaynak stage: {manifest.get('stage')}")

    actual_sha256 = sha256_file(predictions_path)
    declared = (
        manifest.get("output_file_fingerprints", {})
        .get("predictions", {})
        .get("sha256")
    )
    if declared != actual_sha256:
        raise ValueError(
            f"Kaynak prediction hash uyuşmuyor: {declared} != {actual_sha256}"
        )

    parameters = manifest.get("parameters", {})
    if parameters.get("model") != teacher:
        raise ValueError(f"Kaynak öğretmen manifesti uyuşmuyor: {teacher}")
    if parameters.get("prompt_type") != "gt_bbox":
        raise ValueError("Pseudo referans kaynağı GT bbox manifesti olmalıdır")
    model_config = parameters.get("model_config", {})
    if teacher == "sam3":
        if model_config.get("inference_interface") != "sam3_tracker_pvs":
            raise ValueError("SAM3 pseudo referansı yalnız tracker PVS kaynağından üretilebilir")
        if float(model_config.get("mask_threshold", 1.0)) != 0.0:
            raise ValueError("SAM3 PVS mask_threshold 0.0 olmalıdır")
        if int(model_config.get("box_batch_size", 0)) <= 0:
            raise ValueError("SAM3 PVS box_batch_size manifestte bulunmalıdır")
    config_hash = str(manifest.get("config_hash", ""))
    if len(config_hash) != 64:
        raise ValueError("Kaynak prediction config hash'i geçersiz")
    return manifest_path, manifest


def main() -> None:
    for dataset_id, source in DATASETS.items():
        for teacher in ("sam2", "sam3"):
            predictions_path = prediction_path(source, teacher, "gt_bbox")
            output_path = reference_path(dataset_id, teacher)
            source_manifest_path, source_manifest = validate_source_manifest(
                predictions_path,
                teacher=teacher,
            )
            predictions = read_jsonl(predictions_path)
            references = build_pseudo_reference_rows(
                predictions,
                teacher=teacher,
            )
            if len(references) != source.teacher_instance_count:
                raise ValueError(
                    f"{dataset_id}/{teacher}: {source.teacher_instance_count} "
                    f"yerine {len(references)} referans üretildi"
                )
            source_model_config = source_manifest["parameters"]["model_config"]
            source_manifest_sha256 = sha256_file(source_manifest_path)
            for reference in references:
                reference["teacher_prediction_config_hash"] = source_manifest[
                    "config_hash"
                ]
                reference["teacher_prediction_manifest_sha256"] = (
                    source_manifest_sha256
                )
                reference["teacher_inference_interface"] = source_model_config.get(
                    "inference_interface",
                    "bbox_prompted_image_segmentation",
                )
            write_jsonl(output_path, references)
            empty_reference_count = sum(
                bool(reference["reference_is_empty"])
                for reference in references
            )
            write_json(
                output_path.with_suffix(".manifest.json"),
                {
                    "schema_version": 2,
                    "status": "completed",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_id": dataset_id,
                    "reference_type": f"pseudo_{teacher}",
                    "construction": "frozen_gt_bbox_prediction_identity",
                    "instance_count": len(references),
                    "source_predictions": portable_path(predictions_path, REPO_ROOT),
                    "source_predictions_sha256": sha256_file(predictions_path),
                    "source_prediction_manifest": portable_path(
                        source_manifest_path,
                        REPO_ROOT,
                    ),
                    "source_prediction_manifest_sha256": source_manifest_sha256,
                    "source_prediction_config_hash": source_manifest["config_hash"],
                    "source_prediction_run_id": source_manifest["run_id"],
                    "output": portable_path(output_path, REPO_ROOT),
                    "output_sha256": sha256_file(output_path),
                    "teacher_model_id": references[0]["teacher_model_id"],
                    "teacher_model_version": references[0]["teacher_model_version"],
                    "teacher_inference_interface": references[0][
                        "teacher_inference_interface"
                    ],
                    "teacher_model_config": source_model_config,
                    "empty_reference_count": empty_reference_count,
                    "empty_reference_rate": empty_reference_count / len(references),
                    "known_positive_empty_reference_policy": "score_zero",
                    "scientific_warning": (
                        "Öğretmenin boş olmayan aynı GT-bbox tahmini kendi pseudo "
                        "referansına karşı 1.0 veren identity control'dür. Bilinen "
                        "instance için boş pseudo referans eksik etikettir ve 0 alır."
                    ),
                },
            )
            print(output_path)


if __name__ == "__main__":
    main()
