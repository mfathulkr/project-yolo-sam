from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


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
)
from teacher_reference_bias_multiteacher.paths import (  # noqa: E402
    BBOX_SOURCES,
    DATASETS,
    MODELS,
    evaluation_path,
    prediction_path,
    reference_path,
)
from teacher_reference_bias_multiteacher.pseudo_reference import (  # noqa: E402
    validate_pseudo_reference_identity,
)
from teacher_reference_bias_multiteacher.rle_metrics import (  # noqa: E402
    binary_metrics_from_rle,
)
from yolo_sam.runtime.manifest import (  # noqa: E402
    validate_completed_run_manifest,
    validate_completed_run_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dondurulmuş SAM tahminlerini SAM2 ve SAM3 referanslarıyla ölç."
    )
    parser.add_argument("--dataset", choices=tuple(DATASETS))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--bbox-source", choices=BBOX_SOURCES)
    return parser.parse_args()


def selected(values: tuple[str, ...], value: str | None) -> tuple[str, ...]:
    return (value,) if value is not None else values


def validate_reference_artifact(
    path: Path,
    *,
    teacher: str,
) -> tuple[Path, list[dict[str, object]], list[Path]]:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"Pseudo reference is not completed: {manifest_path}")
    if manifest.get("output_sha256") != sha256_file(path):
        raise ValueError(f"Pseudo reference hash mismatch: {path}")
    if manifest.get("reference_type") != f"pseudo_{teacher}":
        raise ValueError(f"Pseudo reference teacher mismatch: {path}")
    if teacher == "sam3" and manifest.get("teacher_inference_interface") != "sam3_tracker_pvs":
        raise ValueError("SAM3 pseudo reference is not derived from tracker PVS")
    if manifest.get("known_positive_empty_reference_policy") != "score_zero":
        raise ValueError("Pseudo reference empty-mask policy is missing or invalid")

    source_predictions = REPO_ROOT / str(manifest.get("source_predictions", ""))
    source_manifest_path = REPO_ROOT / str(
        manifest.get("source_prediction_manifest", "")
    )
    if manifest.get("source_predictions_sha256") != sha256_file(source_predictions):
        raise ValueError(f"Pseudo reference source prediction hash mismatch: {path}")
    if manifest.get("source_prediction_manifest_sha256") != sha256_file(
        source_manifest_path
    ):
        raise ValueError(f"Pseudo reference source manifest hash mismatch: {path}")
    source_manifest = validate_completed_run_manifest(source_manifest_path)
    if source_manifest.get("stage") != "gt_bbox_segmentation":
        raise ValueError(f"Pseudo reference source is not GT bbox: {source_manifest_path}")
    if source_manifest.get("config_hash") != manifest.get(
        "source_prediction_config_hash"
    ):
        raise ValueError(f"Pseudo reference config lineage mismatch: {path}")
    if source_manifest.get("run_id") != manifest.get("source_prediction_run_id"):
        raise ValueError(f"Pseudo reference run lineage mismatch: {path}")
    source_parameters = source_manifest.get("parameters", {})
    if source_parameters.get("model") != teacher:
        raise ValueError(f"Pseudo reference teacher lineage mismatch: {path}")
    if source_parameters.get("prompt_type") != "gt_bbox":
        raise ValueError(f"Pseudo reference source prompt mismatch: {path}")

    references = read_jsonl(path)
    source_rows = read_jsonl(source_predictions)
    validate_pseudo_reference_identity(
        source_rows,
        references,
        teacher=teacher,
    )
    empty_count = sum(bool(row["reference_is_empty"]) for row in references)
    if int(manifest.get("instance_count", -1)) != len(references):
        raise ValueError(f"Pseudo reference instance count mismatch: {path}")
    if int(manifest.get("empty_reference_count", -1)) != empty_count:
        raise ValueError(f"Pseudo reference empty count mismatch: {path}")
    return manifest_path, references, [source_predictions, source_manifest_path]


def validate_prediction_artifact(
    path: Path,
    *,
    model: str,
    bbox_source: str,
) -> tuple[Path, dict[str, object]]:
    manifest_path = path.parent / "manifest.json"
    manifest = validate_completed_run_output(
        manifest_path,
        output_name="predictions",
        output_path=path,
    )
    parameters = manifest.get("parameters", {})
    expected_stage = (
        "gt_bbox_segmentation"
        if bbox_source == "gt_bbox"
        else "yolo_bbox_segmentation"
    )
    if manifest.get("stage") != expected_stage:
        raise ValueError(f"Unexpected prediction stage: {manifest_path}")
    if parameters.get("model") != model:
        raise ValueError(f"Prediction model lineage mismatch: {manifest_path}")

    if bbox_source == "gt_bbox":
        if parameters.get("prompt_type") != "gt_bbox":
            raise ValueError(f"GT-bbox prediction prompt mismatch: {manifest_path}")
        model_config = parameters.get("model_config", {})
    else:
        if int(parameters.get("seed", -1)) != 42:
            raise ValueError(
                f"YOLO-bbox prediction is not canonical seed 42: {manifest_path}"
            )
        model_config = parameters.get("segmenter", {})

    if model == "sam3":
        if model_config.get("inference_interface") != "sam3_tracker_pvs":
            raise ValueError(f"SAM3 prediction is not tracker PVS: {manifest_path}")
        if float(model_config.get("mask_threshold", 1.0)) != 0.0:
            raise ValueError(
                f"SAM3 prediction has unsafe mask threshold: {manifest_path}"
            )
        if int(model_config.get("box_batch_size", 0)) <= 0:
            raise ValueError(f"SAM3 prediction has no box batch size: {manifest_path}")
    return manifest_path, manifest


def evaluate_condition(dataset_id: str, model: str, bbox_source: str) -> Path:
    source = DATASETS[dataset_id]
    predictions_path = prediction_path(source, model, bbox_source)
    output_path = evaluation_path(dataset_id, model, bbox_source)
    prediction_manifest_path, prediction_manifest = validate_prediction_artifact(
        predictions_path,
        model=model,
        bbox_source=bbox_source,
    )
    predictions = read_jsonl(predictions_path)
    if len(predictions) != source.teacher_instance_count:
        raise ValueError(
            f"{dataset_id}/{model}/{bbox_source}: beklenmeyen tahmin sayısı "
            f"{len(predictions)}"
        )
    prediction_ids = [str(row["instance_id"]) for row in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("Tahminlerde yinelenen instance ID var")

    references: dict[str, dict[str, dict]] = {}
    reference_files: dict[str, Path] = {}
    reference_manifests: dict[str, Path] = {}
    reference_lineage_files: list[Path] = []
    for teacher in ("sam2", "sam3"):
        reference_type = f"pseudo_{teacher}"
        path = reference_path(dataset_id, teacher)
        (
            reference_manifests[reference_type],
            rows,
            lineage_files,
        ) = validate_reference_artifact(
            path, teacher=teacher
        )
        reference_lineage_files.extend(lineage_files)
        by_instance = {str(row["instance_id"]): row for row in rows}
        if len(by_instance) != len(rows):
            raise ValueError(f"{reference_type} içinde yinelenen instance ID var")
        if set(by_instance) != set(prediction_ids):
            raise ValueError(
                f"{dataset_id}/{model}/{bbox_source}/{reference_type}: "
                "tahmin ve referans instance kümeleri farklı"
            )
        references[reference_type] = by_instance
        reference_files[reference_type] = path

    metric_rows: list[dict[str, object]] = []
    for prediction in predictions:
        instance_id = str(prediction["instance_id"])
        for reference_type, by_instance in references.items():
            reference = by_instance[instance_id]
            metrics = binary_metrics_from_rle(
                prediction["predicted_mask_rle"],
                reference["mask_rle"],
                known_positive_instance=True,
            )
            metric_rows.append(
                {
                    "run_id": str(prediction["run_id"]),
                    "model_id": str(prediction["model_id"]),
                    "model_version": str(prediction["model_version"]),
                    "prompt_type": str(prediction["prompt_type"]),
                    "image_id": str(prediction["image_id"]),
                    "instance_id": instance_id,
                    "source_scene_id": str(prediction["source_scene_id"]),
                    "reference_type": reference_type,
                    "stratum": str(prediction["stratum"]),
                    "prediction_is_empty": int(
                        metrics["true_positive_pixels"]
                        + metrics["false_positive_pixels"]
                    )
                    == 0,
                    "reference_is_empty": int(
                        metrics["true_positive_pixels"]
                        + metrics["false_negative_pixels"]
                    )
                    == 0,
                    **metrics,
                    # Full-metric belgeleri BIoU kullanmıyor. Eski şemayla
                    # birleşebilmek için sütun korunuyor, değer açıkça NA'dır.
                    "boundary_iou": np.nan,
                }
            )

    frame = pd.DataFrame(metric_rows)
    expected_rows = source.teacher_instance_count * 2
    if len(frame) != expected_rows:
        raise ValueError(f"{expected_rows} yerine {len(frame)} metrik satırı üretildi")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    write_json(
        output_path.with_name("manifest.json"),
        {
            "schema_version": 3,
            "status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dataset_id": dataset_id,
            "model": model,
            "bbox_source": bbox_source,
            "detector_seed": 42 if bbox_source == "yolo_bbox" else None,
            "instance_count": source.teacher_instance_count,
            "metric_row_count": len(frame),
            "reference_types": ["pseudo_sam2", "pseudo_sam3"],
            "metric_definition": "instance_equal_weight_rle_exact",
            "metric_schema_version": 2,
            "primary_granularity": "instance",
            "instance_weighting": "equal",
            "known_positive_empty_reference_policy": "score_zero",
            "boundary_iou": "not_evaluated_not_reported",
            "upstream_prediction_run_id": prediction_manifest.get("run_id"),
            "inputs": {
                portable_path(predictions_path, REPO_ROOT): sha256_file(predictions_path),
                portable_path(prediction_manifest_path, REPO_ROOT): sha256_file(
                    prediction_manifest_path
                ),
                **{
                    portable_path(path, REPO_ROOT): sha256_file(path)
                    for path in reference_files.values()
                },
                **{
                    portable_path(path, REPO_ROOT): sha256_file(path)
                    for path in reference_manifests.values()
                },
                **{
                    portable_path(path, REPO_ROOT): sha256_file(path)
                    for path in reference_lineage_files
                },
            },
            "output": portable_path(output_path, REPO_ROOT),
            "output_sha256": sha256_file(output_path),
            "outputs": {
                portable_path(output_path, REPO_ROOT): sha256_file(output_path)
            },
        },
    )
    print(output_path)
    return output_path


def main() -> None:
    args = parse_args()
    for dataset_id in selected(tuple(DATASETS), args.dataset):
        for model in selected(MODELS, args.model):
            for bbox_source in selected(BBOX_SOURCES, args.bbox_source):
                evaluate_condition(dataset_id, model, bbox_source)


if __name__ == "__main__":
    main()
