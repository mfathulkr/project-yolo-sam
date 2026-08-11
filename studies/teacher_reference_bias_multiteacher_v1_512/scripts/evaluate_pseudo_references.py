from __future__ import annotations

import argparse
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
from teacher_reference_bias_multiteacher.rle_metrics import (  # noqa: E402
    binary_metrics_from_rle,
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


def evaluate_condition(dataset_id: str, model: str, bbox_source: str) -> Path:
    source = DATASETS[dataset_id]
    predictions_path = prediction_path(source, model, bbox_source)
    output_path = evaluation_path(dataset_id, model, bbox_source)
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
    for teacher in ("sam2", "sam3"):
        reference_type = f"pseudo_{teacher}"
        path = reference_path(dataset_id, teacher)
        rows = read_jsonl(path)
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
            "schema_version": 1,
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
            "boundary_iou": "not_evaluated_not_reported",
            "inputs": {
                portable_path(predictions_path, REPO_ROOT): sha256_file(predictions_path),
                **{
                    portable_path(path, REPO_ROOT): sha256_file(path)
                    for path in reference_files.values()
                },
            },
            "output": portable_path(output_path, REPO_ROOT),
            "output_sha256": sha256_file(output_path),
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
