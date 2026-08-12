from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pycocotools.coco import COCO


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias_multiteacher.io import (  # noqa: E402
    portable_path,
    read_jsonl,
    sha256_file,
    write_json,
)
from teacher_reference_bias_multiteacher.paths import (  # noqa: E402
    BBOX_SOURCES,
    DATASETS,
    MODELS,
    prediction_path,
    reference_path,
)
from teacher_reference_bias_multiteacher.rle_metrics import (  # noqa: E402
    binary_metrics_from_rle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dört deney için model × prompt × referans metrik küplerini üret."
    )
    parser.add_argument("--experiment", choices=tuple(DATASETS))
    return parser.parse_args()


def native_references(source) -> dict[str, dict[str, Any]]:
    coco = COCO(str(source.coco_path))
    rows: dict[str, dict[str, Any]] = {}
    for annotation_id in coco.getAnnIds():
        annotation = coco.loadAnns([annotation_id])[0]
        instance_id = (
            f"{source.dataset_id}:{int(annotation['image_id'])}:"
            f"{int(annotation['id'])}"
        )
        rows[instance_id] = {
            "mask_rle": coco.annToRLE(annotation),
            "reference_type": source.reference_types[0],
        }
    if len(rows) != source.instance_count:
        raise ValueError(
            f"{source.experiment_id}: {source.instance_count} yerine "
            f"{len(rows)} yerel referans var"
        )
    return rows


def load_references(source) -> tuple[dict[str, dict[str, dict[str, Any]]], list[Path]]:
    references = {source.reference_types[0]: native_references(source)}
    inputs = [source.coco_path]
    for reference_type in source.reference_types[1:]:
        path = reference_path(source, reference_type)
        rows = read_jsonl(path)
        by_instance = {str(row["instance_id"]): row for row in rows}
        if len(by_instance) != source.instance_count:
            raise ValueError(f"{source.experiment_id}/{reference_type}: eksik referans")
        references[reference_type] = by_instance
        inputs.append(path)
        manifest_path = path.with_suffix(".manifest.json")
        if manifest_path.is_file():
            inputs.append(manifest_path)
    return references, inputs


def evaluate_experiment(experiment_id: str) -> Path:
    source = DATASETS[experiment_id]
    references, input_paths = load_references(source)
    native_ids = set(references[source.reference_types[0]])
    metric_frames: list[pd.DataFrame] = []
    for model in MODELS:
        for bbox_source in BBOX_SOURCES:
            path = prediction_path(source, model, bbox_source)
            predictions = read_jsonl(path)
            prediction_ids = [str(row["instance_id"]) for row in predictions]
            if len(prediction_ids) != len(set(prediction_ids)):
                raise ValueError(f"{experiment_id}/{model}/{bbox_source}: yinelenen ID")
            if set(prediction_ids) != native_ids:
                raise ValueError(
                    f"{experiment_id}/{model}/{bbox_source}: tahmin ve referans "
                    "instance kümeleri farklı"
                )
            rows: list[dict[str, object]] = []
            for prediction in predictions:
                instance_id = str(prediction["instance_id"])
                for reference_type in source.reference_types:
                    reference = references[reference_type][instance_id]
                    metrics = binary_metrics_from_rle(
                        prediction["predicted_mask_rle"],
                        reference["mask_rle"],
                        known_positive_instance=True,
                    )
                    rows.append(
                        {
                            "dataset_id": source.dataset_id,
                            "experiment_id": experiment_id,
                            "model": model,
                            "bbox_source": bbox_source,
                            "detector_seed": 42 if bbox_source == "yolo_bbox" else pd.NA,
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
                            "boundary_iou": np.nan,
                        }
                    )
            metric_frames.append(pd.DataFrame(rows))
            input_paths.append(path)
            manifest_path = path.with_name("manifest.json")
            if manifest_path.is_file():
                input_paths.append(manifest_path)

    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics["detector_seed"] = metrics["detector_seed"].astype("Int64")
    expected = source.instance_count * len(MODELS) * len(BBOX_SOURCES) * len(
        source.reference_types
    )
    if len(metrics) != expected:
        raise ValueError(f"{experiment_id}: {expected} yerine {len(metrics)} satır")
    output_path = source.analysis_root / "canonical_instance_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)
    manifest_path = source.analysis_root / "metric_cube_manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 3,
            "status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "dataset_id": source.dataset_id,
            "models": list(MODELS),
            "bbox_sources": list(BBOX_SOURCES),
            "reference_types": list(source.reference_types),
            "instance_count": source.instance_count,
            "metric_row_count": len(metrics),
            "metric_definition": "instance_equal_weight_rle_exact",
            "known_positive_empty_reference_policy": "score_zero",
            "boundary_iou": "not_evaluated_not_reported",
            "inputs": {
                portable_path(path, REPO_ROOT): sha256_file(path)
                for path in sorted(set(input_paths))
            },
            "outputs": {
                portable_path(output_path, REPO_ROOT): sha256_file(output_path)
            },
        },
    )
    return output_path


def main() -> None:
    args = parse_args()
    experiments = (args.experiment,) if args.experiment else tuple(DATASETS)
    for experiment_id in experiments:
        print(evaluate_experiment(experiment_id))


if __name__ == "__main__":
    main()
