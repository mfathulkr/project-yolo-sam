from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias_multiteacher.analysis import (  # noqa: E402
    aggregate_metrics,
    paired_reference_effects,
    ranking_table,
    reference_agreement_table,
    teacher_advantage_table,
    validate_metric_cube,
)
from teacher_reference_bias_multiteacher.io import (  # noqa: E402
    portable_path,
    sha256_file,
    write_json,
)
from teacher_reference_bias_multiteacher.paths import (  # noqa: E402
    BBOX_SOURCES,
    DATASETS,
    MODELS,
    evaluation_path,
)


OUTPUT_ROOT = STUDY_ROOT / "results" / "analysis"


def load_existing_metrics() -> tuple[list[pd.DataFrame], list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for dataset_id, source in DATASETS.items():
        path = source.canonical_analysis_root / "canonical_instance_metrics.csv"
        frame = pd.read_csv(path)
        frame = frame[
            (frame["dataset_id"] == dataset_id)
            & (frame["reference_type"].isin(("human", "pseudo_sam1")))
            & (
                frame["detector_seed"].isna()
                | (frame["detector_seed"].astype("Int64") == 42)
            )
        ].copy()
        frames.append(frame)
        paths.append(path)
    return frames, paths


def load_new_metrics() -> tuple[list[pd.DataFrame], list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for dataset_id in DATASETS:
        for model in MODELS:
            for bbox_source in BBOX_SOURCES:
                path = evaluation_path(dataset_id, model, bbox_source)
                frame = pd.read_csv(path)
                frame.insert(0, "dataset_id", dataset_id)
                frame.insert(1, "model", model)
                frame.insert(2, "bbox_source", bbox_source)
                frame.insert(3, "detector_seed", 42 if bbox_source == "yolo_bbox" else pd.NA)
                frame["metric_source"] = str(path)
                frames.append(frame)
                paths.append(path)
    return frames, paths


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    existing_frames, existing_paths = load_existing_metrics()
    new_frames, new_paths = load_new_metrics()
    metrics = pd.concat([*existing_frames, *new_frames], ignore_index=True)
    metrics["detector_seed"] = metrics["detector_seed"].astype("Int64")
    validate_metric_cube(metrics)

    aggregates = aggregate_metrics(metrics)
    effects = paired_reference_effects(metrics)
    rankings = ranking_table(aggregates)
    teacher_advantage = teacher_advantage_table(aggregates)
    agreement = reference_agreement_table(aggregates)

    outputs = [
        write_csv(metrics, OUTPUT_ROOT / "canonical_instance_metrics.csv"),
        write_csv(aggregates, OUTPUT_ROOT / "aggregate_metrics.csv"),
        write_csv(effects, OUTPUT_ROOT / "paired_reference_effects.csv"),
        write_csv(rankings, OUTPUT_ROOT / "ranking_by_reference.csv"),
        write_csv(teacher_advantage, OUTPUT_ROOT / "teacher_advantage.csv"),
        write_csv(agreement, OUTPUT_ROOT / "reference_agreement.csv"),
    ]
    write_json(
        OUTPUT_ROOT / "manifest.json",
        {
            "schema_version": 1,
            "status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "detector_seed": 42,
            "bootstrap_seed": 42,
            "bootstrap_samples": 10_000,
            "confidence_level": 0.95,
            "datasets": list(DATASETS),
            "models": list(MODELS),
            "reference_types": [
                "human",
                "pseudo_sam1",
                "pseudo_sam2",
                "pseudo_sam3",
            ],
            "inputs": {
                portable_path(path, REPO_ROOT): sha256_file(path)
                for path in [*existing_paths, *new_paths]
            },
            "outputs": {
                portable_path(path, REPO_ROOT): sha256_file(path) for path in outputs
            },
            "boundary_iou": (
                "SAM2/SAM3 extension rows do not evaluate BIoU; full-metric and "
                "paper tables use IoU, Dice, Precision, Recall and IoU success rates."
            ),
        },
    )
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
