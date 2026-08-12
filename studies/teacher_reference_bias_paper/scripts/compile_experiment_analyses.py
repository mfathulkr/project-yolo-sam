from __future__ import annotations

import argparse
import json
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
from teacher_reference_bias_multiteacher.paths import DATASETS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deney bazında özet analizleri üret.")
    parser.add_argument("--experiment", choices=tuple(DATASETS))
    return parser.parse_args()


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def compile_one(experiment_id: str) -> Path:
    source = DATASETS[experiment_id]
    input_path = source.analysis_root / "canonical_instance_metrics.csv"
    metrics = pd.read_csv(input_path)
    metrics["detector_seed"] = metrics["detector_seed"].astype("Int64")
    validate_metric_cube(metrics)
    baseline = source.reference_types[0]
    aggregates = aggregate_metrics(metrics)
    effects = paired_reference_effects(metrics, baseline_reference=baseline)
    rankings = ranking_table(
        aggregates,
        reference_order=source.reference_types,
        baseline_reference=baseline,
    )
    advantages = teacher_advantage_table(aggregates)
    agreements = reference_agreement_table(
        aggregates,
        reference_order=source.reference_types,
    )
    empty_stats = (
        metrics[
            (metrics["model"] == "sam1")
            & (metrics["bbox_source"] == "gt_bbox")
        ]
        .groupby("reference_type")["reference_is_empty"]
        .agg(["sum", "mean"])
        .reset_index()
        .rename(columns={"sum": "empty_count", "mean": "empty_rate"})
    )
    detector_metrics_path = (
        source.detector_root / "seed_42" / "evaluation" / "test" / "metrics.json"
    )
    detector_metrics = json.loads(detector_metrics_path.read_text(encoding="utf-8"))
    detector_metric_names = (
        "fixed_confidence_threshold",
        "bbox_AP50",
        "bbox_AP75",
        "bbox_AP90",
        "bbox_AP50_95",
        "precision_at_bbox_iou50",
        "recall_at_bbox_iou50",
        "precision_at_bbox_iou75",
        "recall_at_bbox_iou75",
        "precision_at_bbox_iou90",
        "recall_at_bbox_iou90",
    )
    detector_summary = pd.DataFrame(
        [
            {
                "dataset_id": source.dataset_id,
                "seed_count": 1,
                "seed_ids": "42",
                **{
                    f"{metric}_mean": float(detector_metrics[metric])
                    for metric in detector_metric_names
                },
                **{f"{metric}_std": pd.NA for metric in detector_metric_names},
            }
        ]
    )
    outputs = [
        write_csv(aggregates, source.analysis_root / "aggregate_metrics.csv"),
        write_csv(effects, source.analysis_root / "paired_reference_effects.csv"),
        write_csv(rankings, source.analysis_root / "ranking_by_reference.csv"),
        write_csv(advantages, source.analysis_root / "teacher_advantage.csv"),
        write_csv(agreements, source.analysis_root / "reference_agreement.csv"),
        write_csv(empty_stats, source.analysis_root / "reference_empty_stats.csv"),
        write_csv(detector_summary, source.analysis_root / "detector_summary.csv"),
    ]
    manifest_path = source.analysis_root / "manifest.json"
    write_json(
        manifest_path,
        {
            "schema_version": 3,
            "status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "dataset_id": source.dataset_id,
            "baseline_reference": baseline,
            "reference_types": list(source.reference_types),
            "bootstrap_seed": 42,
            "bootstrap_samples": 10_000,
            "confidence_level": 0.95,
            "inputs": {
                portable_path(input_path, REPO_ROOT): sha256_file(input_path),
                portable_path(detector_metrics_path, REPO_ROOT): sha256_file(
                    detector_metrics_path
                ),
            },
            "outputs": {
                portable_path(path, REPO_ROOT): sha256_file(path) for path in outputs
            },
        },
    )
    return manifest_path


def main() -> None:
    args = parse_args()
    experiments = (args.experiment,) if args.experiment else tuple(DATASETS)
    for experiment_id in experiments:
        print(compile_one(experiment_id))


if __name__ == "__main__":
    main()
