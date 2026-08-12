from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias.reporting.analysis import (
    aggregate_metrics,
    collect_canonical_metrics,
    collect_detector_metrics,
    collect_prediction_status_audit,
    collect_training_health,
    detector_seed_summary,
    paired_model_comparisons,
    ranking_comparisons,
    reference_inflation,
    segmentation_seed_summary,
    write_analysis_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile canonical matched-study metrics and statistical tables."
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=STUDY_ROOT / "results",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--detector-seed",
        action="append",
        type=int,
        dest="detector_seeds",
        help=(
            "Include only this detector seed in canonical YOLO-bbox analysis. "
            "Repeat for multiple seeds."
        ),
    )
    return parser.parse_args()


def write_csv(frame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def _path_seed(path: Path) -> int | None:
    for part in path.parts:
        match = re.match(r"seed_(\d+)(?:_|$)", part)
        if match is not None:
            return int(match.group(1))
    return None


def _selected_paths(paths: list[Path], seeds: set[int]) -> list[Path]:
    return [
        path
        for path in paths
        if (seed := _path_seed(path)) is None or seed in seeds
    ]


def main() -> None:
    args = parse_args()
    selected_seeds = set(args.detector_seeds or (42,))
    if not selected_seeds:
        raise ValueError("At least one detector seed must be selected")
    evaluation_root = args.study_root / "evaluation"
    output_root = args.study_root / "analysis"
    output_root.mkdir(parents=True, exist_ok=True)

    metrics, metric_files = collect_canonical_metrics(
        evaluation_root,
        detector_seeds=selected_seeds,
    )
    metrics = metrics[
        metrics["detector_seed"].isna()
        | metrics["detector_seed"].astype("Int64").isin(selected_seeds)
    ].copy()
    aggregates = aggregate_metrics(
        metrics,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    model_comparisons = paired_model_comparisons(
        metrics,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    inflation = reference_inflation(
        metrics,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )
    rankings = ranking_comparisons(metrics)
    detector_metrics, detector_files = collect_detector_metrics(
        args.study_root / "detectors",
        detector_seeds=selected_seeds,
    )
    detector_metrics = detector_metrics[
        detector_metrics["seed"].astype(int).isin(selected_seeds)
    ].copy()
    detector_files = _selected_paths(detector_files, selected_seeds)
    training_health, training_files = collect_training_health(
        args.study_root / "detectors",
        detector_seeds=selected_seeds,
    )
    training_health = training_health[
        training_health["seed"].astype(int).isin(selected_seeds)
    ].copy()
    training_files = _selected_paths(training_files, selected_seeds)
    detector_summary = detector_seed_summary(detector_metrics)
    segmentation_summary = segmentation_seed_summary(aggregates)
    prediction_status, prediction_files = collect_prediction_status_audit(
        args.study_root / "predictions",
        detector_seeds=selected_seeds,
    )
    prediction_status = prediction_status[
        prediction_status["detector_seed"].isna()
        | prediction_status["detector_seed"].astype("Int64").isin(selected_seeds)
    ].copy()
    prediction_files = _selected_paths(prediction_files, selected_seeds)
    shared_audit_root = output_root / "shared_human_reference_audit"
    shared_inputs = [
        path
        for path in (
            shared_audit_root / "manifest.json",
            shared_audit_root / "reference_quality_summary.csv",
            shared_audit_root / "model_dual_reference_summary.csv",
            shared_audit_root / "model_reference_inflation_ci.json",
            shared_audit_root / "ranking_comparison.json",
            shared_audit_root / "unique_human_object_sensitivity.csv",
        )
        if path.exists()
    ]

    outputs = [
        write_csv(metrics, output_root / "canonical_instance_metrics.csv"),
        write_csv(aggregates, output_root / "aggregate_metrics.csv"),
        write_csv(
            model_comparisons,
            output_root / "paired_model_comparisons.csv",
        ),
        write_csv(inflation, output_root / "reference_inflation.csv"),
        write_csv(rankings, output_root / "ranking_comparisons.csv"),
        write_csv(detector_metrics, output_root / "detector_metrics_by_seed.csv"),
        write_csv(
            training_health,
            output_root / "training_health_audit.csv",
        ),
        write_csv(detector_summary, output_root / "detector_seed_summary.csv"),
        write_csv(
            segmentation_summary,
            output_root / "segmentation_seed_summary.csv",
        ),
        write_csv(
            prediction_status,
            output_root / "prediction_status_audit.csv",
        ),
    ]
    write_analysis_manifest(
        output_root / "manifest.json",
        inputs=[
            *metric_files,
            *detector_files,
            *training_files,
            *prediction_files,
            *shared_inputs,
        ],
        outputs=outputs,
        parameters={
            "bootstrap_samples": args.bootstrap_samples,
            "confidence_level": args.confidence_level,
            "bootstrap_seed": args.bootstrap_seed,
            "detector_seeds": sorted(selected_seeds),
        },
    )
    print(output_root)


if __name__ == "__main__":
    main()
