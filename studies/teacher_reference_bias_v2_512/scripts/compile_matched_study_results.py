from __future__ import annotations

import argparse
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
    return parser.parse_args()


def write_csv(frame, path: Path) -> Path:
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    evaluation_root = args.study_root / "evaluation"
    output_root = args.study_root / "analysis"
    output_root.mkdir(parents=True, exist_ok=True)

    metrics, metric_files = collect_canonical_metrics(evaluation_root)
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
        args.study_root / "detectors"
    )
    training_health, training_files = collect_training_health(
        args.study_root / "detectors"
    )
    detector_summary = detector_seed_summary(detector_metrics)
    segmentation_summary = segmentation_seed_summary(aggregates)
    prediction_status, prediction_files = collect_prediction_status_audit(
        args.study_root / "predictions"
    )
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
        },
    )
    print(output_root)


if __name__ == "__main__":
    main()
