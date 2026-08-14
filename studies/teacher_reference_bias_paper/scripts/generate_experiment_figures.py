from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias_multiteacher.paper_figures import (  # noqa: E402
    model_reference_matrix_figure,
    qualitative_figure,
    qualitative_selection_records,
    reference_effect_figure,
    write_figure_manifest,
)
from teacher_reference_bias_multiteacher.paths import (  # noqa: E402
    DATASETS,
    MODELS,
    prediction_path,
    reference_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deney figürlerini üret.")
    parser.add_argument("--experiment", choices=tuple(DATASETS))
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Mevcut figürleri yeniden çizmeden yalnız portable manifesti yenile.",
    )
    return parser.parse_args()


def generate_one(experiment_id: str, *, manifest_only: bool = False) -> Path:
    source = DATASETS[experiment_id]
    metrics_path = source.analysis_root / "canonical_instance_metrics.csv"
    aggregates_path = source.analysis_root / "aggregate_metrics.csv"
    effects_path = source.analysis_root / "paired_reference_effects.csv"
    metrics = pd.read_csv(metrics_path)
    aggregates = pd.read_csv(aggregates_path)
    effects = pd.read_csv(effects_path)
    qualitative_outputs = [
        source.figures_root / f"{reference_type}_gt_bbox_qualitative.png"
        for reference_type in source.reference_types
    ]
    matrix_output = source.figures_root / "model_reference_iou_matrix.png"
    effect_output = source.figures_root / "reference_effect_with_ci.png"
    outputs = [*qualitative_outputs, matrix_output, effect_output]
    if not manifest_only:
        for reference_type, output_path in zip(
            source.reference_types, qualitative_outputs, strict=True
        ):
            qualitative_figure(
                source=source,
                reference_type=reference_type,
                metrics=metrics,
                output_path=output_path,
            )
        model_reference_matrix_figure(source, aggregates, matrix_output)
        reference_effect_figure(source, effects, effect_output)
    missing = [path for path in outputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Eksik deney figürleri: {missing}")
    inputs = [metrics_path, aggregates_path, effects_path, source.coco_path]
    for model in MODELS:
        inputs.append(prediction_path(source, model, "gt_bbox"))
    for reference_type in source.reference_types[1:]:
        inputs.append(reference_path(source, reference_type))
    return write_figure_manifest(
        source,
        outputs,
        inputs,
        qualitative_selection_records(source),
    )


def main() -> None:
    args = parse_args()
    experiments = (args.experiment,) if args.experiment else tuple(DATASETS)
    for experiment_id in experiments:
        print(generate_one(experiment_id, manifest_only=args.manifest_only))


if __name__ == "__main__":
    main()
