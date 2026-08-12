from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias_multiteacher.figures import (  # noqa: E402
    model_reference_matrix_figure,
    qualitative_figure,
    reference_effect_figure,
    write_figure_manifest,
)
from teacher_reference_bias_multiteacher.paths import (  # noqa: E402
    DATASETS,
    MODELS,
    prediction_path,
    reference_path,
)


def main() -> None:
    analysis_root = STUDY_ROOT / "results" / "analysis"
    output_root = STUDY_ROOT / "results" / "figures"
    input_paths = [
        analysis_root / "manifest.json",
        analysis_root / "canonical_instance_metrics.csv",
        analysis_root / "aggregate_metrics.csv",
        analysis_root / "paired_reference_effects.csv",
    ]
    metrics = pd.read_csv(input_paths[1])
    aggregates = pd.read_csv(input_paths[2])
    effects = pd.read_csv(input_paths[3])
    outputs: list[Path] = []
    for dataset_id in DATASETS:
        source = DATASETS[dataset_id]
        input_paths.extend(
            (
                source.prepared_root / "content_manifest.json",
                source.coco_path,
            )
        )
        for model in MODELS:
            path = prediction_path(source, model, "gt_bbox")
            input_paths.extend((path, path.with_name("manifest.json")))
        for teacher in ("sam2", "sam3"):
            path = reference_path(dataset_id, teacher)
            input_paths.extend((path, path.with_suffix(".manifest.json")))
            outputs.append(
                qualitative_figure(
                    dataset_id=dataset_id,
                    reference_type=f"pseudo_{teacher}",
                    metrics=metrics,
                    output_path=(
                        output_root
                        / f"{dataset_id}_pseudo_{teacher}_gt_bbox_qualitative.png"
                    ),
                )
            )
    outputs.extend(
        (
            model_reference_matrix_figure(
                aggregates,
                output_root / "model_reference_iou_matrix.png",
            ),
            reference_effect_figure(
                effects,
                output_root / "reference_effect_with_ci.png",
            ),
        )
    )
    output_root.mkdir(parents=True, exist_ok=True)
    write_figure_manifest(
        outputs,
        output_root / "manifest.json",
        input_paths=input_paths,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
