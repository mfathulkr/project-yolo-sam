from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias_multiteacher.comparison_report import (  # noqa: E402
    write_comparison_report,
)
from teacher_reference_bias_multiteacher.io import (  # noqa: E402
    portable_path,
    sha256_file,
    write_json,
)
from teacher_reference_bias_multiteacher.paths import (  # noqa: E402
    DATASETS,
    MODELS,
    prediction_path,
)


def main() -> None:
    analysis_root = STUDY_ROOT / "results" / "analysis"
    output_dir = STUDY_ROOT / "reports" / "teacher_comparison"
    matrix_figure = STUDY_ROOT / "results" / "figures" / "model_reference_iou_matrix.png"
    effect_figure = STUDY_ROOT / "results" / "figures" / "reference_effect_with_ci.png"
    outputs = write_comparison_report(
        output_dir=output_dir,
        aggregates=pd.read_csv(analysis_root / "aggregate_metrics.csv"),
        agreement=pd.read_csv(analysis_root / "reference_agreement.csv"),
        rankings=pd.read_csv(analysis_root / "ranking_by_reference.csv"),
        matrix_figure=matrix_figure,
        effect_figure=effect_figure,
    )
    inputs = [
        analysis_root / "manifest.json",
        analysis_root / "aggregate_metrics.csv",
        analysis_root / "reference_agreement.csv",
        analysis_root / "ranking_by_reference.csv",
        STUDY_ROOT / "results" / "figures" / "manifest.json",
        matrix_figure,
        effect_figure,
    ]
    for source in DATASETS.values():
        for model in MODELS:
            predictions = prediction_path(source, model, "gt_bbox")
            inputs.extend((predictions, predictions.with_name("manifest.json")))
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "status": "completed",
            "inputs": {
                portable_path(path, REPO_ROOT): sha256_file(path)
                for path in sorted(set(inputs))
            },
            "outputs": {
                portable_path(path, REPO_ROOT): sha256_file(path)
                for path in sorted(outputs.values())
            },
        },
    )
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
