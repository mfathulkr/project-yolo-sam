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


def main() -> None:
    analysis_root = STUDY_ROOT / "results" / "analysis"
    outputs = write_comparison_report(
        output_dir=STUDY_ROOT / "reports" / "teacher_comparison",
        aggregates=pd.read_csv(analysis_root / "aggregate_metrics.csv"),
        agreement=pd.read_csv(analysis_root / "reference_agreement.csv"),
        rankings=pd.read_csv(analysis_root / "ranking_by_reference.csv"),
        matrix_figure=STUDY_ROOT / "results" / "figures" / "model_reference_iou_matrix.png",
        effect_figure=STUDY_ROOT / "results" / "figures" / "reference_effect_with_ci.png",
    )
    for path in outputs.values():
        print(path)


if __name__ == "__main__":
    main()
