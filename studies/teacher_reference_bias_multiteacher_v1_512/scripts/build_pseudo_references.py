from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


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
    write_jsonl,
)
from teacher_reference_bias_multiteacher.paths import (  # noqa: E402
    DATASETS,
    prediction_path,
    reference_path,
)
from teacher_reference_bias_multiteacher.pseudo_reference import (  # noqa: E402
    build_pseudo_reference_rows,
)


def main() -> None:
    for dataset_id, source in DATASETS.items():
        for teacher in ("sam2", "sam3"):
            predictions_path = prediction_path(source, teacher, "gt_bbox")
            output_path = reference_path(dataset_id, teacher)
            predictions = read_jsonl(predictions_path)
            references = build_pseudo_reference_rows(
                predictions,
                teacher=teacher,
            )
            if len(references) != source.teacher_instance_count:
                raise ValueError(
                    f"{dataset_id}/{teacher}: {source.teacher_instance_count} "
                    f"yerine {len(references)} referans üretildi"
                )
            write_jsonl(output_path, references)
            write_json(
                output_path.with_suffix(".manifest.json"),
                {
                    "schema_version": 1,
                    "status": "completed",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_id": dataset_id,
                    "reference_type": f"pseudo_{teacher}",
                    "construction": "frozen_gt_bbox_prediction_identity",
                    "instance_count": len(references),
                    "source_predictions": portable_path(predictions_path, REPO_ROOT),
                    "source_predictions_sha256": sha256_file(predictions_path),
                    "output": portable_path(output_path, REPO_ROOT),
                    "output_sha256": sha256_file(output_path),
                    "teacher_model_id": references[0]["teacher_model_id"],
                    "teacher_model_version": references[0]["teacher_model_version"],
                    "scientific_warning": (
                        "Öğretmen modelin aynı GT-bbox tahmini kendi pseudo "
                        "referansına karşı özdeşlik gereği 1.0 verir. Bu satır "
                        "başarı değil, teacher self-reference kontrolüdür."
                    ),
                },
            )
            print(output_path)


if __name__ == "__main__":
    main()
