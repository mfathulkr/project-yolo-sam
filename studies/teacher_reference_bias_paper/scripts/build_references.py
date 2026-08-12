from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


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
    write_jsonl,
)
from teacher_reference_bias_multiteacher.paths import (  # noqa: E402
    DATASETS,
    MODELS,
    prediction_path,
    reference_path,
)
from teacher_reference_bias_multiteacher.pseudo_reference import (  # noqa: E402
    build_pseudo_reference_rows,
    validate_pseudo_reference_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GT-bbox SAM tahminlerinden dondurulmuş pseudo referansları üret."
    )
    parser.add_argument("--experiment", choices=tuple(DATASETS))
    return parser.parse_args()


def output_reference_type(dataset_family: str, teacher: str) -> str:
    if dataset_family == "samrs" and teacher == "sam1":
        return "reproduced_pseudo_sam1"
    return f"pseudo_{teacher}"


def build_one(experiment_id: str, teacher: str) -> Path:
    source = DATASETS[experiment_id]
    predictions_path = prediction_path(source, teacher, "gt_bbox")
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    predictions = read_jsonl(predictions_path)
    if len(predictions) != source.instance_count:
        raise ValueError(
            f"{experiment_id}/{teacher}: {source.instance_count} yerine "
            f"{len(predictions)} tahmin var"
        )
    reference_type = output_reference_type(source.dataset_family, teacher)
    output_path = reference_path(source, reference_type)
    references = build_pseudo_reference_rows(predictions, teacher=teacher)
    for row in references:
        row["reference_type"] = reference_type
        row["dataset_id"] = source.dataset_id
        row["construction"] = "frozen_gt_bbox_prediction_identity"
    write_jsonl(output_path, references)

    identity_rows = [dict(row, reference_type=f"pseudo_{teacher}") for row in references]
    validate_pseudo_reference_identity(
        predictions,
        identity_rows,
        teacher=teacher,
    )
    empty_count = sum(bool(row["reference_is_empty"]) for row in references)
    prediction_manifest = predictions_path.with_name("manifest.json")
    manifest_path = output_path.with_suffix(".manifest.json")
    write_json(
        manifest_path,
        {
            "schema_version": 3,
            "status": "completed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": experiment_id,
            "dataset_id": source.dataset_id,
            "reference_type": reference_type,
            "teacher": teacher,
            "teacher_prompt_type": "gt_bbox",
            "teacher_inference_interface": (
                "sam3_tracker_pvs" if teacher == "sam3" else "native_bbox_prompt"
            ),
            "construction": "frozen_gt_bbox_prediction_identity",
            "instance_count": len(references),
            "empty_reference_count": empty_count,
            "empty_reference_rate": empty_count / len(references),
            "known_positive_empty_reference_policy": "score_zero",
            "inputs": {
                portable_path(predictions_path, REPO_ROOT): sha256_file(predictions_path),
                **(
                    {
                        portable_path(prediction_manifest, REPO_ROOT): sha256_file(
                            prediction_manifest
                        )
                    }
                    if prediction_manifest.is_file()
                    else {}
                ),
            },
            "outputs": {
                portable_path(output_path, REPO_ROOT): sha256_file(output_path)
            },
            "scientific_role": (
                "coverage-aware identity control for the matching GT-bbox teacher; "
                "not independent ground truth"
            ),
        },
    )
    return output_path


def main() -> None:
    args = parse_args()
    experiments = (args.experiment,) if args.experiment else tuple(DATASETS)
    for experiment_id in experiments:
        for teacher in MODELS:
            print(build_one(experiment_id, teacher))


if __name__ == "__main__":
    main()
