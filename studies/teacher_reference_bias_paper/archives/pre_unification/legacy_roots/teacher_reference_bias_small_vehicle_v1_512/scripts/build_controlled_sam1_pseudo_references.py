from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias.pseudo_reference import (
    build_sam1_pseudo_reference_rows,
    read_prediction_jsonl,
)
from yolo_sam.runtime.manifest import (
    acquire_run_lock,
    finish_run_manifest,
    new_run_manifest,
    validate_completed_run_output,
    write_run_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a controlled SAM1 pseudo-reference from frozen GT-bbox "
            "predictions. This is a construction baseline, not human ground truth."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-model-id", required=True)
    parser.add_argument("--expected-model-version", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config-hash", default="not-provided")
    parser.add_argument("--run-id", default="controlled-sam1-pseudo-reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"{args.output} already exists. Use --force for an intentional rebuild."
        )
    source_manifest_path = args.predictions.parent / "manifest.json"
    source_manifest = validate_completed_run_output(
        source_manifest_path,
        output_name="predictions",
        output_path=args.predictions,
    )
    source_parameters = source_manifest.get("parameters", {})
    source_model_config = source_parameters.get("model_config", {})
    if source_manifest.get("stage") != "gt_bbox_segmentation":
        raise ValueError("SAM1 pseudo reference source must be GT-bbox segmentation")
    if source_parameters.get("model") != "sam1" or source_parameters.get("prompt_type") != "gt_bbox":
        raise ValueError("SAM1 pseudo reference source manifest has the wrong protocol")
    if source_model_config.get("model_id") != args.expected_model_id or source_model_config.get("revision") != args.expected_model_version:
        raise ValueError("SAM1 pseudo reference source manifest has the wrong model identity")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer_lock = acquire_run_lock(args.output.parent / ".pseudo_reference_writer.lock")
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = new_run_manifest(
        project_root=ROOT,
        run_id=args.run_id,
        stage="controlled_pseudo_reference_construction",
        config_hash=args.config_hash,
        inputs={
            "source_predictions": str(args.predictions.resolve()),
            "source_prediction_manifest": str(source_manifest_path.resolve()),
        },
        parameters={
            "reference_type": "pseudo_sam1",
            "construction_role": "teacher_self_reference_baseline",
            "expected_teacher_model_id": args.expected_model_id,
            "expected_teacher_model_version": args.expected_model_version,
            "source_prediction_run_id": source_manifest.get("run_id"),
            "source_prediction_config_hash": source_manifest.get("config_hash"),
            "teacher_inference_interface": "sam1_native_bbox",
            "known_positive_empty_reference_policy": "score_zero",
            "scientific_warning": (
                "A non-empty SAM1 prediction evaluated against this identical mask "
                "is an identity control, not segmentation performance. Empty pseudo "
                "references are missing labels for known instances and score zero."
            ),
        },
    )
    write_run_manifest(manifest_path, manifest)
    try:
        predictions = read_prediction_jsonl(args.predictions)
        references = build_sam1_pseudo_reference_rows(
            predictions,
            expected_model_id=args.expected_model_id,
            expected_model_version=args.expected_model_version,
        )
        temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            for row in references:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
        temporary_path.replace(args.output)
        manifest["outputs"] = {
            "pseudo_references": str(args.output.resolve()),
            "instances": len(references),
            "empty_references": sum(
                bool(reference["reference_is_empty"])
                for reference in references
            ),
        }
    except Exception as exc:
        temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary_path.unlink(missing_ok=True)
        finish_run_manifest(manifest, status="failed", error=str(exc))
        write_run_manifest(manifest_path, manifest)
        writer_lock.close()
        raise
    finish_run_manifest(manifest, status="completed")
    write_run_manifest(manifest_path, manifest)
    writer_lock.close()
    print(args.output)
    print(manifest_path)


if __name__ == "__main__":
    main()
