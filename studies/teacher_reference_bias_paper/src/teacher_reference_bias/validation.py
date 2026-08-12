from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _valid_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _validate_legacy_resume_record(
    *,
    record_path: Path,
    runtime_path: Path,
    args_resume_path: Path,
    training_manifest: Mapping[str, Any],
) -> None:
    if training_manifest.get("input_drift") != []:
        raise ValueError("Legacy resume manifest still declares input drift")
    if training_manifest.get("expected_input_drift") != []:
        raise ValueError("Legacy resume manifest has a stale expected drift")
    if not record_path.is_file():
        raise ValueError("Legacy resume fingerprint record is missing")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (
        record.get("schema_version") != 1
        or record.get("status") != "record_only_original_bytes_unavailable"
    ):
        raise ValueError("Legacy resume fingerprint record has the wrong schema")

    fingerprints = (
        record.get("start_fingerprint"),
        record.get("finish_fingerprint_after_in_place_resume"),
    )
    for fingerprint in fingerprints:
        if not isinstance(fingerprint, Mapping):
            raise ValueError("Legacy resume fingerprint is incomplete")
        fingerprint_path = Path(str(fingerprint.get("path", "")))
        if (
            fingerprint_path.name != "last.pt"
            or not isinstance(fingerprint.get("bytes"), int)
            or fingerprint["bytes"] <= 0
            or not _valid_sha256(fingerprint.get("sha256"))
        ):
            raise ValueError("Legacy resume fingerprint is invalid")
        if fingerprint_path != runtime_path:
            raise ValueError("Legacy resume checkpoint paths disagree")

    if runtime_path != args_resume_path:
        raise ValueError("Resume checkpoint paths disagree")
    limitation = training_manifest.get("provenance_limitation")
    migration = training_manifest.get("provenance_migration", {})
    best_fingerprint = (
        training_manifest.get("output_file_fingerprints", {})
        .get("best_weights", {})
    )
    if not isinstance(limitation, str) or not limitation.strip():
        raise ValueError("Legacy resume provenance limitation is not declared")
    if (
        not isinstance(migration, Mapping)
        or migration.get("schema_version") != 1
        or migration.get("future_policy")
        != "resume_from_immutable_checkpoint_copy"
        or migration.get("best_weights_sha256") != best_fingerprint.get("sha256")
        or not _valid_sha256(migration.get("best_weights_sha256"))
    ):
        raise ValueError("Legacy resume migration record is incomplete")


def validate_detector_base_provenance(
    *,
    training_args: Mapping[str, Any],
    training_manifest: Mapping[str, Any],
    expected_base_model: str,
) -> None:
    expected_name = Path(expected_base_model).name
    manifest_inputs = training_manifest.get("inputs", {})
    manifest_parameters = training_manifest.get("parameters", {})

    declared_bases = (
        manifest_inputs.get("base_weights"),
        manifest_parameters.get("base_weights"),
    )
    if any(
        not value or Path(str(value)).name != expected_name
        for value in declared_bases
    ):
        raise ValueError(
            f"Training manifest does not declare {expected_name} as its base model"
        )

    runtime_model = training_args.get("model")
    if runtime_model and Path(str(runtime_model)).name == expected_name:
        return

    args_resume = training_args.get("resume")
    manifest_resume = manifest_inputs.get("resume_checkpoint")
    if manifest_parameters.get("resume") is not True:
        raise ValueError("Training args use a checkpoint without a resumed run")
    if not runtime_model or not args_resume:
        raise ValueError("Resumed training provenance is incomplete")

    runtime_path = Path(str(runtime_model))
    args_resume_path = Path(str(args_resume))
    legacy_record = manifest_inputs.get("resume_checkpoint_start_record")
    if not manifest_resume and legacy_record:
        _validate_legacy_resume_record(
            record_path=Path(str(legacy_record)),
            runtime_path=runtime_path,
            args_resume_path=args_resume_path,
            training_manifest=training_manifest,
        )
        return
    if not manifest_resume:
        raise ValueError("Resumed training provenance is incomplete")
    manifest_resume_path = Path(str(manifest_resume))
    if any(
        path.name != "last.pt"
        for path in (runtime_path, args_resume_path, manifest_resume_path)
    ):
        raise ValueError("Resumed training did not use last.pt")
    if not (runtime_path == args_resume_path == manifest_resume_path):
        raise ValueError("Resume checkpoint paths disagree")
