from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


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
    if not runtime_model or not args_resume or not manifest_resume:
        raise ValueError("Resumed training provenance is incomplete")

    runtime_path = Path(str(runtime_model))
    args_resume_path = Path(str(args_resume))
    manifest_resume_path = Path(str(manifest_resume))
    if any(
        path.name != "last.pt"
        for path in (runtime_path, args_resume_path, manifest_resume_path)
    ):
        raise ValueError("Resumed training did not use last.pt")
    if not (runtime_path == args_resume_path == manifest_resume_path):
        raise ValueError("Resume checkpoint paths disagree")
