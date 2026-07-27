from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_NAMES = (
    "numpy",
    "opencv-python",
    "pandas",
    "Pillow",
    "matplotlib",
    "pycocotools",
    "python-docx",
    "PyYAML",
    "safetensors",
    "scipy",
    "torch",
    "transformers",
    "ultralytics",
)
RECORDED_ENVIRONMENT_VARIABLES = (
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_ALLOC_CONF",
    "PYTORCH_CUDA_ALLOC_CONF",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def declared_file_fingerprints(values: dict[str, Any]) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for name, value in values.items():
        if not isinstance(value, str):
            continue
        path = Path(value)
        if not path.is_file():
            continue
        fingerprints[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return fingerprints


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def environment_snapshot(project_root: Path) -> dict[str, Any]:
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    cuda: dict[str, Any]
    try:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "torch_cuda_version": torch.version.cuda,
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:
        cuda = {"available": False, "inspection_error": str(exc)}

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
        "environment_variables": {
            name: os.environ[name]
            for name in RECORDED_ENVIRONMENT_VARIABLES
            if name in os.environ
        },
        "git": {
            "head": _git(project_root, "rev-parse", "HEAD"),
            "branch": _git(project_root, "branch", "--show-current"),
            "dirty": bool(_git(project_root, "status", "--porcelain=v1")),
        },
    }


def write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def new_run_manifest(
    *,
    project_root: Path,
    run_id: str,
    stage: str,
    config_hash: str,
    inputs: dict[str, Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    input_fingerprints = declared_file_fingerprints(inputs)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "stage": stage,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "finished_at_utc": None,
        "config_hash": config_hash,
        "inputs": inputs,
        "input_file_fingerprints": input_fingerprints,
        "input_fingerprint_capture": "start",
        "input_file_fingerprints_at_finish": None,
        "input_drift": [],
        "parameters": parameters,
        "environment": environment_snapshot(project_root),
        "error": None,
    }


def finish_run_manifest(
    manifest: dict[str, Any],
    *,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    if status not in {"completed", "failed"}:
        raise ValueError(f"Invalid final run status: {status}")
    manifest["status"] = status
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["error"] = error
    inputs = manifest.get("inputs")
    if isinstance(inputs, dict):
        finish_fingerprints = declared_file_fingerprints(inputs)
        start_fingerprints = manifest.get("input_file_fingerprints")
        if not isinstance(start_fingerprints, dict):
            start_fingerprints = finish_fingerprints
            manifest["input_file_fingerprints"] = start_fingerprints
            manifest["input_fingerprint_capture"] = "finish_only"
        manifest["input_file_fingerprints_at_finish"] = finish_fingerprints
        manifest["input_drift"] = sorted(
            name
            for name in set(start_fingerprints) | set(finish_fingerprints)
            if start_fingerprints.get(name) != finish_fingerprints.get(name)
        )
    outputs = manifest.get("outputs")
    if isinstance(outputs, dict):
        manifest["output_file_fingerprints"] = declared_file_fingerprints(outputs)
    return manifest
