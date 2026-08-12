from __future__ import annotations

import importlib.metadata
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


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
SOURCE_SUFFIXES = {".py", ".toml", ".yaml", ".yml"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_repository_root(manifest_path: Path) -> Path:
    """Resolve repository-relative paths recorded in portable manifests."""
    for candidate in (manifest_path.parent, *manifest_path.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").is_file():
            return candidate
    raise ValueError(
        f"Cannot resolve repository root for relative manifest paths: {manifest_path}"
    )


def _resolve_manifest_file(manifest_path: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _manifest_repository_root(manifest_path.resolve()) / path


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


def _source_tree_fingerprint(project_root: Path) -> dict[str, Any]:
    paths: set[Path] = set()
    source_root = project_root / "src"
    if source_root.is_dir():
        paths.update(path for path in source_root.rglob("*") if path.is_file())
    studies_root = project_root / "studies"
    if studies_root.is_dir():
        for path in studies_root.rglob("*"):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            relative_parts = path.relative_to(studies_root).parts
            if any(part in {"configs", "scripts", "src"} for part in relative_parts):
                paths.add(path)
    for name in ("pyproject.toml", "pytest.ini"):
        path = project_root / name
        if path.is_file():
            paths.add(path)

    digest = hashlib.sha256()
    selected = sorted(path for path in paths if path.suffix in SOURCE_SUFFIXES)
    for path in selected:
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return {"sha256": digest.hexdigest(), "file_count": len(selected)}


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
        "source_tree": _source_tree_fingerprint(project_root),
    }


def write_run_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_run_lock(path: Path) -> TextIO:
    """Hold an exclusive writer lock until the returned handle is closed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"Another writer is already active for {path.parent}") from exc
    return handle


def validate_completed_run_output(
    manifest_path: Path,
    *,
    output_name: str,
    output_path: Path,
) -> dict[str, Any]:
    """Fail closed unless an upstream output belongs to a completed hashed run."""
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(
            f"Upstream run is not completed: {manifest_path} "
            f"(status={manifest.get('status')!r})"
        )
    fingerprints = manifest.get("output_file_fingerprints")
    if not isinstance(fingerprints, dict) or output_name not in fingerprints:
        raise ValueError(
            f"Upstream manifest has no fingerprint for {output_name!r}: "
            f"{manifest_path}"
        )
    fingerprint = fingerprints[output_name]
    if not isinstance(fingerprint, dict):
        raise ValueError(f"Invalid output fingerprint in {manifest_path}")
    if not output_path.is_file():
        raise FileNotFoundError(output_path)
    actual_bytes = output_path.stat().st_size
    actual_sha256 = _sha256_file(output_path)
    if fingerprint.get("bytes") != actual_bytes or fingerprint.get("sha256") != actual_sha256:
        raise ValueError(
            f"Upstream output fingerprint mismatch for {output_path}: "
            f"expected bytes={fingerprint.get('bytes')} sha256={fingerprint.get('sha256')}, "
            f"actual bytes={actual_bytes} sha256={actual_sha256}"
        )
    return manifest


def validate_completed_run_manifest(
    manifest_path: Path,
    *,
    allow_changed_input_names: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Validate a completed run and every non-exempt current file fingerprint.

    Exemptions apply only to named inputs. Output fingerprints always remain strict.
    Callers must independently validate the semantics of every exempt historical input.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(
            f"Run is not completed: {manifest_path} "
            f"(status={manifest.get('status')!r})"
        )
    input_drift = manifest.get("input_drift", [])
    if not isinstance(input_drift, list) or not all(
        isinstance(name, str) for name in input_drift
    ):
        raise ValueError(f"Invalid input_drift declaration: {manifest_path}")
    expected_input_drift = manifest.get("expected_input_drift", [])
    if not isinstance(expected_input_drift, list) or not all(
        isinstance(name, str) for name in expected_input_drift
    ):
        raise ValueError(
            f"Invalid expected_input_drift declaration: {manifest_path}"
        )
    declared_inputs = manifest.get("input_file_fingerprints_at_finish")
    if not isinstance(declared_inputs, dict):
        raise ValueError(f"Invalid input fingerprints: {manifest_path}")
    unknown_expected_drift = sorted(
        set(expected_input_drift) - set(declared_inputs)
    )
    if unknown_expected_drift:
        raise ValueError(
            "Expected input drift names must identify declared inputs: "
            f"{manifest_path}: {unknown_expected_drift}"
        )
    unexpected_input_drift = sorted(
        set(input_drift) - set(expected_input_drift)
    )
    if unexpected_input_drift:
        raise ValueError(
            f"Run inputs changed while the stage was active: {manifest_path}: "
            f"{unexpected_input_drift}"
        )

    fingerprint_groups = (
        ("input", manifest.get("input_file_fingerprints_at_finish")),
        ("output", manifest.get("output_file_fingerprints")),
    )
    for group_name, fingerprints in fingerprint_groups:
        if not isinstance(fingerprints, dict) or not fingerprints:
            raise ValueError(
                f"Completed run has no {group_name} fingerprints: {manifest_path}"
            )
        for name, fingerprint in fingerprints.items():
            if not isinstance(fingerprint, dict):
                raise ValueError(
                    f"Invalid {group_name} fingerprint {name!r}: {manifest_path}"
                )
            path_value = fingerprint.get("path")
            if not isinstance(path_value, str):
                raise ValueError(
                    f"Missing path for {group_name} fingerprint {name!r}: "
                    f"{manifest_path}"
                )
            path = _resolve_manifest_file(manifest_path, path_value)
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_bytes = path.stat().st_size
            actual_sha256 = _sha256_file(path)
            if (
                fingerprint.get("bytes") != actual_bytes
                or fingerprint.get("sha256") != actual_sha256
            ):
                if group_name == "input" and name in allow_changed_input_names:
                    continue
                raise ValueError(
                    f"Current {group_name} fingerprint mismatch for {path}: "
                    f"manifest={manifest_path}"
                )
    return manifest


def new_run_manifest(
    *,
    project_root: Path,
    run_id: str,
    stage: str,
    config_hash: str,
    inputs: dict[str, Any],
    parameters: dict[str, Any],
    expected_input_drift: tuple[str, ...] = (),
) -> dict[str, Any]:
    unknown_expected_drift = sorted(set(expected_input_drift) - set(inputs))
    if unknown_expected_drift:
        raise ValueError(
            "Expected input drift names must identify declared inputs: "
            f"{unknown_expected_drift}"
        )
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
        "expected_input_drift": sorted(set(expected_input_drift)),
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
