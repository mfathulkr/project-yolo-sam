from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
STUDY_CLI = STUDY_ROOT / "scripts" / "study.py"
DATASET_IDS = ("isaid_small_vehicle", "samrs_sota_small_vehicle")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for both per-dataset post-training pipelines, then build "
            "and validate every canonical small-vehicle report and bundle."
        )
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest root must be an object: {path}")
    return payload


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def dependency_manifests() -> list[Path]:
    return [
        STUDY_ROOT
        / "results"
        / "post_training"
        / dataset_id
        / "manifest.json"
        for dataset_id in DATASET_IDS
    ]


def wait_for_dependencies(paths: list[Path], poll_seconds: int) -> None:
    while True:
        states = {path: read_manifest(path).get("status") for path in paths}
        failed = [path for path, status in states.items() if status == "failed"]
        if failed:
            raise RuntimeError(f"Post-training pipeline failed: {failed}")
        if all(status == "completed" for status in states.values()):
            return
        summary = ", ".join(
            f"{path.parent.name}={status or 'missing'}"
            for path, status in states.items()
        )
        print(f"[{utc_now()}] Waiting for post-training: {summary}", flush=True)
        time.sleep(poll_seconds)


def command_plan() -> list[list[str]]:
    python = sys.executable
    scripts = STUDY_ROOT / "scripts"
    return [
        [python, str(STUDY_CLI), "analyze"],
        [python, str(STUDY_CLI), "figures"],
        [python, str(scripts / "write_full_metric_reports.py")],
        [python, str(scripts / "validate_full_metric_reports.py")],
        [python, str(scripts / "build_portable_bundles.py")],
        [
            python,
            str(scripts / "manage_local_assets.py"),
            "status",
            "--verify-hashes",
            "--strict",
        ],
    ]


def main() -> None:
    args = parse_args()
    output_manifest = (
        STUDY_ROOT / "results" / "finalization" / "manifest.json"
    )
    existing = read_manifest(output_manifest)
    if existing.get("status") == "completed":
        print(f"SKIP completed finalization: {output_manifest}")
        return

    dependencies = dependency_manifests()
    commands = command_plan()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "waiting_for_post_training",
        "started_at": utc_now(),
        "dependencies": [str(path) for path in dependencies],
        "commands": [" ".join(command) for command in commands],
        "completed_commands": [],
    }
    write_manifest(output_manifest, manifest)

    try:
        wait_for_dependencies(dependencies, args.poll_seconds)
        manifest["status"] = "running"
        write_manifest(output_manifest, manifest)
        for command in commands:
            print(f"[{utc_now()}] RUN {' '.join(command)}", flush=True)
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            manifest["completed_commands"].append(" ".join(command))
            write_manifest(output_manifest, manifest)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["finished_at"] = utc_now()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_manifest(output_manifest, manifest)
        raise

    manifest["status"] = "completed"
    manifest["finished_at"] = utc_now()
    write_manifest(output_manifest, manifest)
    print(output_manifest)


if __name__ == "__main__":
    main()
