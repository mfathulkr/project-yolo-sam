from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
STUDY_CLI = STUDY_ROOT / "scripts" / "study.py"
MODELS = ("sam1", "sam2", "sam3")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a seed-42 detector, then complete detector evaluation "
            "and all YOLO-bbox SAM conditions for one dataset."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    return parser.parse_args()


def read_dataset_id(dataset_path: Path) -> str:
    payload = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    dataset_id = payload.get("dataset_id") if isinstance(payload, dict) else None
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError(f"Missing dataset_id in {dataset_path}")
    return dataset_id


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


def wait_for_training(manifest_path: Path, poll_seconds: int) -> None:
    while True:
        manifest = read_manifest(manifest_path)
        status = manifest.get("status")
        if status == "completed":
            return
        if status == "failed":
            raise RuntimeError(
                f"Detector training failed: {manifest.get('error', manifest_path)}"
            )
        print(
            f"[{utc_now()}] Waiting for detector training: "
            f"{manifest_path} status={status or 'missing'}",
            flush=True,
        )
        time.sleep(poll_seconds)


def study_command(*arguments: str) -> list[str]:
    return [sys.executable, str(STUDY_CLI), *arguments]


def command_plan(dataset_path: Path, device: str) -> list[list[str]]:
    common = ["--dataset", str(dataset_path), "--seed", "42"]
    commands = [
        study_command(
            "detect",
            *common,
            "--split",
            "test",
            "--device",
            device,
        )
    ]
    for model in MODELS:
        commands.append(
            study_command(
                "infer",
                "--dataset",
                str(dataset_path),
                "--model",
                model,
                "--bbox-source",
                "yolo_bbox",
                "--seed",
                "42",
                "--split",
                "test",
                "--device",
                device,
            )
        )
        commands.append(
            study_command(
                "evaluate",
                "--dataset",
                str(dataset_path),
                "--model",
                model,
                "--bbox-source",
                "yolo_bbox",
                "--seed",
                "42",
                "--split",
                "test",
            )
        )
    return commands


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    dataset_id = read_dataset_id(dataset_path)
    detector_manifest = (
        STUDY_ROOT
        / "results"
        / "detectors"
        / dataset_id
        / "seed_42"
        / "manifest.json"
    )
    output_manifest = (
        STUDY_ROOT
        / "results"
        / "post_training"
        / dataset_id
        / "manifest.json"
    )
    existing = read_manifest(output_manifest)
    if existing.get("status") == "completed":
        print(f"SKIP completed post-training pipeline: {output_manifest}")
        return

    commands = command_plan(dataset_path, str(args.device))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "detector_seed": 42,
        "device": str(args.device),
        "status": "waiting_for_training",
        "started_at": utc_now(),
        "training_manifest": str(detector_manifest),
        "commands": [" ".join(command) for command in commands],
        "completed_commands": [],
    }
    write_manifest(output_manifest, manifest)

    try:
        wait_for_training(detector_manifest, args.poll_seconds)
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
