from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias.reporting.analysis import sha256_file
from teacher_reference_bias.config import load_matched_study_config


def configuration_fingerprints(
    model_root: Path,
    checkpoint: Path,
) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    configuration_suffixes = {".json", ".txt", ".model", ".yaml", ".yml"}
    for path in sorted(model_root.rglob("*")):
        if (
            not path.is_file()
            or path.resolve() == checkpoint.resolve()
            or path.suffix.lower() not in configuration_suffixes
        ):
            continue
        relative = path.relative_to(model_root).as_posix()
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return rows, digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify pinned SAM model revisions and checkpoint hashes."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=STUDY_ROOT / "configs" / "protocol.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            STUDY_ROOT
            / "results"
            / "audits"
            / "segmenter_provenance.json"
        ),
    )
    return parser.parse_args()


def hf_model_row(name: str, config: dict[str, Any]) -> dict[str, Any]:
    model_id = str(config["model_id"])
    revision = str(config["revision"])
    snapshot = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    ).resolve()
    checkpoint = snapshot / "model.safetensors"
    actual_sha256 = sha256_file(checkpoint)
    expected_sha256 = str(config["checkpoint_sha256"])
    configuration_files, configuration_tree_sha256 = (
        configuration_fingerprints(snapshot, checkpoint)
    )
    return {
        "model": name,
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": str(snapshot),
        "checkpoint_path": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "configuration_files": configuration_files,
        "configuration_tree_sha256": configuration_tree_sha256,
        "passed": actual_sha256 == expected_sha256,
    }


def local_model_row(config: dict[str, Any]) -> dict[str, Any]:
    model_dir = Path(str(config["model_dir"]))
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    checkpoint = model_dir / str(config["checkpoint_file"])
    actual_sha256 = sha256_file(checkpoint)
    expected_sha256 = str(config["checkpoint_sha256"])
    configuration_files, configuration_tree_sha256 = (
        configuration_fingerprints(model_dir, checkpoint)
    )
    return {
        "model": "sam3",
        "model_id": str(config["model_id"]),
        "revision": "local_checkpoint",
        "snapshot_path": str(model_dir.resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "configuration_files": configuration_files,
        "configuration_tree_sha256": configuration_tree_sha256,
        "passed": actual_sha256 == expected_sha256,
    }


def main() -> None:
    args = parse_args()
    protocol = load_matched_study_config(args.protocol)
    rows = [
        hf_model_row("sam1", protocol.segmenter_configs["sam1"]),
        hf_model_row("sam2", protocol.segmenter_configs["sam2"]),
        local_model_row(protocol.segmenter_configs["sam3"]),
    ]
    passed = all(bool(row["passed"]) for row in rows)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if passed else "fail",
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "models": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
