from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT

DEFAULT_ROOTS = (
    "studies/teacher_reference_bias_v1/configs",
    "studies/teacher_reference_bias_v1/docs",
    "studies/teacher_reference_bias_v1/scripts",
    "studies/teacher_reference_bias_v1/src",
    "src/yolo_sam",
)
ROOT_FILES = (
    ".gitignore",
    "README.md",
    "REPORT.md",
    "requirements.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a hash-based pre-refactor repository snapshot.")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            STUDY_ROOT
            / "results"
            / "legacy"
            / "repository_snapshot.json"
        ),
    )
    return parser.parse_args()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relevant_files() -> list[Path]:
    files: set[Path] = set()
    for relative_root in DEFAULT_ROOTS:
        root = ROOT / relative_root
        if root.exists():
            files.update(path for path in root.rglob("*") if path.is_file())
    for relative_path in ROOT_FILES:
        path = ROOT / relative_path
        if path.exists():
            files.add(path)
    return sorted(files)


def build_manifest() -> dict[str, Any]:
    tracked = set(git("ls-files").splitlines())
    file_rows = []
    for path in relevant_files():
        relative = path.relative_to(ROOT).as_posix()
        stat = path.stat()
        file_rows.append(
            {
                "path": relative,
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
                "tracked": relative in tracked,
            }
        )

    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "head": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "status_porcelain": git("status", "--porcelain=v1").splitlines(),
        },
        "scope": {
            "directories": list(DEFAULT_ROOTS),
            "root_files": list(ROOT_FILES),
            "excluded": [
                "raw datasets",
                "model checkpoints",
                "training run tensors",
                "bulk result images outside presentation directories",
            ],
        },
        "artifact_validity": {
            "studies/isaid_vehicle_study": {
                "status": "historical_context_only",
                "reason": "Protocol is not matched to the planned plane study.",
            },
            "studies/samrs_sota_plane_study": {
                "status": "invalid_for_paper_evidence",
                "reason": (
                    "The legacy protocol was not matched to iSAID, source-scene "
                    "split leakage existed, and some prompts used mask-derived boxes."
                ),
            },
        },
        "files": file_rows,
    }


def main() -> None:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_manifest(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
