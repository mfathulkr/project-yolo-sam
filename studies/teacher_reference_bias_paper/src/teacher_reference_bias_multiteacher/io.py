from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path, repository_root: Path) -> str:
    try:
        relative = path.resolve().relative_to(repository_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Repository dışındaki yol manifest'e yazılamaz: {path}") from exc
    return relative.as_posix()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_hash_manifest(
    path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate all portable file hashes declared by a completed manifest."""
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise ValueError(f"Tamamlanmamış manifest: {path}")

    def validate_entries(entries: Any, *, group: str) -> None:
        if isinstance(entries, dict):
            normalized = [
                {"path": name, "sha256": expected}
                for name, expected in entries.items()
            ]
        elif isinstance(entries, list):
            normalized = entries
        else:
            raise ValueError(f"{path}: geçersiz {group} hash listesi")
        for entry in normalized:
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: geçersiz {group} hash kaydı")
            relative = Path(str(entry.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{path}: taşınabilir olmayan yol: {relative}")
            target = repository_root / relative
            if not target.is_file():
                raise FileNotFoundError(target)
            if entry.get("sha256") != sha256_file(target):
                raise ValueError(f"{path}: {group} hash uyuşmazlığı: {target}")

    if "inputs" in payload:
        validate_entries(payload["inputs"], group="input")
    if "outputs" in payload:
        validate_entries(payload["outputs"], group="output")
    elif "output" in payload and "output_sha256" in payload:
        validate_entries(
            [{"path": payload["output"], "sha256": payload["output_sha256"]}],
            group="output",
        )
    else:
        raise ValueError(f"{path}: output hash kaydı yok")
    return payload


def relativize_path_hash_manifest(path: Path, repository_root: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for group in ("inputs", "outputs"):
        if group not in payload:
            continue
        payload[group] = {
            portable_path(Path(name), repository_root): expected
            for name, expected in payload[group].items()
        }
    write_json(path, payload)
