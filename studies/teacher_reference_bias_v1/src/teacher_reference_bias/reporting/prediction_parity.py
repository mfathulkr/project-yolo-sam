from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def summarize_prediction_masks(path: Path) -> dict[str, Any]:
    """Hash only the prediction identity, status, and encoded mask content."""
    records: list[dict[str, Any]] = []
    seen_instance_ids: set[str] = set()
    model_versions: set[str] = set()
    ok_count = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            instance_id = str(row["instance_id"])
            if instance_id in seen_instance_ids:
                raise ValueError(
                    f"Duplicate instance_id {instance_id!r} in {path}:{line_number}"
                )
            seen_instance_ids.add(instance_id)
            status = str(row.get("status", ""))
            if status == "ok":
                ok_count += 1
            model_versions.add(str(row.get("model_version", "")))
            records.append(
                {
                    "instance_id": instance_id,
                    "status": status,
                    "predicted_mask_rle": row.get("predicted_mask_rle"),
                }
            )

    records.sort(key=lambda row: row["instance_id"])
    digest = hashlib.sha256()
    for record in records:
        canonical = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")

    return {
        "row_count": len(records),
        "ok_count": ok_count,
        "model_versions": sorted(model_versions),
        "canonical_mask_sha256": digest.hexdigest(),
    }
