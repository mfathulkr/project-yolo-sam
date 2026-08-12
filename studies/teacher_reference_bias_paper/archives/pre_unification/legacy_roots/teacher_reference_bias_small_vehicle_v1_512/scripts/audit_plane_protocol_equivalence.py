from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
DEFAULT_PLANE_PROTOCOL = (
    REPO_ROOT / "studies" / "teacher_reference_bias_v2_512" / "configs" / "protocol.yaml"
)
DEFAULT_SMALL_VEHICLE_PROTOCOL = STUDY_ROOT / "configs" / "protocol.yaml"
DEFAULT_OUTPUT = STUDY_ROOT / "results" / "audits" / "plane_protocol_equivalence.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that plane and small-vehicle protocols differ only by study_id."
    )
    parser.add_argument("--plane-protocol", type=Path, default=DEFAULT_PLANE_PROTOCOL)
    parser.add_argument(
        "--small-vehicle-protocol",
        type=Path,
        default=DEFAULT_SMALL_VEHICLE_PROTOCOL,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Protocol must be a mapping: {path}")
    return payload


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    args = parse_args()
    plane = load_protocol(args.plane_protocol)
    small_vehicle = load_protocol(args.small_vehicle_protocol)
    plane_study_id = plane.pop("study_id", None)
    small_vehicle_study_id = small_vehicle.pop("study_id", None)
    equivalent = plane == small_vehicle
    report = {
        "status": "completed" if equivalent else "failed",
        "equivalent_except_study_id": equivalent,
        "plane_study_id": plane_study_id,
        "small_vehicle_study_id": small_vehicle_study_id,
        "plane_protocol_hash_without_study_id": canonical_hash(plane),
        "small_vehicle_protocol_hash_without_study_id": canonical_hash(small_vehicle),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not equivalent:
        raise ValueError(
            "Plane and small-vehicle protocols differ in fields other than study_id"
        )
    print(args.output)


if __name__ == "__main__":
    main()
