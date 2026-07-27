from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SplitCandidate:
    image_id: str
    source_scene_id: str
    stratum: str
    instance_count: int = 1

    def __post_init__(self) -> None:
        if not self.image_id or not self.source_scene_id or not self.stratum:
            raise ValueError("image_id, source_scene_id, and stratum are required")
        if self.instance_count <= 0:
            raise ValueError("instance_count must be positive")


@dataclass(frozen=True)
class SplitManifestRow:
    image_id: str
    source_scene_id: str
    stratum: str
    instance_count: int
    split: str


def validate_split_fractions(split_fractions: dict[str, float]) -> None:
    if len(split_fractions) < 2:
        raise ValueError("At least two splits are required")
    if any(value <= 0 for value in split_fractions.values()):
        raise ValueError("Every split fraction must be positive")
    if abs(sum(split_fractions.values()) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")


def _group_candidates(
    candidates: Iterable[SplitCandidate],
) -> dict[str, list[SplitCandidate]]:
    grouped: dict[str, list[SplitCandidate]] = defaultdict(list)
    image_ids: set[str] = set()
    for candidate in candidates:
        if candidate.image_id in image_ids:
            raise ValueError(f"Duplicate image_id in split candidates: {candidate.image_id}")
        image_ids.add(candidate.image_id)
        grouped[candidate.source_scene_id].append(candidate)
    return dict(grouped)


def grouped_stratified_split(
    candidates: Iterable[SplitCandidate],
    split_fractions: dict[str, float],
    seed: int,
) -> list[SplitManifestRow]:
    validate_split_fractions(split_fractions)
    grouped = _group_candidates(candidates)
    if len(grouped) < len(split_fractions):
        raise ValueError("There are fewer source scenes than requested splits")

    total_images = sum(len(rows) for rows in grouped.values())
    total_instances = sum(row.instance_count for rows in grouped.values() for row in rows)
    total_strata = Counter(row.stratum for rows in grouped.values() for row in rows)
    targets = {
        split: {
            "images": total_images * fraction,
            "instances": total_instances * fraction,
            "strata": {
                stratum: count * fraction
                for stratum, count in total_strata.items()
            },
        }
        for split, fraction in split_fractions.items()
    }

    rng = random.Random(seed)
    group_items = list(grouped.items())
    rng.shuffle(group_items)
    group_items.sort(
        key=lambda item: (
            -sum(row.instance_count for row in item[1]),
            -len(item[1]),
        )
    )

    assignments: dict[str, str] = {}
    current_images = Counter()
    current_instances = Counter()
    current_strata: dict[str, Counter[str]] = {
        split: Counter() for split in split_fractions
    }

    split_names = list(split_fractions)
    for index, (scene_id, rows) in enumerate(group_items):
        if index < len(split_names):
            selected_split = split_names[index]
        else:
            group_images = len(rows)
            group_instances = sum(row.instance_count for row in rows)
            group_strata = Counter(row.stratum for row in rows)

            def assignment_cost(split: str) -> float:
                image_target = max(targets[split]["images"], 1.0)
                instance_target = max(targets[split]["instances"], 1.0)
                image_ratio = (current_images[split] + group_images) / image_target
                instance_ratio = (current_instances[split] + group_instances) / instance_target
                stratum_ratios = []
                for stratum, target in targets[split]["strata"].items():
                    if target > 0:
                        stratum_ratios.append(
                            (current_strata[split][stratum] + group_strata[stratum]) / target
                        )
                return image_ratio**2 + instance_ratio**2 + sum(
                    ratio**2 for ratio in stratum_ratios
                ) / max(len(stratum_ratios), 1)

            selected_split = min(
                split_names,
                key=lambda split: (assignment_cost(split), current_images[split], split),
            )

        assignments[scene_id] = selected_split
        current_images[selected_split] += len(rows)
        current_instances[selected_split] += sum(row.instance_count for row in rows)
        current_strata[selected_split].update(row.stratum for row in rows)

    manifest = [
        SplitManifestRow(
            image_id=row.image_id,
            source_scene_id=row.source_scene_id,
            stratum=row.stratum,
            instance_count=row.instance_count,
            split=assignments[row.source_scene_id],
        )
        for scene_rows in grouped.values()
        for row in scene_rows
    ]
    validate_split_manifest(manifest)
    return sorted(manifest, key=lambda row: (row.split, row.source_scene_id, row.image_id))


def validate_split_manifest(rows: Iterable[SplitManifestRow]) -> None:
    scene_splits: dict[str, set[str]] = defaultdict(set)
    image_splits: dict[str, set[str]] = defaultdict(set)
    count = 0
    for row in rows:
        count += 1
        scene_splits[row.source_scene_id].add(row.split)
        image_splits[row.image_id].add(row.split)
    if count == 0:
        raise ValueError("Split manifest is empty")

    leaking_scenes = {
        scene: sorted(splits)
        for scene, splits in scene_splits.items()
        if len(splits) > 1
    }
    if leaking_scenes:
        raise ValueError(f"Source-scene split leakage detected: {leaking_scenes}")

    duplicate_images = {
        image_id: sorted(splits)
        for image_id, splits in image_splits.items()
        if len(splits) > 1
    }
    if duplicate_images:
        raise ValueError(f"Images occur in multiple splits: {duplicate_images}")


def write_split_manifest(
    rows: list[SplitManifestRow],
    csv_path: Path,
    json_path: Path,
) -> None:
    validate_split_manifest(rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
