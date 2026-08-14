from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias.config import (  # noqa: E402
    DatasetStudyConfig,
    MatchedStudyConfig,
    load_dataset_study_config,
    load_matched_study_config,
)
from yolo_sam.data.isaid import (  # noqa: E402
    rasterize_clipped_annotation,
    summarize_box_overlap,
    write_yolo_data_yaml,
    yolo_label_line,
)
from yolo_sam.data.prepared_validation import (  # noqa: E402
    validate_prepared_content_manifest,
)


MASTER_SPLITS = ("train", "validation", "test_pool")
STRATA = (
    "no_overlap__low_mask_area",
    "no_overlap__high_mask_area",
    "overlap__low_mask_area",
    "overlap__high_mask_area",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a 512-image, source-scene-disjoint matched corpus from "
            "the fully validated v1 prepared tile pool."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=STUDY_ROOT / "configs" / "protocol.yaml",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_master(
    dataset: DatasetStudyConfig,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[dict[str, Any]]]:
    master_root = dataset.master_prepared_root
    if master_root is None:
        raise ValueError("Dataset config requires master_prepared_root")
    manifest_path = master_root / "content_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = validate_prepared_content_manifest(master_root, manifest)
    if errors:
        raise ValueError(
            "Master prepared corpus failed content-manifest validation: "
            + "; ".join(errors[:10])
        )

    metadata_frames: list[pd.DataFrame] = []
    records_by_file: dict[str, dict[str, Any]] = {}
    categories: list[dict[str, Any]] | None = None
    for split in MASTER_SPLITS:
        split_root = master_root / split
        metadata = pd.read_csv(split_root / "metadata.csv")
        metadata["_master_split"] = split
        metadata_frames.append(metadata)

        coco = json.loads(
            (split_root / "_annotations.coco.json").read_text(encoding="utf-8")
        )
        if categories is None:
            categories = list(coco["categories"])
        elif categories != list(coco["categories"]):
            raise ValueError("Master split COCO categories are inconsistent")

        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        for image in coco["images"]:
            file_name = str(image["file_name"])
            if file_name in records_by_file:
                raise ValueError(f"Duplicate master tile file name: {file_name}")
            records_by_file[file_name] = {
                "split": split,
                "image": image,
                "annotations": annotations_by_image.get(int(image["id"]), []),
            }

    combined = pd.concat(metadata_frames, ignore_index=True)
    if combined["file_name"].duplicated().any():
        duplicates = combined.loc[
            combined["file_name"].duplicated(), "file_name"
        ].tolist()
        raise ValueError(f"Duplicate master metadata rows: {duplicates[:10]}")
    missing_records = sorted(set(combined["file_name"]) - set(records_by_file))
    if missing_records:
        raise ValueError(f"Master COCO records missing: {missing_records[:10]}")
    return combined, records_by_file, categories or []


def load_isaid_source_annotations(
    dataset: DatasetStudyConfig,
) -> dict[tuple[str, int], dict[str, Any]]:
    source_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for split in ("train", "val"):
        path = (
            dataset.raw_root
            / split
            / "Annotations"
            / f"iSAID_{split}.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        target_ids = {
            int(category["id"])
            for category in payload["categories"]
            if str(category["name"]).strip().lower()
            == dataset.target_category.strip().lower()
        }
        if len(target_ids) != 1:
            raise ValueError(
                f"Expected one {dataset.target_category!r} category in {path}"
            )
        file_name_by_image_id = {
            int(image["id"]): str(image["file_name"])
            for image in payload["images"]
        }
        for annotation in payload["annotations"]:
            if int(annotation["category_id"]) not in target_ids:
                continue
            key = (
                file_name_by_image_id[int(annotation["image_id"])],
                int(annotation["id"]),
            )
            if key in source_by_key:
                raise ValueError(f"Duplicate source annotation key: {key}")
            source_by_key[key] = annotation
    return source_by_key


def canonicalize_isaid_annotations(
    metadata: pd.DataFrame,
    records_by_file: dict[str, dict[str, Any]],
    *,
    dataset: DatasetStudyConfig,
    min_instance_area: int,
) -> pd.DataFrame:
    source_by_key = load_isaid_source_annotations(dataset)
    output = metadata.copy()
    for row_index, row in output.iterrows():
        file_name = str(row["file_name"])
        record = records_by_file[file_name]
        boxes: list[list[float]] = []
        canonical_annotations: list[dict[str, Any]] = []
        total_area = 0
        for source_annotation_record in record["annotations"]:
            key = (
                str(row["source_file_name"]),
                int(source_annotation_record["source_annotation_id"]),
            )
            source_annotation = source_by_key.get(key)
            if source_annotation is None:
                raise KeyError(f"Missing official iSAID annotation: {key}")
            clipped = rasterize_clipped_annotation(
                source_annotation,
                tile_x=int(row["tile_x"]),
                tile_y=int(row["tile_y"]),
                tile_size=int(row["tile_size"]),
                min_instance_area=min_instance_area,
            )
            if clipped is None:
                continue
            segmentation, bbox, area = clipped
            annotation = dict(source_annotation_record)
            annotation["segmentation"] = segmentation
            annotation["bbox"] = bbox
            annotation["area"] = area
            canonical_annotations.append(annotation)
            boxes.append(bbox)
            total_area += area
        record["annotations"] = canonical_annotations
        max_pair_iou, overlap_pairs = summarize_box_overlap(boxes)
        output.at[row_index, "num_objects"] = len(canonical_annotations)
        output.at[row_index, "mask_area_pixels"] = total_area
        output.at[row_index, "mask_area_ratio"] = total_area / float(
            int(row["tile_size"]) ** 2
        )
        output.at[row_index, "max_pair_bbox_iou"] = max_pair_iou
        output.at[row_index, "num_bbox_overlap_pairs"] = overlap_pairs
        output.at[row_index, "has_bbox_overlap"] = overlap_pairs > 0
    return output


def add_strata(
    metadata: pd.DataFrame,
    *,
    area_threshold: float,
    overlap_threshold: float,
) -> pd.DataFrame:
    output = metadata.copy()
    positive = (output["num_objects"] >= 1) & (output["mask_area_ratio"] > 0.0)
    output["overlap_group"] = np.select(
        [
            output["max_pair_bbox_iou"] <= 0.0,
            output["max_pair_bbox_iou"] >= overlap_threshold,
        ],
        ["no_overlap", "overlap"],
        default="ambiguous_overlap",
    )
    output["area_group"] = np.where(
        output["mask_area_ratio"] >= area_threshold,
        "high_mask_area",
        "low_mask_area",
    )
    output["stratum"] = np.where(
        positive & (output["overlap_group"] != "ambiguous_overlap"),
        output["overlap_group"] + "__" + output["area_group"],
        "negative",
    )
    return output


def select_test_scene_pool(
    positive: pd.DataFrame,
    *,
    target_per_stratum: int,
    seed: int,
    trials: int = 512,
) -> tuple[set[str], dict[str, int]]:
    scene_counts = (
        positive.groupby(["source_scene_id", "stratum"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STRATA, fill_value=0)
    )
    scene_ids = np.asarray(scene_counts.index.astype(str))
    matrix = scene_counts.to_numpy(dtype=np.int64)
    target = np.full(len(STRATA), target_per_stratum, dtype=np.int64)
    best: tuple[tuple[int, int], np.ndarray, np.ndarray] | None = None

    for trial in range(trials):
        rng = random.Random(seed + trial)
        priority = np.asarray([rng.random() for _ in scene_ids])
        available = np.ones(len(scene_ids), dtype=bool)
        coverage = np.zeros(len(STRATA), dtype=np.int64)
        selected_indices: list[int] = []
        while np.any(coverage < target):
            deficit = np.maximum(target - coverage, 0)
            gains = np.minimum(matrix, deficit).sum(axis=1)
            spills = np.maximum(matrix - deficit, 0).sum(axis=1)
            scores = gains * 1_000_000 - spills * 100 + priority
            scores[~available] = -1
            scores[gains == 0] = -1
            chosen = int(np.argmax(scores))
            if scores[chosen] < 0:
                break
            selected_indices.append(chosen)
            available[chosen] = False
            coverage += matrix[chosen]

        if np.any(coverage < target):
            continue
        objective = (
            len(selected_indices),
            int(np.maximum(coverage - target, 0).sum()),
        )
        if best is None or objective < best[0]:
            best = (objective, np.asarray(selected_indices), coverage.copy())

    if best is None:
        available_counts = {
            stratum: int(positive["stratum"].eq(stratum).sum())
            for stratum in STRATA
        }
        raise RuntimeError(
            "No source-scene-disjoint test scene pool can satisfy the target: "
            f"{available_counts}"
        )
    selected_scenes = set(scene_ids[best[1]].tolist())
    coverage = {
        stratum: int(value)
        for stratum, value in zip(STRATA, best[2], strict=True)
    }
    return selected_scenes, coverage


def select_exact_test_rows(
    positive: pd.DataFrame,
    *,
    scene_pool: set[str],
    target_per_stratum: int,
    seed: int,
) -> pd.DataFrame:
    pool = positive[positive["source_scene_id"].astype(str).isin(scene_pool)]
    selected: list[pd.DataFrame] = []
    for index, stratum in enumerate(STRATA):
        candidates = pool[pool["stratum"] == stratum].copy()
        if len(candidates) < target_per_stratum:
            raise RuntimeError(
                f"{stratum} has {len(candidates)} rows, needs "
                f"{target_per_stratum}"
            )
        candidates = candidates.sample(
            n=target_per_stratum,
            random_state=seed + index,
        )
        selected.append(candidates)
    output = pd.concat(selected, ignore_index=True)
    return output.sort_values(["stratum", "file_name"]).reset_index(drop=True)


def split_remaining_scenes(
    metadata: pd.DataFrame,
    *,
    test_scenes: set[str],
    validation_fraction: float,
    seed: int,
) -> dict[str, str]:
    remaining = metadata[
        ~metadata["source_scene_id"].astype(str).isin(test_scenes)
    ].copy()
    positive = remaining[remaining["stratum"].isin(STRATA)]
    signatures: dict[str, str] = {}
    for scene_id, scene_rows in positive.groupby("source_scene_id"):
        counts = scene_rows["stratum"].value_counts()
        signatures[str(scene_id)] = str(
            sorted(counts.index, key=lambda value: (-counts[value], value))[0]
        )
    for scene_id in remaining["source_scene_id"].astype(str).unique():
        signatures.setdefault(str(scene_id), "negative")

    by_signature: dict[str, list[str]] = defaultdict(list)
    for scene_id, signature in signatures.items():
        by_signature[signature].append(scene_id)

    validation_scenes: set[str] = set()
    for offset, signature in enumerate(sorted(by_signature)):
        scenes = sorted(by_signature[signature])
        random.Random(seed + 10_000 + offset).shuffle(scenes)
        count = max(1, round(len(scenes) * validation_fraction))
        if count >= len(scenes) and len(scenes) > 1:
            count = len(scenes) - 1
        validation_scenes.update(scenes[:count])

    assignments = {
        scene_id: (
            "test"
            if scene_id in test_scenes
            else "validation"
            if scene_id in validation_scenes
            else "train"
        )
        for scene_id in metadata["source_scene_id"].astype(str).unique()
    }
    return assignments


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_split(
    *,
    split: str,
    rows: pd.DataFrame,
    output_root: Path,
    master_root: Path,
    records_by_file: dict[str, dict[str, Any]],
    categories: list[dict[str, Any]],
    area_threshold: float,
) -> dict[str, int]:
    split_root = output_root / split
    (split_root / "images").mkdir(parents=True, exist_ok=True)
    (split_root / "labels").mkdir(parents=True, exist_ok=True)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    next_annotation_id = 1
    for next_image_id, (_, row) in enumerate(
        rows.sort_values("file_name").iterrows(),
        start=1,
    ):
        file_name = str(row["file_name"])
        record = records_by_file[file_name]
        master_split = str(record["split"])
        source_split_root = master_root / master_split
        hardlink_or_copy(
            source_split_root / "images" / file_name,
            split_root / "images" / file_name,
        )
        image = dict(record["image"])
        image["id"] = next_image_id
        images.append(image)
        label_lines: list[str] = []
        for source_annotation in record["annotations"]:
            annotation = dict(source_annotation)
            annotation["id"] = next_annotation_id
            annotation["image_id"] = next_image_id
            annotations.append(annotation)
            label_lines.append(
                yolo_label_line(
                    annotation["bbox"],
                    image_width=int(image["width"]),
                    image_height=int(image["height"]),
                )
            )
            next_annotation_id += 1
        label_name = Path(file_name).with_suffix(".txt").name
        (split_root / "labels" / label_name).write_text(
            "\n".join(label_lines),
            encoding="utf-8",
        )

        metadata_row = {
            key: value
            for key, value in row.to_dict().items()
            if key != "_master_split"
        }
        metadata_row["image_id"] = next_image_id
        metadata_row["split"] = split
        metadata_row["area_threshold"] = area_threshold
        metadata_row["stratify_by"] = "mask_area"
        metadata_row["no_overlap_iou_max"] = 0.0
        metadata_row["overlap_iou_min"] = 0.001
        metadata_row["selected_eval"] = split == "test"
        metadata_row["source_image_id"] = int(record["image"]["id"])
        metadata_rows.append(metadata_row)

    coco = {
        "info": {
            "description": (
                "Teacher-reference-bias matched split for "
                f"target category {categories[0]['name']}"
            ),
            "reference_type": (
                annotations[0].get("reference_type")
                if annotations
                else None
            ),
            "bbox_source": (
                annotations[0].get("bbox_source")
                if annotations
                else None
            ),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    (split_root / "_annotations.coco.json").write_text(
        json.dumps(coco, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    pd.DataFrame(metadata_rows).to_csv(
        split_root / "metadata.csv",
        index=False,
    )
    return {"images": len(images), "annotations": len(annotations)}


def main() -> None:
    args = parse_args()
    protocol: MatchedStudyConfig = load_matched_study_config(args.protocol)
    dataset: DatasetStudyConfig = load_dataset_study_config(args.dataset)
    if protocol.study_id != STUDY_ROOT.name:
        raise ValueError(
            f"Protocol study_id {protocol.study_id} does not match "
            f"{STUDY_ROOT.name}"
        )
    if dataset.master_prepared_root is None:
        raise ValueError("master_prepared_root is required")
    if dataset.area_threshold is None:
        raise ValueError("area_threshold is required")

    output_root = dataset.prepared_root
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_root} already exists; use --force intentionally"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    metadata, records_by_file, categories = load_master(dataset)
    if dataset.profile_id == "isaid":
        metadata = canonicalize_isaid_annotations(
            metadata,
            records_by_file,
            dataset=dataset,
            min_instance_area=int(protocol.detector["min_instance_area"]),
        )
    metadata = add_strata(
        metadata,
        area_threshold=dataset.area_threshold,
        overlap_threshold=float(protocol.evaluation["overlap_threshold"]),
    )
    positive = metadata[metadata["stratum"].isin(STRATA)].copy()
    target_per_stratum = int(protocol.evaluation["max_per_stratum"])
    scene_pool, pool_coverage = select_test_scene_pool(
        positive,
        target_per_stratum=target_per_stratum,
        seed=protocol.split_seed,
    )
    test_rows = select_exact_test_rows(
        positive,
        scene_pool=scene_pool,
        target_per_stratum=target_per_stratum,
        seed=protocol.split_seed,
    )
    test_scenes = set(test_rows["source_scene_id"].astype(str))
    train_validation_fraction = (
        protocol.split_fractions["validation"]
        / (
            protocol.split_fractions["train"]
            + protocol.split_fractions["validation"]
        )
    )
    assignments = split_remaining_scenes(
        metadata,
        test_scenes=test_scenes,
        validation_fraction=train_validation_fraction,
        seed=protocol.split_seed,
    )
    train_rows = metadata[
        metadata["source_scene_id"].astype(str).map(assignments).eq("train")
    ].copy()
    validation_rows = metadata[
        metadata["source_scene_id"]
        .astype(str)
        .map(assignments)
        .eq("validation")
    ].copy()

    split_rows = {
        "train": train_rows,
        "validation": validation_rows,
        "test": test_rows,
    }
    summaries = {
        split: write_split(
            split=split,
            rows=rows,
            output_root=output_root,
            master_root=dataset.master_prepared_root,
            records_by_file=records_by_file,
            categories=categories,
            area_threshold=dataset.area_threshold,
        )
        for split, rows in split_rows.items()
    }
    write_yolo_data_yaml(
        output_root,
        train_split="train",
        val_split="validation",
        class_name=dataset.target_category,
    )

    scene_rows = []
    for scene_id in sorted(assignments):
        scene_metadata = metadata[
            metadata["source_scene_id"].astype(str) == scene_id
        ]
        scene_rows.append(
            {
                "source_scene_id": scene_id,
                "split": assignments[scene_id],
                "master_tile_count": len(scene_metadata),
                **{
                    f"{stratum}_count": int(
                        scene_metadata["stratum"].eq(stratum).sum()
                    )
                    for stratum in STRATA
                },
            }
        )
    pd.DataFrame(scene_rows).to_csv(
        output_root / "source_scene_split.csv",
        index=False,
    )
    (output_root / "source_scene_split.json").write_text(
        json.dumps(scene_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    selected_counts = {
        stratum: int(test_rows["stratum"].eq(stratum).sum())
        for stratum in STRATA
    }
    scenes_by_split = {
        split: {
            row["source_scene_id"]
            for row in scene_rows
            if row["split"] == split
        }
        for split in ("train", "validation", "test")
    }
    provenance = {
        "schema_version": 1,
        "status": "completed",
        "dataset_id": dataset.dataset_id,
        "master_prepared_root": str(dataset.master_prepared_root),
        "master_content_manifest": str(
            dataset.master_prepared_root / "content_manifest.json"
        ),
        "master_content_manifest_sha256": sha256_file(
            dataset.master_prepared_root / "content_manifest.json"
        ),
        "area_threshold": dataset.area_threshold,
        "target_per_stratum": target_per_stratum,
        "test_scene_pool_coverage": pool_coverage,
        "selected_test_counts": selected_counts,
        "selected_test_scene_count": len(test_scenes),
        "source_scene_overlap": {
            "train_validation": len(
                scenes_by_split["train"] & scenes_by_split["validation"]
            ),
            "train_test": len(
                scenes_by_split["train"] & scenes_by_split["test"]
            ),
            "validation_test": len(
                scenes_by_split["validation"] & scenes_by_split["test"]
            ),
        },
        "split_summaries": summaries,
    }
    (output_root / "master_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if selected_counts != {
        stratum: target_per_stratum for stratum in STRATA
    }:
        raise RuntimeError(f"Unexpected test counts: {selected_counts}")
    for split, summary in summaries.items():
        print(
            f"{split}: images={summary['images']} "
            f"annotations={summary['annotations']}"
        )
    print(f"test scenes={len(test_scenes)}")
    print(f"test strata={selected_counts}")
    print(output_root)


if __name__ == "__main__":
    main()
