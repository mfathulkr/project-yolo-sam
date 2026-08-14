from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.data.isaid import (
    convert_isaid_split_to_tiles,
    create_balanced_eval_split,
    write_yolo_data_yaml,
)
from yolo_sam.data.profiles import ISAID_PROFILE
from yolo_sam.data.prepared_validation import build_prepared_content_manifest
from yolo_sam.data.provenance import audit_isaid_coco_dataset
from teacher_reference_bias.config import (
    load_dataset_study_config,
    load_matched_study_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a source-scene-safe iSAID target-class master corpus."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=STUDY_ROOT / "configs" / "protocol.yaml",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Experiment-specific master_config.yaml path.",
    )
    parser.add_argument(
        "--eval-per-stratum",
        type=int,
        help="Intentional override; defaults to evaluation.max_per_stratum.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_train_scene_rows(raw_root: Path, target_category: str) -> list[dict[str, object]]:
    annotation_path = raw_root / "train" / "Annotations" / "iSAID_train.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {
        int(category["id"]): str(category["name"])
        for category in payload["categories"]
    }
    target_ids = {
        category_id
        for category_id, category_name in categories.items()
        if category_name.lower() == target_category.lower()
    }
    if len(target_ids) != 1:
        raise ValueError(f"Expected one train target ID for {target_category}, got {target_ids}")

    counts: dict[int, int] = {}
    for annotation in payload["annotations"]:
        if int(annotation["category_id"]) in target_ids:
            image_id = int(annotation["image_id"])
            counts[image_id] = counts.get(image_id, 0) + 1

    return [
        {
            "image_id": int(image["id"]),
            "file_name": str(image["file_name"]),
            "source_scene_id": Path(str(image["file_name"])).stem,
            "target_instances": counts.get(int(image["id"]), 0),
        }
        for image in payload["images"]
    ]


def split_train_validation_scenes(
    rows: list[dict[str, object]],
    validation_fraction: float,
    seed: int,
) -> tuple[set[str], set[str], list[dict[str, object]]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    positives = [row for row in rows if int(row["target_instances"]) > 0]
    negatives = [row for row in rows if int(row["target_instances"]) == 0]
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    validation_positive_count = max(1, round(len(positives) * validation_fraction))
    validation_negative_count = max(1, round(len(negatives) * validation_fraction))
    validation_rows = (
        positives[:validation_positive_count]
        + negatives[:validation_negative_count]
    )
    validation_names = {str(row["file_name"]) for row in validation_rows}
    train_names = {
        str(row["file_name"])
        for row in rows
        if str(row["file_name"]) not in validation_names
    }
    manifest_rows = [
        {
            **row,
            "split": "validation" if str(row["file_name"]) in validation_names else "train",
        }
        for row in rows
    ]
    return train_names, validation_names, manifest_rows


def write_source_manifest(rows: list[dict[str, object]], output_root: Path) -> None:
    json_path = output_root / "source_scene_split.json"
    csv_path = output_root / "source_scene_split.csv"
    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    protocol = load_matched_study_config(args.protocol)
    dataset = load_dataset_study_config(args.dataset)
    eval_per_stratum = (
        args.eval_per_stratum
        if args.eval_per_stratum is not None
        else int(protocol.evaluation["max_per_stratum"])
    )
    if dataset.profile_id != "isaid":
        raise ValueError("This preparer only supports the iSAID dataset profile")

    audit = audit_isaid_coco_dataset(
        dataset.raw_root,
        profile=ISAID_PROFILE,
        target_category=dataset.target_category,
    )
    if not audit.passed:
        codes = ", ".join(finding.code for finding in audit.findings)
        raise ValueError(f"iSAID provenance audit failed: {codes}")

    output_root = dataset.prepared_root
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_root} already exists. Use --force only after verifying the prior artifact."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    source_rows = load_train_scene_rows(dataset.raw_root, dataset.target_category)
    validation_fraction = (
        protocol.split_fractions["validation"]
        / (
            protocol.split_fractions["train"]
            + protocol.split_fractions["validation"]
        )
    )
    train_names, validation_names, source_manifest = split_train_validation_scenes(
        source_rows,
        validation_fraction=validation_fraction,
        seed=protocol.split_seed,
    )

    common = {
        "target_category_names": {dataset.target_category},
        "merged_category_name": dataset.target_category,
        "tile_size": protocol.image_size,
        "stride": protocol.image_size,
        "include_edge_tiles": True,
        "image_format": "png",
        "min_instance_area": int(protocol.detector["min_instance_area"]),
        "sampling_seed": protocol.split_seed,
    }
    train_summary = convert_isaid_split_to_tiles(
        raw_split_root=dataset.raw_root / "train",
        output_split_root=output_root / "train",
        split_name="train",
        negative_ratio=float(protocol.detector["negative_ratio"]),
        selected_source_names=train_names,
        **common,
    )
    validation_summary = convert_isaid_split_to_tiles(
        raw_split_root=dataset.raw_root / "train",
        output_split_root=output_root / "validation",
        split_name="train",
        negative_ratio=float(protocol.detector["negative_ratio"]),
        selected_source_names=validation_names,
        **common,
    )
    test_pool_summary = convert_isaid_split_to_tiles(
        raw_split_root=dataset.raw_root / "val",
        output_split_root=output_root / "test_pool",
        split_name="val",
        negative_ratio=0.0,
        selected_source_names=None,
        **common,
    )
    test_summary = create_balanced_eval_split(
        dataset_root=output_root,
        source_split="test_pool",
        eval_split="test",
        min_objects=1,
        max_objects=500,
        overlap_iou_threshold=float(protocol.evaluation["overlap_threshold"]),
        no_overlap_iou_max=0.0,
        overlap_iou_min=float(protocol.evaluation["overlap_threshold"]),
        area_threshold="median",
        stratify_by="mask_area",
        low_object_count_max=None,
        high_object_count_min=None,
        max_per_stratum=eval_per_stratum,
        balance_to_smallest_stratum=True,
        sampling_seed=protocol.split_seed,
    )

    test_source_names = {
        path.name
        for path in (dataset.raw_root / "val" / "images").rglob("*")
        if path.is_file()
    }
    source_manifest.extend(
        {
            "image_id": "",
            "file_name": name,
            "source_scene_id": Path(name).stem,
            "target_instances": "",
            "split": "test",
        }
        for name in sorted(test_source_names)
    )
    write_source_manifest(source_manifest, output_root)
    write_yolo_data_yaml(
        output_root,
        train_split="train",
        val_split="validation",
        class_name=dataset.target_category,
    )
    (output_root / "content_manifest.json").write_text(
        json.dumps(
            build_prepared_content_manifest(
                output_root,
                splits=("train", "validation", "test_pool", "test"),
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summaries = {
        "train": train_summary,
        "validation": validation_summary,
        "test_pool": test_pool_summary,
        "test": test_summary,
    }
    for split, summary in summaries.items():
        print(
            f"{split}: images={summary.images}, positive={summary.positive_images}, "
            f"negative={summary.negative_images}, annotations={summary.annotations}"
        )
    print(output_root)


if __name__ == "__main__":
    main()
