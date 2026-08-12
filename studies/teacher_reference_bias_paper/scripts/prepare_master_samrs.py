from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.data.isaid import (
    create_balanced_eval_split,
    summarize_box_overlap,
    write_yolo_data_yaml,
)
from yolo_sam.data.profiles import SAMRS_SOTA_PROFILE
from yolo_sam.data.prepared_validation import build_prepared_content_manifest
from yolo_sam.data.provenance import (
    audit_samrs_pickle_dataset,
    source_scene_id,
)
from yolo_sam.data.samrs import (
    convert_samrs_split,
    load_target_instances,
    resolve_samrs_layout,
    target_category_ids,
)
from yolo_sam.data.split import (
    SplitCandidate,
    grouped_stratified_split,
    validate_split_manifest,
)
from teacher_reference_bias.config import (
    load_dataset_study_config,
    load_matched_study_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a source-scene-safe SAMRS SOTA target-class master corpus."
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


def scan_candidates(
    mask_dir: Path,
    *,
    target_ids: set[int],
    min_instance_area: int,
    image_size: int,
    overlap_threshold: float,
) -> tuple[list[SplitCandidate], list[dict[str, object]]]:
    raw_rows: list[dict[str, object]] = []
    for mask_path in tqdm(sorted(mask_dir.glob("*.pkl")), desc="Scanning SAMRS split candidates"):
        instances = load_target_instances(
            mask_path,
            target_ids=target_ids,
            min_instance_area=min_instance_area,
        )
        boxes = [list(instance["bbox"]) for instance in instances]
        area = sum(int(instance["area"]) for instance in instances)
        max_pair_iou, overlap_pairs = summarize_box_overlap(boxes)
        stem = mask_path.stem
        raw_rows.append(
            {
                "image_id": stem,
                "source_scene_id": source_scene_id(stem),
                "instance_count": len(instances),
                "mask_area_ratio": area / float(image_size * image_size),
                "max_pair_bbox_iou": max_pair_iou,
                "overlap_pairs": overlap_pairs,
            }
        )

    positive_areas = [
        float(row["mask_area_ratio"])
        for row in raw_rows
        if int(row["instance_count"]) > 0
    ]
    if not positive_areas:
        raise RuntimeError("No target instances were found in SAMRS SOTA")
    area_threshold = float(np.median(positive_areas))

    candidates: list[SplitCandidate] = []
    manifest_rows: list[dict[str, object]] = []
    for row in raw_rows:
        count = int(row["instance_count"])
        if count == 0:
            stratum = "negative"
        else:
            overlap = (
                "overlap"
                if float(row["max_pair_bbox_iou"]) >= overlap_threshold
                else "no_overlap"
            )
            area = (
                "high_mask_area"
                if float(row["mask_area_ratio"]) >= area_threshold
                else "low_mask_area"
            )
            stratum = f"{overlap}__{area}"
        candidates.append(
            SplitCandidate(
                image_id=str(row["image_id"]),
                source_scene_id=str(row["source_scene_id"]),
                stratum=stratum,
                instance_count=max(count, 1),
            )
        )
        manifest_rows.append(
            {
                **row,
                "preliminary_stratum": stratum,
                "preliminary_area_threshold": area_threshold,
            }
        )
    return candidates, manifest_rows


def write_source_manifest(
    rows: list[dict[str, object]],
    assignments: dict[str, str],
    output_root: Path,
) -> None:
    resolved = [
        {
            **row,
            "split": assignments[str(row["image_id"])],
        }
        for row in rows
    ]
    json_path = output_root / "source_scene_split.json"
    csv_path = output_root / "source_scene_split.csv"
    json_path.write_text(
        json.dumps(resolved, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(resolved[0]))
        writer.writeheader()
        writer.writerows(resolved)


def main() -> None:
    args = parse_args()
    protocol = load_matched_study_config(args.protocol)
    dataset = load_dataset_study_config(args.dataset)
    if dataset.profile_id != SAMRS_SOTA_PROFILE.profile_id:
        raise ValueError("This preparer only supports the SAMRS SOTA profile")
    if dataset.version.strip().lower() == "unverified":
        raise ValueError("SAMRS dataset config is still marked unverified")

    target_ids = target_category_ids({dataset.target_category})
    if len(target_ids) != 1:
        raise ValueError("The matched SAMRS study requires exactly one target category")
    target_id = next(iter(target_ids))
    audit = audit_samrs_pickle_dataset(
        dataset.raw_root,
        SAMRS_SOTA_PROFILE,
        target_category=dataset.target_category,
        declared_target_id=target_id,
        allow_raw_scene_overlap=True,
    )
    if not audit.passed:
        codes = ", ".join(finding.code for finding in audit.findings)
        raise ValueError(f"SAMRS provenance audit failed: {codes}")

    layout = resolve_samrs_layout(dataset.raw_root, mask_subdir="rhbox_segs_init/ins")
    candidates, source_rows = scan_candidates(
        layout.masks_dir,
        target_ids=target_ids,
        min_instance_area=int(protocol.detector["min_instance_area"]),
        image_size=protocol.image_size,
        overlap_threshold=float(protocol.evaluation["overlap_threshold"]),
    )
    split_rows = grouped_stratified_split(
        candidates,
        split_fractions=protocol.split_fractions,
        seed=protocol.split_seed,
    )
    validate_split_manifest(split_rows)
    assignments = {row.image_id: row.split for row in split_rows}
    stems_by_split = {
        split: {
            row.image_id
            for row in split_rows
            if row.split == split
        }
        for split in protocol.split_fractions
    }

    output_root = dataset.prepared_root
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_root} already exists. Use --force after checking prior artifacts."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    common = {
        "layout": layout,
        "target_ids": target_ids,
        "merged_category_name": dataset.target_category,
        "image_format": "png",
        "min_instance_area": int(protocol.detector["min_instance_area"]),
        "sampling_seed": protocol.split_seed,
    }
    summaries = {}
    summaries["train"] = convert_samrs_split(
        output_split_root=output_root / "train",
        split_name="train",
        negative_ratio=float(protocol.detector["negative_ratio"]),
        selected_stems=stems_by_split["train"],
        **common,
    )
    summaries["validation"] = convert_samrs_split(
        output_split_root=output_root / "validation",
        split_name="valid",
        negative_ratio=float(protocol.detector["negative_ratio"]),
        selected_stems=stems_by_split["validation"],
        **common,
    )
    summaries["test_pool"] = convert_samrs_split(
        output_split_root=output_root / "test_pool",
        split_name="valid",
        negative_ratio=0.0,
        selected_stems=stems_by_split["test"],
        **common,
    )

    eval_per_stratum = (
        args.eval_per_stratum
        if args.eval_per_stratum is not None
        else int(protocol.evaluation["max_per_stratum"])
    )
    summaries["test"] = create_balanced_eval_split(
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
    write_source_manifest(source_rows, assignments, output_root)
    (output_root / "split_manifest.json").write_text(
        json.dumps([asdict(row) for row in split_rows], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
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

    for split, summary in summaries.items():
        print(
            f"{split}: images={summary.images}, positive={summary.positive_images}, "
            f"negative={summary.negative_images}, annotations={summary.annotations}"
        )
    print(output_root)


if __name__ == "__main__":
    main()
