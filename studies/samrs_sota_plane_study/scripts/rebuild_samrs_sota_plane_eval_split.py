from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.config import load_config, resolve_path


SOURCE_SPLITS = ["train", "val"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a SAMRS SOTA plane eval split from train+val positives, then remove "
            "selected eval images from their source split to avoid leakage."
        )
    )
    parser.add_argument("--config", type=Path, default=STUDY_ROOT / "configs" / "yolo26x.yaml")
    parser.add_argument("--target-per-stratum", type=int, default=None)
    return parser.parse_args()


def load_source_metadata(dataset_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in SOURCE_SPLITS:
        metadata_path = dataset_root / split / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata: {metadata_path}")
        metadata = pd.read_csv(metadata_path)
        metadata["source_split"] = split
        metadata["source_image_id"] = metadata["image_id"].astype(int)
        frames.append(metadata)
    return pd.concat(frames, ignore_index=True)


def stratify_candidates(metadata: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, float]:
    evaluation = config["evaluation"]
    eligible = metadata[metadata["num_objects"] >= int(evaluation.get("min_objects_per_image", 1))].copy()
    max_objects = evaluation.get("max_objects_per_image")
    if max_objects is not None:
        eligible = eligible[eligible["num_objects"] <= int(max_objects)].copy()
    eligible = eligible[eligible["mask_area_ratio"] > 0.0].copy()
    if eligible.empty:
        raise RuntimeError("No positive SAMRS SOTA plane candidates for eval split.")

    no_overlap_iou_max = float(evaluation.get("no_overlap_iou_max", 0.0))
    overlap_iou_min = float(evaluation.get("overlap_iou_min", evaluation.get("overlap_iou_threshold", 0.0)))
    if no_overlap_iou_max >= overlap_iou_min:
        raise ValueError("no_overlap_iou_max must be less than overlap_iou_min")

    eligible["overlap_group"] = np.select(
        [
            eligible["max_pair_bbox_iou"] <= no_overlap_iou_max,
            eligible["max_pair_bbox_iou"] >= overlap_iou_min,
        ],
        ["no_overlap", "overlap"],
        default="ambiguous_overlap",
    )
    eligible = eligible[eligible["overlap_group"] != "ambiguous_overlap"].copy()

    area_threshold_cfg = evaluation.get("area_threshold", "median")
    if isinstance(area_threshold_cfg, str):
        if area_threshold_cfg != "median":
            raise ValueError("Only area_threshold='median' or a numeric value is supported.")
        area_threshold = float(eligible["mask_area_ratio"].median())
    else:
        area_threshold = float(area_threshold_cfg)

    eligible["area_group"] = np.where(
        eligible["mask_area_ratio"] >= area_threshold,
        "high_mask_area",
        "low_mask_area",
    )
    eligible["stratum"] = eligible["overlap_group"] + "__" + eligible["area_group"]
    eligible["area_threshold"] = area_threshold
    eligible["stratify_by"] = "mask_area"
    eligible["low_object_count_max"] = evaluation.get("low_object_count_max")
    eligible["high_object_count_min"] = evaluation.get("high_object_count_min")
    eligible["no_overlap_iou_max"] = no_overlap_iou_max
    eligible["overlap_iou_min"] = overlap_iou_min
    eligible["selected_eval"] = True
    return eligible, area_threshold


def choose_eval_rows(candidates: pd.DataFrame, target_per_stratum: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    expected_strata = {
        "no_overlap__low_mask_area",
        "no_overlap__high_mask_area",
        "overlap__low_mask_area",
        "overlap__high_mask_area",
    }
    selected_parts: list[pd.DataFrame] = []
    counts = candidates["stratum"].value_counts().to_dict()
    missing = sorted(stratum for stratum in expected_strata if counts.get(stratum, 0) < target_per_stratum)
    if missing:
        raise RuntimeError(
            f"Insufficient eval candidates for {missing}; target={target_per_stratum}, counts={counts}"
        )

    for stratum in sorted(expected_strata):
        stratum_rows = candidates[candidates["stratum"] == stratum].copy()
        chosen_indices: list[int] = []
        for split in SOURCE_SPLITS:
            split_indices = list(stratum_rows[stratum_rows["source_split"] == split].index)
            rng.shuffle(split_indices)
            needed = target_per_stratum - len(chosen_indices)
            if needed <= 0:
                break
            chosen_indices.extend(split_indices[:needed])
        if len(chosen_indices) != target_per_stratum:
            raise RuntimeError(f"Failed to choose {target_per_stratum} rows for {stratum}")
        selected_parts.append(candidates.loc[chosen_indices].copy())

    selected = pd.concat(selected_parts, ignore_index=True)
    return selected.sort_values(["stratum", "source_split", "file_name"]).reset_index(drop=True)


def read_coco(split_root: Path) -> dict[str, Any]:
    return json.loads((split_root / "_annotations.coco.json").read_text(encoding="utf-8"))


def write_coco(path: Path, images: list[dict[str, Any]], annotations: list[dict[str, Any]], categories: list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(
            {
                "info": {"description": "SAMRS SOTA plane split"},
                "licenses": [],
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def rebuild_source_split(dataset_root: Path, split: str, selected_file_names: set[str]) -> tuple[int, int]:
    split_root = dataset_root / split
    metadata = pd.read_csv(split_root / "metadata.csv")
    coco = read_coco(split_root)

    kept_metadata = metadata[~metadata["file_name"].isin(selected_file_names)].copy()
    old_to_new_image_id: dict[int, int] = {}
    kept_images: list[dict[str, Any]] = []
    next_image_id = 1
    for image in coco["images"]:
        if image["file_name"] in selected_file_names:
            continue
        old_id = int(image["id"])
        new_image = dict(image)
        new_image["id"] = next_image_id
        old_to_new_image_id[old_id] = next_image_id
        kept_images.append(new_image)
        next_image_id += 1

    kept_annotations: list[dict[str, Any]] = []
    next_annotation_id = 1
    for annotation in coco["annotations"]:
        old_image_id = int(annotation["image_id"])
        if old_image_id not in old_to_new_image_id:
            continue
        new_annotation = dict(annotation)
        new_annotation["id"] = next_annotation_id
        new_annotation["image_id"] = old_to_new_image_id[old_image_id]
        kept_annotations.append(new_annotation)
        next_annotation_id += 1

    kept_metadata["image_id"] = kept_metadata["image_id"].astype(int).map(old_to_new_image_id)
    if kept_metadata["image_id"].isna().any():
        raise RuntimeError(f"{split}: metadata/COCO mismatch while removing eval rows")
    kept_metadata["image_id"] = kept_metadata["image_id"].astype(int)
    kept_metadata = kept_metadata.sort_values(["file_name"]).reset_index(drop=True)
    kept_metadata.to_csv(split_root / "metadata.csv", index=False)
    write_coco(split_root / "_annotations.coco.json", kept_images, kept_annotations, coco["categories"])

    for file_name in selected_file_names:
        (split_root / "images" / file_name).unlink(missing_ok=True)
        (split_root / "labels" / f"{Path(file_name).stem}.txt").unlink(missing_ok=True)
    return len(kept_images), len(kept_annotations)


def build_eval_split(dataset_root: Path, selected: pd.DataFrame) -> tuple[int, int]:
    eval_root = dataset_root / "eval"
    if eval_root.exists():
        shutil.rmtree(eval_root)
    (eval_root / "images").mkdir(parents=True)
    (eval_root / "labels").mkdir(parents=True)

    cocos = {split: read_coco(dataset_root / split) for split in SOURCE_SPLITS}
    images_by_split_and_file = {
        split: {image["file_name"]: image for image in coco["images"]}
        for split, coco in cocos.items()
    }
    annotations_by_split_and_image_id: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for split, coco in cocos.items():
        for annotation in coco["annotations"]:
            annotations_by_split_and_image_id.setdefault((split, int(annotation["image_id"])), []).append(annotation)

    eval_images: list[dict[str, Any]] = []
    eval_annotations: list[dict[str, Any]] = []
    eval_metadata_rows: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1
    for _, row in selected.iterrows():
        split = str(row["source_split"])
        file_name = str(row["file_name"])
        source_root = dataset_root / split
        shutil.copy2(source_root / "images" / file_name, eval_root / "images" / file_name)
        shutil.copy2(source_root / "labels" / f"{Path(file_name).stem}.txt", eval_root / "labels" / f"{Path(file_name).stem}.txt")

        source_image = images_by_split_and_file[split][file_name]
        old_image_id = int(source_image["id"])
        eval_image = dict(source_image)
        eval_image["id"] = next_image_id
        eval_images.append(eval_image)

        for annotation in annotations_by_split_and_image_id.get((split, old_image_id), []):
            eval_annotation = dict(annotation)
            eval_annotation["id"] = next_annotation_id
            eval_annotation["image_id"] = next_image_id
            eval_annotations.append(eval_annotation)
            next_annotation_id += 1

        metadata_row = row.to_dict()
        metadata_row["image_id"] = next_image_id
        metadata_row["source_image_id"] = old_image_id
        eval_metadata_rows.append(metadata_row)
        next_image_id += 1

    categories = cocos[SOURCE_SPLITS[0]]["categories"]
    write_coco(eval_root / "_annotations.coco.json", eval_images, eval_annotations, categories)
    pd.DataFrame(eval_metadata_rows).to_csv(eval_root / "metadata.csv", index=False)
    return len(eval_images), len(eval_annotations)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset_root = resolve_path(config["paths"]["prepared_dataset_dir"])
    target_per_stratum = int(args.target_per_stratum or config["evaluation"].get("max_per_stratum", 128))
    seed = int(config["dataset"].get("sampling_seed", 42))

    metadata = load_source_metadata(dataset_root)
    candidates, area_threshold = stratify_candidates(metadata, config)
    selected = choose_eval_rows(candidates, target_per_stratum=target_per_stratum, seed=seed)

    eval_images, eval_annotations = build_eval_split(dataset_root, selected)
    source_summaries: dict[str, tuple[int, int]] = {}
    for split in SOURCE_SPLITS:
        selected_file_names = set(selected[selected["source_split"] == split]["file_name"].tolist())
        source_summaries[split] = rebuild_source_split(dataset_root, split, selected_file_names)

    print(f"Rebuilt SAMRS SOTA plane eval split with area_threshold={area_threshold:.10f}")
    print(f"eval: images={eval_images}, annotations={eval_annotations}")
    for split, (images, annotations) in source_summaries.items():
        print(f"{split}: images={images}, annotations={annotations}")
    print(selected["stratum"].value_counts().sort_index().to_string())
    print(selected.groupby(["stratum", "source_split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
