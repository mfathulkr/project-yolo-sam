from __future__ import annotations

import json
import os
import pickle
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from tqdm import tqdm

from yolo_sam.data.contracts import BBoxSource, ReferenceType
from yolo_sam.data.isaid import (
    PreparedSplitSummary,
    create_balanced_eval_split,
    summarize_box_overlap,
    write_split_metadata,
    write_yolo_data_yaml,
    yolo_label_line,
)
from yolo_sam.data.profiles import SAMRS_SOTA_PROFILE, normalize_category_name
from yolo_sam.data.provenance import audit_samrs_pickle_dataset, source_scene_id
from yolo_sam.io_utils import IMAGE_EXTENSIONS


SOTA_CATEGORIES = SAMRS_SOTA_PROFILE.categories


@dataclass(frozen=True)
class SamrsLayout:
    root: Path
    images_dir: Path
    masks_dir: Path
    train_list: Path | None
    val_list: Path | None
    train_json: Path | None
    val_json: Path | None


def target_category_ids(category_names: set[str]) -> set[int]:
    normalized_targets = {normalize_category_name(name) for name in category_names}
    ids = {idx for idx, name in enumerate(SOTA_CATEGORIES) if normalize_category_name(name) in normalized_targets}
    if not ids:
        raise ValueError(f"Target categories not found in SOTA mapping: {sorted(category_names)}")
    return ids


def resolve_samrs_layout(raw_root: Path, mask_subdir: str) -> SamrsLayout:
    root = raw_root
    candidates = [
        root,
        root / "trainval",
        root / "SOTA",
        root / "sota",
        root / "dotav2_1024",
        root / "dotav2_1024_rbb",
        root / "dotav2_1024_rbb" / "trainval",
    ]

    for candidate in candidates:
        images_dir = candidate / "images"
        masks_dir = candidate / mask_subdir
        train_json = first_existing(
            [
                candidate / "sota_rbb_train_ins_segmentation.json",
                candidate / "sota_train_ins_segmentation.json",
                candidate / "train_ins_segmentation.json",
                candidate / "instances_train.json",
            ]
        )
        val_json = first_existing(
            [
                candidate / "sota_rbb_valid_ins_segmentation.json",
                candidate / "sota_valid_ins_segmentation.json",
                candidate / "valid_ins_segmentation.json",
                candidate / "instances_valid.json",
            ]
        )
        if images_dir.exists() and (masks_dir.exists() or train_json is not None or val_json is not None):
            train_list = candidate / "train.txt"
            val_list = candidate / "valid.txt"
            return SamrsLayout(
                root=candidate,
                images_dir=images_dir,
                masks_dir=masks_dir,
                train_list=train_list if train_list.exists() else None,
                val_list=val_list if val_list.exists() else None,
                train_json=train_json,
                val_json=val_json,
            )

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not locate SAMRS SOTA layout under {raw_root}. "
        f"Expected images/ plus {mask_subdir}/ or SAMRS COCO json files in one of: {searched}"
    )


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def split_json_path(layout: SamrsLayout, split: str) -> Path | None:
    return layout.train_json if split == "train" else layout.val_json


def read_split_names(layout: SamrsLayout, split: str, sampling_seed: int) -> list[str]:
    split_path = layout.train_list if split == "train" else layout.val_list
    if split_path is not None:
        return [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if not layout.masks_dir.exists():
        json_path = split_json_path(layout, split)
        if json_path is not None:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            return sorted(Path(image["file_name"]).stem for image in data.get("images", []))

    stems = sorted(path.stem for path in layout.masks_dir.glob("*.pkl"))
    if not stems:
        raise FileNotFoundError(f"No SAMRS mask pkl files found in {layout.masks_dir}")
    rng = random.Random(sampling_seed)
    rng.shuffle(stems)
    cut = int(len(stems) * 0.8)
    return sorted(stems[:cut] if split == "train" else stems[cut:])


def find_image_path(images_dir: Path, stem: str) -> Path:
    for extension in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        path = images_dir / f"{stem}{extension}"
        if path.exists():
            return path
    matches = [path for path in images_dir.glob(f"{stem}.*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing SAMRS image for stem {stem} under {images_dir}")


def jsonable_rle(rle: dict[str, Any]) -> dict[str, Any]:
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(rle["size"][0]), int(rle["size"][1])], "counts": counts}


def horizontal_bbox_from_rhbox(values: Any) -> list[float]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    if flattened.size != 4:
        raise ValueError(f"Expected RHBox [x_min, y_min, x_max, y_max], got shape {flattened.shape}")
    x_min, y_min, x_max, y_max = (float(value) for value in flattened)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid RHBox coordinates: {flattened.tolist()}")
    return [x_min, y_min, width, height]


def clip_bbox_to_image(
    bbox: list[float],
    *,
    image_width: int,
    image_height: int,
) -> list[float]:
    x, y, width, height = (float(value) for value in bbox)
    x1 = max(0.0, min(x, float(image_width)))
    y1 = max(0.0, min(y, float(image_height)))
    x2 = max(0.0, min(x + width, float(image_width)))
    y2 = max(0.0, min(y + height, float(image_height)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Original detection bbox does not intersect the image: {bbox}"
        )
    return [x1, y1, x2 - x1, y2 - y1]


def annotation_rle(annotation: dict[str, Any], height: int, width: int) -> dict[str, Any]:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, dict):
        return jsonable_rle(segmentation)
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        merged = mask_utils.merge(rles)
        return jsonable_rle(merged)
    raise ValueError(f"Unsupported SAMRS segmentation format: {type(segmentation).__name__}")


def load_target_instances(mask_path: Path, target_ids: set[int], min_instance_area: int) -> list[dict[str, Any]]:
    with mask_path.open("rb") as handle:
        instances = pickle.load(handle)

    selected: list[dict[str, Any]] = []
    for instance in instances:
        label = int(instance["label"])
        if label not in target_ids:
            continue

        rle = instance["mask"]
        area = int(mask_utils.area(rle))
        if area < min_instance_area:
            continue

        if "rhbox" not in instance:
            raise ValueError(
                f"Instance in {mask_path} has no original RHBox; mask-derived boxes are forbidden"
            )
        source_bbox = horizontal_bbox_from_rhbox(instance["rhbox"])
        rle_height, rle_width = (int(value) for value in rle["size"])
        bbox = clip_bbox_to_image(
            source_bbox,
            image_width=rle_width,
            image_height=rle_height,
        )
        selected.append(
            {
                "rle": jsonable_rle(rle),
                "bbox": bbox,
                "source_bbox": source_bbox,
                "bbox_was_clipped": not np.allclose(
                    np.asarray(bbox),
                    np.asarray(source_bbox),
                    rtol=0.0,
                    atol=1e-6,
                ),
                "area": area,
                "source_label": label,
                "bbox_source": BBoxSource.ORIGINAL_DETECTION_ANNOTATION.value,
                "reference_type": ReferenceType.PSEUDO_SAM1.value,
            }
        )
    return selected


def load_target_instances_from_coco(
    annotations: list[dict[str, Any]],
    target_ids: set[int],
    min_instance_area: int,
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for annotation in annotations:
        label = int(annotation["category_id"])
        if label not in target_ids:
            continue

        rle = annotation_rle(annotation, height=height, width=width)
        area = int(annotation.get("area") or mask_utils.area(rle))
        if area < min_instance_area:
            continue

        if not annotation.get("bbox"):
            raise ValueError(
                "SAMRS COCO annotation has no original bbox; mask-derived boxes are forbidden"
            )
        bbox = [float(value) for value in annotation["bbox"]]
        selected.append(
            {
                "rle": rle,
                "bbox": bbox,
                "area": area,
                "source_label": label,
                "bbox_source": BBoxSource.ORIGINAL_DETECTION_ANNOTATION.value,
                "reference_type": ReferenceType.PSEUDO_SAM1.value,
            }
        )
    return selected


def scan_samrs_pkl_rows(
    layout: SamrsLayout,
    split_name: str,
    target_ids: set[int],
    min_instance_area: int,
    sampling_seed: int,
    selected_stems: set[str] | None = None,
) -> tuple[list[tuple[str, Path, list[dict[str, Any]]]], list[tuple[str, Path, list[dict[str, Any]]]]]:
    stems = (
        sorted(selected_stems)
        if selected_stems is not None
        else read_split_names(layout, split_name, sampling_seed=sampling_seed)
    )
    positive_rows: list[tuple[str, Path, list[dict[str, Any]]]] = []
    negative_rows: list[tuple[str, Path, list[dict[str, Any]]]] = []

    for stem in tqdm(stems, desc=f"Scanning SAMRS SOTA {split_name}"):
        mask_path = layout.masks_dir / f"{stem}.pkl"
        if not mask_path.exists():
            continue
        image_path = find_image_path(layout.images_dir, stem)
        instances = load_target_instances(mask_path, target_ids=target_ids, min_instance_area=min_instance_area)
        row = (stem, image_path, instances)
        if instances:
            positive_rows.append(row)
        else:
            negative_rows.append(row)
    return positive_rows, negative_rows


def scan_samrs_coco_rows(
    layout: SamrsLayout,
    split_name: str,
    target_ids: set[int],
    min_instance_area: int,
) -> tuple[list[tuple[str, Path, list[dict[str, Any]]]], list[tuple[str, Path, list[dict[str, Any]]]]]:
    json_path = split_json_path(layout, split_name)
    if json_path is None:
        raise FileNotFoundError(f"No SAMRS pkl masks or COCO json found for split {split_name}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in data.get("annotations", []):
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    positive_rows: list[tuple[str, Path, list[dict[str, Any]]]] = []
    negative_rows: list[tuple[str, Path, list[dict[str, Any]]]] = []
    for image in tqdm(data.get("images", []), desc=f"Scanning SAMRS SOTA {split_name} COCO"):
        file_name = str(image["file_name"])
        stem = Path(file_name).stem
        image_path = layout.images_dir / file_name
        if not image_path.exists():
            image_path = find_image_path(layout.images_dir, stem)
        instances = load_target_instances_from_coco(
            annotations_by_image.get(int(image["id"]), []),
            target_ids=target_ids,
            min_instance_area=min_instance_area,
            height=int(image["height"]),
            width=int(image["width"]),
        )
        row = (stem, image_path, instances)
        if instances:
            positive_rows.append(row)
        else:
            negative_rows.append(row)
    return positive_rows, negative_rows


def bbox_fill_ratio(area: int, bbox: list[float]) -> float:
    box_area = max(0.0, float(bbox[2]) * float(bbox[3]))
    return float(area / box_area) if box_area > 0 else 0.0


def copy_image_as_jpg_or_png(source: Path, destination: Path, image_format: str) -> tuple[int, int]:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read SAMRS image: {source}")
    height, width = image.shape[:2]
    destination.parent.mkdir(parents=True, exist_ok=True)
    ext = image_format.lower().lstrip(".")
    source_ext = source.suffix.lower().lstrip(".")
    if source_ext == ext or ({source_ext, ext} <= {"jpg", "jpeg"}):
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return width, height
    if ext in {"jpg", "jpeg"}:
        cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ext == "png":
        cv2.imwrite(str(destination), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        raise ValueError(f"Unsupported output image format: {image_format}")
    return width, height


def convert_samrs_split(
    layout: SamrsLayout,
    output_split_root: Path,
    split_name: str,
    target_ids: set[int],
    merged_category_name: str,
    image_format: str,
    min_instance_area: int,
    negative_ratio: float | None,
    sampling_seed: int,
    selected_stems: set[str] | None = None,
) -> PreparedSplitSummary:
    images_dir = output_split_root / "images"
    labels_dir = output_split_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(sampling_seed)
    if layout.masks_dir.exists():
        positive_rows, negative_rows = scan_samrs_pkl_rows(
            layout=layout,
            split_name=split_name,
            target_ids=target_ids,
            min_instance_area=min_instance_area,
            sampling_seed=sampling_seed,
            selected_stems=selected_stems,
        )
    else:
        positive_rows, negative_rows = scan_samrs_coco_rows(
            layout=layout,
            split_name=split_name,
            target_ids=target_ids,
            min_instance_area=min_instance_area,
        )

    kept_rows = list(positive_rows)
    if negative_ratio is not None and negative_ratio > 0:
        rng.shuffle(negative_rows)
        kept_rows.extend(negative_rows[: int(len(positive_rows) * negative_ratio)])
    kept_rows.sort(key=lambda item: item[0])

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, object]] = []
    next_image_id = 1
    next_annotation_id = 1
    image_ext = image_format.lower().lstrip(".")

    for stem, image_path, instances in tqdm(kept_rows, desc=f"Writing SAMRS SOTA {split_name}"):
        output_name = f"{stem}.{image_ext}"
        width, height = copy_image_as_jpg_or_png(image_path, images_dir / output_name, image_format=image_format)

        yolo_lines: list[str] = []
        boxes: list[list[float]] = []
        mask_area_pixels = 0
        fill_ratios: list[float] = []

        for instance in instances:
            bbox = instance["bbox"]
            area = int(instance["area"])
            boxes.append(bbox)
            mask_area_pixels += area
            fill_ratios.append(bbox_fill_ratio(area, bbox))
            yolo_lines.append(yolo_label_line(bbox, image_width=width, image_height=height))
            coco_annotations.append(
                {
                    "id": next_annotation_id,
                    "image_id": next_image_id,
                    "category_id": 1,
                    "segmentation": instance["rle"],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                    "source_label": int(instance["source_label"]),
                    "source_bbox": instance.get("source_bbox", bbox),
                    "bbox_was_clipped": bool(instance.get("bbox_was_clipped", False)),
                    "bbox_source": str(instance["bbox_source"]),
                    "reference_type": str(instance["reference_type"]),
                }
            )
            next_annotation_id += 1

        (labels_dir / f"{stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
        coco_images.append({"id": next_image_id, "file_name": output_name, "width": width, "height": height})
        max_pair_iou, overlap_pairs = summarize_box_overlap(boxes)
        metadata_rows.append(
            {
                "image_id": next_image_id,
                "file_name": output_name,
                "source_file_name": image_path.name,
                "source_stem": stem,
                "source_scene_id": source_scene_id(stem),
                "num_objects": len(instances),
                "mask_area_pixels": mask_area_pixels,
                "mask_area_ratio": mask_area_pixels / float(width * height),
                "mean_bbox_fill_ratio": float(np.mean(fill_ratios)) if fill_ratios else 0.0,
                "median_bbox_fill_ratio": float(np.median(fill_ratios)) if fill_ratios else 0.0,
                "max_pair_bbox_iou": max_pair_iou,
                "num_bbox_overlap_pairs": overlap_pairs,
                "has_bbox_overlap": bool(overlap_pairs > 0),
            }
        )
        next_image_id += 1

    categories = [{"id": 1, "name": merged_category_name, "supercategory": "object"}]
    (output_split_root / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "info": {
                    "description": f"SAMRS SOTA {merged_category_name} subset",
                    "reference_type": ReferenceType.PSEUDO_SAM1.value,
                    "bbox_source": BBoxSource.ORIGINAL_DETECTION_ANNOTATION.value,
                },
                "licenses": [],
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": categories,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_split_metadata(metadata_rows, output_split_root)

    return PreparedSplitSummary(
        images=len(coco_images),
        positive_images=len(positive_rows),
        negative_images=len(coco_images) - len(positive_rows),
        annotations=len(coco_annotations),
    )


def prepare_samrs_sota_dataset(
    raw_root: Path,
    output_root: Path,
    target_category_names: set[str],
    merged_category_name: str,
    train_split: str,
    val_split: str,
    eval_split: str,
    mask_subdir: str,
    image_format: str,
    min_instance_area: int,
    train_negative_ratio: float | None,
    sampling_seed: int,
    min_eval_objects: int,
    max_eval_objects: int | None,
    eval_overlap_iou_threshold: float,
    eval_no_overlap_iou_max: float | None,
    eval_overlap_iou_min: float | None,
    eval_area_threshold: float | str,
    eval_stratify_by: str,
    eval_low_object_count_max: int | None,
    eval_high_object_count_min: int | None,
    eval_max_per_stratum: int | None,
    eval_balance_to_smallest_stratum: bool,
) -> dict[str, PreparedSplitSummary]:
    layout = resolve_samrs_layout(raw_root=raw_root, mask_subdir=mask_subdir)
    target_ids = target_category_ids(target_category_names)
    if layout.masks_dir.exists():
        if len(target_ids) != 1:
            raise ValueError(
                "The matched SAMRS study currently requires exactly one target category"
            )
        audit_report = audit_samrs_pickle_dataset(
            root=raw_root,
            profile=SAMRS_SOTA_PROFILE,
            target_category=next(iter(target_category_names)),
            declared_target_id=next(iter(target_ids)),
            allow_raw_scene_overlap=True,
        )
        if not audit_report.passed:
            codes = ", ".join(finding.code for finding in audit_report.findings)
            raise ValueError(
                "SAMRS provenance audit failed before dataset preparation. "
                f"Blocking findings: {codes}"
            )

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, PreparedSplitSummary] = {}
    summaries[train_split] = convert_samrs_split(
        layout=layout,
        output_split_root=output_root / train_split,
        split_name="train",
        target_ids=target_ids,
        merged_category_name=merged_category_name,
        image_format=image_format,
        min_instance_area=min_instance_area,
        negative_ratio=train_negative_ratio,
        sampling_seed=sampling_seed,
    )
    summaries[val_split] = convert_samrs_split(
        layout=layout,
        output_split_root=output_root / val_split,
        split_name="valid",
        target_ids=target_ids,
        merged_category_name=merged_category_name,
        image_format=image_format,
        min_instance_area=min_instance_area,
        negative_ratio=None,
        sampling_seed=sampling_seed,
    )
    summaries[eval_split] = create_balanced_eval_split(
        dataset_root=output_root,
        source_split=val_split,
        eval_split=eval_split,
        min_objects=min_eval_objects,
        max_objects=max_eval_objects,
        overlap_iou_threshold=eval_overlap_iou_threshold,
        no_overlap_iou_max=eval_no_overlap_iou_max,
        overlap_iou_min=eval_overlap_iou_min,
        area_threshold=eval_area_threshold,
        stratify_by=eval_stratify_by,
        low_object_count_max=eval_low_object_count_max,
        high_object_count_min=eval_high_object_count_min,
        max_per_stratum=eval_max_per_stratum,
        balance_to_smallest_stratum=eval_balance_to_smallest_stratum,
        sampling_seed=sampling_seed,
    )

    write_yolo_data_yaml(output_root, train_split=train_split, val_split=val_split, class_name=merged_category_name)
    return summaries
