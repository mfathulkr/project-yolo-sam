from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm

from yolo_sam.data.contracts import BBoxSource, ReferenceType
from yolo_sam.data.provenance import source_scene_id


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class PreparedSplitSummary:
    images: int
    positive_images: int
    negative_images: int
    annotations: int


def normalize_category_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def find_image_files(images_root: Path) -> dict[str, Path]:
    return {
        path.name: path
        for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def write_yolo_data_yaml(dataset_root: Path, train_split: str, val_split: str, class_name: str) -> None:
    content = "\n".join(
        [
            f"path: {dataset_root.as_posix()}",
            f"train: {train_split}/images",
            f"val: {val_split}/images",
            "names:",
            f"  0: {class_name}",
            "",
        ]
    )
    (dataset_root / "data.yaml").write_text(content, encoding="utf-8")


def yolo_label_line(bbox: list[float], image_width: int, image_height: int) -> str:
    x_min, y_min, box_width, box_height = bbox
    x_center = (x_min + box_width / 2.0) / image_width
    y_center = (y_min + box_height / 2.0) / image_height
    width = box_width / image_width
    height = box_height / image_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def tile_starts(length: int, tile_size: int, stride: int, include_edge_tiles: bool) -> list[int]:
    if length < tile_size:
        return []

    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if include_edge_tiles and starts[-1] != last_start:
        starts.append(last_start)
    return starts


def bbox_intersects_tile(bbox: list[float], tile_x: int, tile_y: int, tile_size: int) -> bool:
    x, y, width, height = bbox
    return (
        x < tile_x + tile_size
        and x + width > tile_x
        and y < tile_y + tile_size
        and y + height > tile_y
    )


def clip_bbox_to_tile(
    bbox: list[float],
    tile_x: int,
    tile_y: int,
    tile_size: int,
) -> list[float] | None:
    x, y, width, height = (float(value) for value in bbox)
    x1 = max(x, float(tile_x))
    y1 = max(y, float(tile_y))
    x2 = min(x + width, float(tile_x + tile_size))
    y2 = min(y + height, float(tile_y + tile_size))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1 - tile_x, y1 - tile_y, x2 - x1, y2 - y1]


def bbox_iou_xywh(left: list[float], right: list[float]) -> float:
    left_x1, left_y1, left_w, left_h = left
    right_x1, right_y1, right_w, right_h = right
    left_x2 = left_x1 + left_w
    left_y2 = left_y1 + left_h
    right_x2 = right_x1 + right_w
    right_y2 = right_y1 + right_h

    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    left_area = left_w * left_h
    right_area = right_w * right_h
    return float(intersection / (left_area + right_area - intersection))


def summarize_box_overlap(boxes: list[list[float]]) -> tuple[float, int]:
    if len(boxes) < 2:
        return 0.0, 0

    max_iou = 0.0
    overlap_pairs = 0
    active: list[list[float]] = []
    for box in sorted(boxes, key=lambda item: item[0]):
        x_min = box[0]
        active = [candidate for candidate in active if candidate[0] + candidate[2] > x_min]
        for candidate in active:
            if candidate[1] + candidate[3] <= box[1] or box[1] + box[3] <= candidate[1]:
                continue
            iou = bbox_iou_xywh(candidate, box)
            if iou > 0.0:
                overlap_pairs += 1
                max_iou = max(max_iou, iou)
        active.append(box)

    return max_iou, overlap_pairs


def encode_coco_rle(mask: np.ndarray) -> dict[str, object]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "size": [int(value) for value in encoded["size"]],
        "counts": str(counts),
    }


def rasterize_clipped_annotation(
    annotation: dict[str, Any],
    tile_x: int,
    tile_y: int,
    tile_size: int,
    min_instance_area: int,
) -> tuple[dict[str, object], list[float], int] | None:
    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)

    for polygon in annotation.get("segmentation", []):
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            continue
        points[:, 0] -= tile_x
        points[:, 1] -= tile_y
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], color=1)

    area = int(mask.sum())
    if area < min_instance_area:
        return None

    bbox = clip_bbox_to_tile(
        annotation["bbox"],
        tile_x=tile_x,
        tile_y=tile_y,
        tile_size=tile_size,
    )
    if bbox is None:
        return None

    return encode_coco_rle(mask), bbox, area


def group_annotations_by_image(
    annotations: list[dict[str, Any]],
    categories_by_id: dict[int, str],
    target_category_names: set[str],
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    normalized_targets = {normalize_category_name(name) for name in target_category_names}

    for annotation in annotations:
        category_name = str(annotation.get("category_name") or categories_by_id.get(int(annotation["category_id"]), ""))
        if normalize_category_name(category_name) not in normalized_targets:
            continue
        grouped.setdefault(int(annotation["image_id"]), []).append(annotation)

    return grouped


def write_split_metadata(rows: list[dict[str, object]], split_root: Path) -> pd.DataFrame:
    metadata = pd.DataFrame(rows)
    if metadata.empty:
        metadata = pd.DataFrame(
            columns=[
                "image_id",
                "file_name",
                "source_file_name",
                "tile_x",
                "tile_y",
                "tile_size",
                "num_objects",
                "mask_area_pixels",
                "mask_area_ratio",
                "max_pair_bbox_iou",
                "num_bbox_overlap_pairs",
                "has_bbox_overlap",
            ]
        )
    metadata.to_csv(split_root / "metadata.csv", index=False)
    return metadata


def convert_isaid_split_to_tiles(
    raw_split_root: Path,
    output_split_root: Path,
    split_name: str,
    target_category_names: set[str],
    merged_category_name: str,
    tile_size: int,
    stride: int,
    include_edge_tiles: bool,
    image_format: str,
    min_instance_area: int,
    negative_ratio: float | None,
    sampling_seed: int,
    selected_source_names: set[str] | None = None,
) -> PreparedSplitSummary:
    annotation_path = raw_split_root / "Annotations" / f"iSAID_{split_name}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing iSAID annotation file: {annotation_path}")

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories_by_id = {int(category["id"]): str(category["name"]) for category in data["categories"]}
    annotations_by_image = group_annotations_by_image(
        annotations=data["annotations"],
        categories_by_id=categories_by_id,
        target_category_names=target_category_names,
    )
    images_by_name = find_image_files(raw_split_root / "images")

    images_dir = output_split_root / "images"
    labels_dir = output_split_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image_extension = image_format.lower().lstrip(".")
    if image_extension not in {"jpg", "jpeg", "png"}:
        raise ValueError(f"Unsupported image format: {image_format}")

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, object]] = []
    next_image_id = 1
    next_annotation_id = 1

    source_images = [
        source_image
        for source_image in data["images"]
        if selected_source_names is None
        or str(source_image["file_name"]) in selected_source_names
    ]
    for source_image in tqdm(source_images, desc=f"Preparing iSAID {split_name}"):
        source_name = str(source_image["file_name"])
        image_path = images_by_name.get(source_name)
        if image_path is None:
            raise FileNotFoundError(f"Missing iSAID image {source_name} under {raw_split_root / 'images'}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        source_annotations = annotations_by_image.get(int(source_image["id"]), [])

        tile_index = 0
        for tile_y in tile_starts(height, tile_size, stride, include_edge_tiles):
            for tile_x in tile_starts(width, tile_size, stride, include_edge_tiles):
                tile_annotations: list[dict[str, Any]] = []
                yolo_lines: list[str] = []
                tile_boxes: list[list[float]] = []
                mask_area_pixels = 0

                for annotation in source_annotations:
                    if not bbox_intersects_tile(annotation["bbox"], tile_x, tile_y, tile_size):
                        continue
                    clipped = rasterize_clipped_annotation(
                        annotation=annotation,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        tile_size=tile_size,
                        min_instance_area=min_instance_area,
                    )
                    if clipped is None:
                        continue

                    segmentation, bbox, area = clipped
                    tile_annotations.append(
                        {
                            "id": next_annotation_id,
                            "image_id": next_image_id,
                            "category_id": 1,
                            "segmentation": segmentation,
                            "area": area,
                            "bbox": bbox,
                            "iscrowd": int(annotation.get("iscrowd", 0)),
                            "source_annotation_id": int(annotation["id"]),
                            "bbox_source": BBoxSource.HUMAN_ANNOTATION.value,
                            "reference_type": ReferenceType.HUMAN.value,
                        }
                    )
                    next_annotation_id += 1
                    mask_area_pixels += area
                    tile_boxes.append(bbox)
                    yolo_lines.append(yolo_label_line(bbox, image_width=tile_size, image_height=tile_size))

                output_stem = f"{Path(source_name).stem}_{tile_index:04d}"
                output_image_name = f"{output_stem}.{image_extension}"
                tile_index += 1

                tile_image = image[tile_y : tile_y + tile_size, tile_x : tile_x + tile_size]
                if image_extension in {"jpg", "jpeg"}:
                    cv2.imwrite(str(images_dir / output_image_name), tile_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    cv2.imwrite(str(images_dir / output_image_name), tile_image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                (labels_dir / f"{output_stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")

                coco_images.append(
                    {
                        "id": next_image_id,
                        "file_name": output_image_name,
                        "width": tile_size,
                        "height": tile_size,
                    }
                )
                coco_annotations.extend(tile_annotations)

                max_pair_iou, overlap_pairs = summarize_box_overlap(tile_boxes)
                metadata_rows.append(
                    {
                        "image_id": next_image_id,
                        "file_name": output_image_name,
                        "source_file_name": source_name,
                        "source_scene_id": source_scene_id(Path(source_name).stem),
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "tile_size": tile_size,
                        "num_objects": len(tile_annotations),
                        "mask_area_pixels": mask_area_pixels,
                        "mask_area_ratio": mask_area_pixels / float(tile_size * tile_size),
                        "max_pair_bbox_iou": max_pair_iou,
                        "num_bbox_overlap_pairs": overlap_pairs,
                        "has_bbox_overlap": bool(overlap_pairs > 0),
                    }
                )
                next_image_id += 1

    kept_image_ids = set(image["id"] for image in coco_images)
    if negative_ratio is not None:
        if negative_ratio < 0:
            raise ValueError(f"negative_ratio must be non-negative, got {negative_ratio}")
        positive_ids = {int(row["image_id"]) for row in metadata_rows if int(row["num_objects"]) > 0}
        negative_ids = [int(row["image_id"]) for row in metadata_rows if int(row["num_objects"]) == 0]
        rng = random.Random(sampling_seed)
        rng.shuffle(negative_ids)
        keep_negative_count = min(len(negative_ids), int(len(positive_ids) * negative_ratio))
        kept_image_ids = positive_ids | set(negative_ids[:keep_negative_count])

        for image in coco_images:
            if int(image["id"]) in kept_image_ids:
                continue
            image_path = images_dir / str(image["file_name"])
            label_path = labels_dir / Path(str(image["file_name"])).with_suffix(".txt").name
            image_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)

        coco_images = [image for image in coco_images if int(image["id"]) in kept_image_ids]
        coco_annotations = [ann for ann in coco_annotations if int(ann["image_id"]) in kept_image_ids]
        metadata_rows = [row for row in metadata_rows if int(row["image_id"]) in kept_image_ids]

    categories = [{"id": 1, "name": merged_category_name, "supercategory": "object"}]
    save_json(
        {
            "info": {
                "description": f"iSAID tiled {merged_category_name} subset",
                "reference_type": ReferenceType.HUMAN.value,
                "bbox_source": BBoxSource.HUMAN_ANNOTATION.value,
            },
            "licenses": [],
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": categories,
        },
        output_split_root / "_annotations.coco.json",
    )
    write_split_metadata(metadata_rows, output_split_root)

    positive_images = sum(1 for row in metadata_rows if int(row["num_objects"]) > 0)
    return PreparedSplitSummary(
        images=len(coco_images),
        positive_images=positive_images,
        negative_images=len(coco_images) - positive_images,
        annotations=len(coco_annotations),
    )


def copy_or_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def create_balanced_eval_split(
    dataset_root: Path,
    source_split: str,
    eval_split: str,
    min_objects: int,
    max_objects: int | None,
    overlap_iou_threshold: float,
    no_overlap_iou_max: float | None,
    overlap_iou_min: float | None,
    area_threshold: float | str,
    stratify_by: str,
    low_object_count_max: int | None,
    high_object_count_min: int | None,
    max_per_stratum: int | None,
    balance_to_smallest_stratum: bool,
    sampling_seed: int,
) -> PreparedSplitSummary:
    source_root = dataset_root / source_split
    output_root = dataset_root / eval_split
    if output_root.exists():
        shutil.rmtree(output_root)
    (output_root / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "labels").mkdir(parents=True, exist_ok=True)

    source_metadata = pd.read_csv(source_root / "metadata.csv")
    eligible = source_metadata[source_metadata["num_objects"] >= min_objects].copy()
    if max_objects is not None:
        eligible = eligible[eligible["num_objects"] <= max_objects].copy()
    eligible = eligible[eligible["mask_area_ratio"] > 0.0].copy()
    if eligible.empty:
        raise RuntimeError("No iSAID eval tiles satisfy the configured stratification filters.")

    if no_overlap_iou_max is not None or overlap_iou_min is not None:
        resolved_no_overlap_iou_max = (
            float(no_overlap_iou_max) if no_overlap_iou_max is not None else overlap_iou_threshold
        )
        resolved_overlap_iou_min = (
            float(overlap_iou_min) if overlap_iou_min is not None else overlap_iou_threshold
        )
        if resolved_no_overlap_iou_max >= resolved_overlap_iou_min:
            raise ValueError("no_overlap_iou_max must be less than overlap_iou_min")

        eligible["overlap_group"] = np.select(
            [
                eligible["max_pair_bbox_iou"] <= resolved_no_overlap_iou_max,
                eligible["max_pair_bbox_iou"] >= resolved_overlap_iou_min,
            ],
            ["no_overlap", "overlap"],
            default="ambiguous_overlap",
        )
        eligible = eligible[eligible["overlap_group"] != "ambiguous_overlap"].copy()
        if eligible.empty:
            raise RuntimeError("No iSAID eval tiles remain after overlap gap filtering.")
    else:
        resolved_no_overlap_iou_max = overlap_iou_threshold
        resolved_overlap_iou_min = overlap_iou_threshold
        eligible["overlap_group"] = np.where(
            eligible["max_pair_bbox_iou"] > overlap_iou_threshold,
            "overlap",
            "no_overlap",
        )
    resolved_area_threshold: float | None = None
    normalized_stratify_by = stratify_by.strip().lower()
    if normalized_stratify_by == "mask_area":
        if isinstance(area_threshold, str):
            if area_threshold != "median":
                raise ValueError("area_threshold must be a float or 'median'")
            resolved_area_threshold = float(eligible["mask_area_ratio"].median())
        else:
            resolved_area_threshold = float(area_threshold)

        eligible["area_group"] = np.where(
            eligible["mask_area_ratio"] >= resolved_area_threshold,
            "high_mask_area",
            "low_mask_area",
        )
        group_column = "area_group"
        group_values = {"high_mask_area", "low_mask_area"}
    elif normalized_stratify_by == "object_count":
        if low_object_count_max is None or high_object_count_min is None:
            raise ValueError("object_count stratification requires low_object_count_max and high_object_count_min")
        if low_object_count_max >= high_object_count_min:
            raise ValueError("low_object_count_max must be less than high_object_count_min")

        eligible["count_group"] = np.select(
            [
                eligible["num_objects"] <= low_object_count_max,
                eligible["num_objects"] >= high_object_count_min,
            ],
            ["low_object_count", "high_object_count"],
            default="middle_object_count",
        )
        eligible = eligible[eligible["count_group"] != "middle_object_count"].copy()
        if eligible.empty:
            raise RuntimeError("No iSAID eval tiles remain after object-count stratification filters.")
        group_column = "count_group"
        group_values = {"high_object_count", "low_object_count"}
    else:
        raise ValueError("stratify_by must be 'mask_area' or 'object_count'")

    eligible["stratum"] = eligible["overlap_group"] + "__" + eligible[group_column]

    rng = random.Random(sampling_seed)
    selected_indices: list[int] = []
    grouped_indices = {
        str(stratum): list(group.index)
        for stratum, group in eligible.groupby("stratum")
    }
    expected_strata = {
        f"overlap__{group_value}" for group_value in group_values
    } | {
        f"no_overlap__{group_value}" for group_value in group_values
    }
    missing_strata = expected_strata - set(grouped_indices)
    if missing_strata:
        counts = {stratum: len(indices) for stratum, indices in grouped_indices.items()}
        raise RuntimeError(f"Missing eval strata {sorted(missing_strata)}. Available counts: {counts}")

    target_count = max_per_stratum if max_per_stratum is not None and max_per_stratum > 0 else None
    if balance_to_smallest_stratum and grouped_indices:
        smallest_count = min(len(indices) for indices in grouped_indices.values())
        target_count = min(target_count, smallest_count) if target_count is not None else smallest_count

    for indices in grouped_indices.values():
        indices = list(indices)
        rng.shuffle(indices)
        if target_count is not None and target_count > 0:
            indices = indices[:target_count]
        selected_indices.extend(indices)

    selected = eligible.loc[selected_indices].sort_values(["stratum", "file_name"]).copy()
    selected["area_threshold"] = resolved_area_threshold
    selected["stratify_by"] = normalized_stratify_by
    selected["low_object_count_max"] = low_object_count_max
    selected["high_object_count_min"] = high_object_count_min
    selected["no_overlap_iou_max"] = resolved_no_overlap_iou_max
    selected["overlap_iou_min"] = resolved_overlap_iou_min
    selected["selected_eval"] = True

    source_coco = json.loads((source_root / "_annotations.coco.json").read_text(encoding="utf-8"))
    selected_source_ids = {int(row["image_id"]) for _, row in selected.iterrows()}

    source_images_by_id = {int(image["id"]): image for image in source_coco["images"]}
    annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for annotation in source_coco["annotations"]:
        annotations_by_image_id.setdefault(int(annotation["image_id"]), []).append(annotation)

    new_images: list[dict[str, Any]] = []
    new_annotations: list[dict[str, Any]] = []
    new_metadata_rows: list[dict[str, object]] = []
    next_image_id = 1
    next_annotation_id = 1

    for _, row in selected.iterrows():
        source_image_id = int(row["image_id"])
        source_image = source_images_by_id[source_image_id]
        file_name = str(source_image["file_name"])
        copy_or_replace(source_root / "images" / file_name, output_root / "images" / file_name)
        copy_or_replace(source_root / "labels" / Path(file_name).with_suffix(".txt").name, output_root / "labels" / Path(file_name).with_suffix(".txt").name)

        new_images.append(
            {
                "id": next_image_id,
                "file_name": file_name,
                "width": int(source_image["width"]),
                "height": int(source_image["height"]),
            }
        )
        for annotation in annotations_by_image_id.get(source_image_id, []):
            new_annotation = dict(annotation)
            new_annotation["id"] = next_annotation_id
            new_annotation["image_id"] = next_image_id
            new_annotations.append(new_annotation)
            next_annotation_id += 1

        metadata_row = row.to_dict()
        metadata_row["source_image_id"] = source_image_id
        metadata_row["image_id"] = next_image_id
        new_metadata_rows.append(metadata_row)
        next_image_id += 1

    save_json(
        {
            "info": {"description": "Balanced iSAID eval split for stratified SAM3 bbox study"},
            "licenses": source_coco.get("licenses", []),
            "images": new_images,
            "annotations": new_annotations,
            "categories": source_coco["categories"],
        },
        output_root / "_annotations.coco.json",
    )
    write_split_metadata(new_metadata_rows, output_root)

    return PreparedSplitSummary(
        images=len(new_images),
        positive_images=len(new_images),
        negative_images=0,
        annotations=len(new_annotations),
    )


def prepare_isaid_vehicle_dataset(
    raw_root: Path,
    output_root: Path,
    target_category_names: set[str],
    merged_category_name: str,
    train_split: str,
    val_split: str,
    eval_split: str,
    tile_size: int,
    stride: int,
    include_edge_tiles: bool,
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
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, PreparedSplitSummary] = {}
    summaries[train_split] = convert_isaid_split_to_tiles(
        raw_split_root=raw_root / train_split,
        output_split_root=output_root / train_split,
        split_name=train_split,
        target_category_names=target_category_names,
        merged_category_name=merged_category_name,
        tile_size=tile_size,
        stride=stride,
        include_edge_tiles=include_edge_tiles,
        image_format=image_format,
        min_instance_area=min_instance_area,
        negative_ratio=train_negative_ratio,
        sampling_seed=sampling_seed,
    )
    summaries[val_split] = convert_isaid_split_to_tiles(
        raw_split_root=raw_root / val_split,
        output_split_root=output_root / val_split,
        split_name=val_split,
        target_category_names=target_category_names,
        merged_category_name=merged_category_name,
        tile_size=tile_size,
        stride=stride,
        include_edge_tiles=include_edge_tiles,
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
