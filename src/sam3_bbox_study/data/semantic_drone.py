from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from sam3_bbox_study.data.isaid import (
    PreparedSplitSummary,
    create_balanced_eval_split,
    save_json,
    summarize_box_overlap,
    write_yolo_data_yaml,
    yolo_label_line,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".tif", ".tiff"}

DEFAULT_CLASS_TO_ID = {
    "unlabeled": 0,
    "paved-area": 1,
    "paved_area": 1,
    "dirt": 2,
    "grass": 3,
    "gravel": 4,
    "water": 5,
    "rocks": 6,
    "pool": 7,
    "vegetation": 8,
    "other_vegetation": 8,
    "roof": 9,
    "wall": 10,
    "window": 11,
    "door": 12,
    "fence": 13,
    "fence-pole": 14,
    "fence_pole": 14,
    "person": 15,
    "dog": 16,
    "car": 17,
    "bicycle": 18,
    "tree": 19,
    "bald-tree": 20,
    "bald_tree": 20,
    "ar-marker": 21,
    "ar_marker": 21,
    "obstacle": 22,
    "conflicting": 23,
}


def normalize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def find_first_existing_dir(root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of these directories under {root}: {candidates}")


def list_files_by_stem(directory: Path, extensions: set[str]) -> dict[str, Path]:
    return {
        path.stem: path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    }


def tile_starts(length: int, tile_size: int, stride: int, include_edge_tiles: bool) -> list[int]:
    if length < tile_size:
        return []

    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if include_edge_tiles and starts[-1] != last_start:
        starts.append(last_start)
    return starts


def load_class_to_id(raw_root: Path, configured_ids: dict[str, int] | None = None) -> dict[str, int]:
    if configured_ids:
        return {normalize_name(key): int(value) for key, value in configured_ids.items()}

    for path in raw_root.rglob("class_to_idx.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        return {normalize_name(str(key)): int(value) for key, value in data.items()}

    for path in raw_root.rglob("idx_to_class.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        return {normalize_name(str(value)): int(key) for key, value in data.items()}

    for path in list(raw_root.rglob("classes.csv")) + list(raw_root.rglob("class_dict.csv")):
        frame = pd.read_csv(path)
        columns = {normalize_name(column): column for column in frame.columns}
        name_column = columns.get("name") or columns.get("class") or columns.get("class_name")
        id_column = columns.get("id") or columns.get("idx") or columns.get("index")
        if name_column and id_column:
            return {
                normalize_name(str(row[name_column])): int(row[id_column])
                for _, row in frame.iterrows()
            }

    return DEFAULT_CLASS_TO_ID


def resolve_target_ids(
    target_categories: list[str],
    class_to_id: dict[str, int],
    explicit_target_ids: list[int] | None,
) -> set[int]:
    if explicit_target_ids:
        return {int(value) for value in explicit_target_ids}

    target_ids: set[int] = set()
    for category in target_categories:
        normalized = normalize_name(category)
        if normalized not in class_to_id:
            raise KeyError(f"Target class '{category}' was not found in Semantic Drone class mapping.")
        target_ids.add(int(class_to_id[normalized]))
    return target_ids


def normalize_target_rgb_colors(target_rgb_colors: list[list[int]] | None) -> list[tuple[int, int, int]]:
    if not target_rgb_colors:
        return []

    normalized: list[tuple[int, int, int]] = []
    for color in target_rgb_colors:
        if len(color) != 3:
            raise ValueError(f"RGB colors must have exactly 3 values, got {color}")
        normalized.append((int(color[0]), int(color[1]), int(color[2])))
    return normalized


def load_target_mask(
    mask_path: Path,
    target_ids: set[int],
    target_rgb_colors: list[tuple[int, int, int]],
) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")

    if mask.ndim == 2:
        return np.isin(mask, list(target_ids))

    if mask.ndim == 3 and mask.shape[2] == 1:
        return np.isin(mask[:, :, 0], list(target_ids))

    if mask.ndim == 3 and target_rgb_colors:
        if mask.shape[2] == 4:
            mask = mask[:, :, :3]
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        target_mask = np.zeros(mask_rgb.shape[:2], dtype=bool)
        for red, green, blue in target_rgb_colors:
            target_mask |= (
                (mask_rgb[:, :, 0] == red)
                & (mask_rgb[:, :, 1] == green)
                & (mask_rgb[:, :, 2] == blue)
            )
        return target_mask

    raise ValueError(
        f"{mask_path} appears to be an RGB color mask. Use indexed label masks "
        "or set dataset.target_rgb_colors in the config."
    )


def component_to_segmentation(component_mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(component_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        flattened = contour.reshape(-1, 2)
        if len(flattened) < 3:
            continue
        polygon = flattened.astype(float).flatten().tolist()
        if len(polygon) >= 6:
            polygons.append(polygon)
    return polygons


def annotations_from_binary_tile(
    binary_mask: np.ndarray,
    image_id: int,
    next_annotation_id: int,
    min_component_area: int,
) -> tuple[list[dict[str, Any]], list[str], list[list[float]], int, int]:
    annotations: list[dict[str, Any]] = []
    yolo_lines: list[str] = []
    boxes: list[list[float]] = []
    mask_area = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
    height, width = binary_mask.shape
    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        component_mask = labels == component_id
        segmentation = component_to_segmentation(component_mask)
        if not segmentation:
            continue

        bbox = [float(x), float(y), float(box_width), float(box_height)]
        annotations.append(
            {
                "id": next_annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": segmentation,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            }
        )
        yolo_lines.append(yolo_label_line(bbox, image_width=width, image_height=height))
        boxes.append(bbox)
        mask_area += area
        next_annotation_id += 1

    return annotations, yolo_lines, boxes, mask_area, next_annotation_id


def copy_subset_split(
    all_pairs: list[tuple[Path, Path]],
    output_split_root: Path,
    target_ids: set[int],
    target_rgb_colors: list[tuple[int, int, int]],
    tile_size: int,
    stride: int,
    include_edge_tiles: bool,
    image_format: str,
    min_component_area: int,
    negative_ratio: float | None,
    sampling_seed: int,
    split_name: str,
) -> PreparedSplitSummary:
    image_extension = image_format.lower().lstrip(".")
    if image_extension not in {"jpg", "jpeg", "png"}:
        raise ValueError(f"Unsupported image format: {image_format}")

    images_dir = output_split_root / "images"
    labels_dir = output_split_root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, object]] = []
    next_image_id = 1
    next_annotation_id = 1

    for image_path, mask_path in tqdm(all_pairs, desc=f"Preparing Semantic Drone {split_name}"):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        binary_mask = load_target_mask(mask_path, target_ids, target_rgb_colors)
        if binary_mask.shape != image.shape[:2]:
            binary_mask = cv2.resize(
                binary_mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        tile_index = 0
        for tile_y in tile_starts(image.shape[0], tile_size, stride, include_edge_tiles):
            for tile_x in tile_starts(image.shape[1], tile_size, stride, include_edge_tiles):
                tile_image = image[tile_y : tile_y + tile_size, tile_x : tile_x + tile_size]
                tile_mask = binary_mask[tile_y : tile_y + tile_size, tile_x : tile_x + tile_size]
                output_stem = f"{image_path.stem}_{tile_index:04d}"
                output_image_name = f"{output_stem}.{image_extension}"
                tile_index += 1

                annotations, yolo_lines, boxes, mask_area, next_annotation_id = annotations_from_binary_tile(
                    binary_mask=tile_mask,
                    image_id=next_image_id,
                    next_annotation_id=next_annotation_id,
                    min_component_area=min_component_area,
                )

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
                coco_annotations.extend(annotations)
                max_pair_iou, overlap_pairs = summarize_box_overlap(boxes)
                metadata_rows.append(
                    {
                        "image_id": next_image_id,
                        "file_name": output_image_name,
                        "source_file_name": image_path.name,
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "tile_size": tile_size,
                        "num_objects": len(annotations),
                        "mask_area_pixels": mask_area,
                        "mask_area_ratio": mask_area / float(tile_size * tile_size),
                        "max_pair_bbox_iou": max_pair_iou,
                        "num_bbox_overlap_pairs": overlap_pairs,
                        "has_bbox_overlap": bool(overlap_pairs > 0),
                    }
                )
                next_image_id += 1

    kept_image_ids = set(image["id"] for image in coco_images)
    if negative_ratio is not None:
        positive_ids = {int(row["image_id"]) for row in metadata_rows if int(row["num_objects"]) > 0}
        negative_ids = [int(row["image_id"]) for row in metadata_rows if int(row["num_objects"]) == 0]
        rng = random.Random(sampling_seed)
        rng.shuffle(negative_ids)
        keep_negative_count = min(len(negative_ids), int(len(positive_ids) * negative_ratio))
        kept_image_ids = positive_ids | set(negative_ids[:keep_negative_count])

        for image_record in coco_images:
            if int(image_record["id"]) in kept_image_ids:
                continue
            image_path = images_dir / str(image_record["file_name"])
            label_path = labels_dir / Path(str(image_record["file_name"])).with_suffix(".txt").name
            image_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)

        coco_images = [image for image in coco_images if int(image["id"]) in kept_image_ids]
        coco_annotations = [ann for ann in coco_annotations if int(ann["image_id"]) in kept_image_ids]
        metadata_rows = [row for row in metadata_rows if int(row["image_id"]) in kept_image_ids]

    save_json(
        {
            "info": {"description": "Semantic Drone Dataset car subset"},
            "licenses": [],
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "car", "supercategory": "vehicle"}],
        },
        output_split_root / "_annotations.coco.json",
    )
    pd.DataFrame(metadata_rows).to_csv(output_split_root / "metadata.csv", index=False)

    positive_images = sum(1 for row in metadata_rows if int(row["num_objects"]) > 0)
    return PreparedSplitSummary(
        images=len(coco_images),
        positive_images=positive_images,
        negative_images=len(coco_images) - positive_images,
        annotations=len(coco_annotations),
    )


def prepare_semantic_drone_dataset(
    raw_root: Path,
    output_root: Path,
    image_dir: str | None,
    mask_dir: str | None,
    target_categories: list[str],
    target_category_ids: list[int] | None,
    target_rgb_colors: list[list[int]] | None,
    tile_size: int,
    stride: int,
    include_edge_tiles: bool,
    image_format: str,
    min_component_area: int,
    val_fraction: float,
    train_split: str,
    val_split: str,
    eval_split: str,
    train_negative_ratio: float | None,
    sampling_seed: int,
    min_eval_objects: int,
    max_eval_objects: int | None,
    eval_overlap_iou_threshold: float,
    eval_area_threshold: float | str,
    eval_max_per_stratum: int | None,
    eval_balance_to_smallest_stratum: bool,
) -> dict[str, PreparedSplitSummary]:
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    images_root = raw_root / image_dir if image_dir else find_first_existing_dir(
        raw_root,
        [
            "aerial_semantic_drone/images",
            "images",
            "training_set/images",
            "semantic_drone_dataset/original_images",
        ],
    )
    masks_root = raw_root / mask_dir if mask_dir else find_first_existing_dir(
        raw_root,
        [
            "aerial_semantic_drone/labels/png",
            "labels/png",
            "training_set/gt/semantic/label_images",
            "training_set/gt/semantic/label_images_semantic",
            "semantic_drone_dataset/label_images_semantic",
        ],
    )

    images_by_stem = list_files_by_stem(images_root, IMAGE_EXTENSIONS)
    masks_by_stem = list_files_by_stem(masks_root, MASK_EXTENSIONS)
    pairs = sorted(
        (image_path, masks_by_stem[stem])
        for stem, image_path in images_by_stem.items()
        if stem in masks_by_stem
    )
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found under {images_root} and {masks_root}")

    class_to_id = load_class_to_id(raw_root)
    target_ids = resolve_target_ids(target_categories, class_to_id, target_category_ids)
    normalized_target_rgb_colors = normalize_target_rgb_colors(target_rgb_colors)

    rng = random.Random(sampling_seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_fraction))
    val_pairs = sorted(shuffled[:val_count])
    train_pairs = sorted(shuffled[val_count:])

    summaries: dict[str, PreparedSplitSummary] = {}
    summaries[train_split] = copy_subset_split(
        all_pairs=train_pairs,
        output_split_root=output_root / train_split,
        target_ids=target_ids,
        target_rgb_colors=normalized_target_rgb_colors,
        tile_size=tile_size,
        stride=stride,
        include_edge_tiles=include_edge_tiles,
        image_format=image_format,
        min_component_area=min_component_area,
        negative_ratio=train_negative_ratio,
        sampling_seed=sampling_seed,
        split_name=train_split,
    )
    summaries[val_split] = copy_subset_split(
        all_pairs=val_pairs,
        output_split_root=output_root / val_split,
        target_ids=target_ids,
        target_rgb_colors=normalized_target_rgb_colors,
        tile_size=tile_size,
        stride=stride,
        include_edge_tiles=include_edge_tiles,
        image_format=image_format,
        min_component_area=min_component_area,
        negative_ratio=None,
        sampling_seed=sampling_seed,
        split_name=val_split,
    )
    summaries[eval_split] = create_balanced_eval_split(
        dataset_root=output_root,
        source_split=val_split,
        eval_split=eval_split,
        min_objects=min_eval_objects,
        max_objects=max_eval_objects,
        overlap_iou_threshold=eval_overlap_iou_threshold,
        area_threshold=eval_area_threshold,
        max_per_stratum=eval_max_per_stratum,
        balance_to_smallest_stratum=eval_balance_to_smallest_stratum,
        sampling_seed=sampling_seed,
    )

    write_yolo_data_yaml(output_root, train_split=train_split, val_split=val_split, class_name="car")
    return summaries
