from __future__ import annotations

import json
import random
import shutil
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from tqdm import tqdm


LANDCOVER_CLASSES = {
    0: "background",
    1: "building",
    2: "woodland",
    3: "water",
    4: "road",
}


def expected_archive_size_bytes(url: str, timeout_seconds: int = 60) -> int | None:
    response = requests.head(url, allow_redirects=True, timeout=timeout_seconds)
    response.raise_for_status()
    header_value = response.headers.get("Content-Length")
    return int(header_value) if header_value is not None else None


def download_landcover_archive(
    url: str,
    archive_path: Path,
    expected_size_bytes: int | None = None,
    timeout_seconds: int = 60,
    chunk_size: int = 1024 * 1024,
) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if archive_path.exists():
        if expected_size_bytes is None or archive_path.stat().st_size == expected_size_bytes:
            return archive_path

    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    if partial_path.exists():
        partial_path.unlink()

    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) or expected_size_bytes or None
        with partial_path.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {archive_path.name}",
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))

    partial_size = partial_path.stat().st_size
    if expected_size_bytes is not None and partial_size != expected_size_bytes:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Downloaded archive size mismatch for {archive_path.name}: "
            f"expected {expected_size_bytes}, got {partial_size}"
        )

    partial_path.replace(archive_path)
    return archive_path


def required_raw_entries_present(raw_root: Path) -> bool:
    required = [
        raw_root / "images",
        raw_root / "masks",
        raw_root / "train.txt",
        raw_root / "val.txt",
        raw_root / "test.txt",
        raw_root / "split.py",
    ]
    return all(path.exists() for path in required)


def extract_landcover_archive(archive_path: Path, raw_root: Path) -> None:
    if required_raw_entries_present(raw_root):
        return

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(raw_root)

    if not required_raw_entries_present(raw_root):
        raise RuntimeError(f"Extraction completed but required files are missing under {raw_root}")


def read_split_ids(raw_root: Path, split: str) -> set[str]:
    split_file = raw_root / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")

    lines = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines()]
    return {line for line in lines if line}


def yolo_label_line(bbox: list[float], image_width: int, image_height: int) -> str:
    x_min, y_min, box_width, box_height = bbox
    x_center = (x_min + box_width / 2.0) / image_width
    y_center = (y_min + box_height / 2.0) / image_height
    width = box_width / image_width
    height = box_height / image_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


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


def component_annotations_from_mask(
    binary_mask: np.ndarray,
    image_id: int,
    next_annotation_id: int,
    min_component_area: int,
) -> tuple[list[dict[str, Any]], list[str], int]:
    annotations: list[dict[str, Any]] = []
    label_lines: list[str] = []

    component_mask = binary_mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(component_mask, connectivity=8)

    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_component_area:
            continue

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        if width <= 0 or height <= 0:
            continue

        single_component = labels == component_id
        segmentation = component_to_segmentation(single_component)
        if not segmentation:
            continue

        bbox = [float(x), float(y), float(width), float(height)]
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
        label_lines.append(yolo_label_line(bbox, image_width=binary_mask.shape[1], image_height=binary_mask.shape[0]))
        next_annotation_id += 1

    return annotations, label_lines, next_annotation_id


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


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def rebalance_train_negatives(
    split_root: Path,
    coco_images: list[dict[str, Any]],
    coco_annotations: list[dict[str, Any]],
    negative_ratio: float,
    sampling_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    if negative_ratio < 0:
        raise ValueError(f"negative_ratio must be non-negative, got {negative_ratio}")

    annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
    for ann in coco_annotations:
        annotations_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

    positive_images = [image for image in coco_images if int(image["id"]) in annotations_by_image_id]
    negative_images = [image for image in coco_images if int(image["id"]) not in annotations_by_image_id]

    max_negatives_to_keep = min(len(negative_images), int(len(positive_images) * negative_ratio))
    rng = random.Random(sampling_seed)
    shuffled_negatives = list(negative_images)
    rng.shuffle(shuffled_negatives)
    kept_negatives = shuffled_negatives[:max_negatives_to_keep]

    kept_image_ids = {int(image["id"]) for image in positive_images}
    kept_image_ids.update(int(image["id"]) for image in kept_negatives)

    dropped_images = [image for image in coco_images if int(image["id"]) not in kept_image_ids]
    for image in dropped_images:
        file_name = Path(str(image["file_name"]))
        image_path = split_root / "images" / file_name
        label_path = split_root / "labels" / file_name.with_suffix(".txt")
        image_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)

    filtered_images = [image for image in coco_images if int(image["id"]) in kept_image_ids]
    filtered_annotations = [ann for ann in coco_annotations if int(ann["image_id"]) in kept_image_ids]

    summary = {
        "images": len(filtered_images),
        "positive_images": len(positive_images),
        "negative_images": len(kept_negatives),
        "annotations": len(filtered_annotations),
    }
    return filtered_images, filtered_annotations, summary


def prepare_landcover_single_class_dataset(
    raw_root: Path,
    output_root: Path,
    category_name: str,
    category_value: int,
    tile_size: int,
    image_format: str,
    min_component_area: int,
    train_split: str,
    val_split: str,
    test_split: str | None = None,
    train_negative_ratio: float | None = None,
    sampling_seed: int = 42,
) -> dict[str, dict[str, int]]:
    if category_value not in LANDCOVER_CLASSES:
        raise KeyError(f"Unsupported LandCover.ai class value: {category_value}")

    source_images_dir = raw_root / "images"
    source_masks_dir = raw_root / "masks"
    if not source_images_dir.exists() or not source_masks_dir.exists():
        raise FileNotFoundError("Expected extracted LandCover.ai images/ and masks/ under raw_root")

    split_names = [train_split, val_split]
    if test_split:
        split_names.append(test_split)

    split_ids = {split: read_split_ids(raw_root, split) for split in split_names}
    tile_to_split = {
        tile_id: split
        for split, ids in split_ids.items()
        for tile_id in ids
    }

    if len(tile_to_split) != sum(len(ids) for ids in split_ids.values()):
        raise RuntimeError("Duplicate tile ids found across split files")

    extension = image_format.lower().lstrip(".")
    if extension not in {"jpg", "jpeg", "png"}:
        raise ValueError(f"Unsupported image format: {image_format}")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    split_roots = {split: output_root / split for split in split_names}
    image_dirs = {split: split_roots[split] / "images" for split in split_names}
    label_dirs = {split: split_roots[split] / "labels" for split in split_names}
    for split in split_names:
        image_dirs[split].mkdir(parents=True, exist_ok=True)
        label_dirs[split].mkdir(parents=True, exist_ok=True)

    coco_images = {split: [] for split in split_names}
    coco_annotations = {split: [] for split in split_names}
    summary = {
        split: {"images": 0, "positive_images": 0, "annotations": 0}
        for split in split_names
    }

    next_image_id = 1
    next_annotation_id = 1

    for image_path in tqdm(sorted(source_images_dir.glob("*.tif")), desc="Preparing LandCover.ai"):
        mask_path = source_masks_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for image: {image_path.name}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Failed to read {image_path.name} or its mask")
        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"Image/mask size mismatch for {image_path.name}")

        stem = image_path.stem
        tile_index = 0
        for y in range(0, image.shape[0], tile_size):
            for x in range(0, image.shape[1], tile_size):
                tile_image = image[y : y + tile_size, x : x + tile_size]
                tile_mask = mask[y : y + tile_size, x : x + tile_size]
                tile_id = f"{stem}_{tile_index}"
                tile_index += 1

                if tile_image.shape[0] != tile_size or tile_image.shape[1] != tile_size:
                    continue

                split = tile_to_split.get(tile_id)
                if split is None:
                    continue

                binary_mask = tile_mask == category_value
                annotations, label_lines, next_annotation_id = component_annotations_from_mask(
                    binary_mask=binary_mask,
                    image_id=next_image_id,
                    next_annotation_id=next_annotation_id,
                    min_component_area=min_component_area,
                )

                if extension in {"jpg", "jpeg"}:
                    output_image_name = f"{tile_id}.jpg"
                    write_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
                else:
                    output_image_name = f"{tile_id}.png"
                    write_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

                output_image_path = image_dirs[split] / output_image_name
                cv2.imwrite(str(output_image_path), tile_image, write_params)

                label_path = label_dirs[split] / f"{tile_id}.txt"
                label_path.write_text("\n".join(label_lines), encoding="utf-8")

                coco_images[split].append(
                    {
                        "id": next_image_id,
                        "file_name": output_image_name,
                        "width": tile_size,
                        "height": tile_size,
                    }
                )
                coco_annotations[split].extend(annotations)

                summary[split]["images"] += 1
                summary[split]["annotations"] += len(annotations)
                if annotations:
                    summary[split]["positive_images"] += 1

                next_image_id += 1

    categories = [{"id": 1, "name": category_name, "supercategory": "structure"}]

    if train_negative_ratio is not None:
        filtered_images, filtered_annotations, rebalanced_summary = rebalance_train_negatives(
            split_root=split_roots[train_split],
            coco_images=coco_images[train_split],
            coco_annotations=coco_annotations[train_split],
            negative_ratio=train_negative_ratio,
            sampling_seed=sampling_seed,
        )
        coco_images[train_split] = filtered_images
        coco_annotations[train_split] = filtered_annotations
        summary[train_split] = {
            "images": rebalanced_summary["images"],
            "positive_images": rebalanced_summary["positive_images"],
            "negative_images": rebalanced_summary["negative_images"],
            "annotations": rebalanced_summary["annotations"],
        }

    for split in split_names:
        coco_record = {
            "info": {"description": "LandCover.ai single-class building subset"},
            "licenses": [],
            "images": coco_images[split],
            "annotations": coco_annotations[split],
            "categories": categories,
        }
        save_json(coco_record, split_roots[split] / "_annotations.coco.json")

        if "negative_images" not in summary[split]:
            summary[split]["negative_images"] = summary[split]["images"] - summary[split]["positive_images"]

    write_yolo_data_yaml(output_root, train_split=train_split, val_split=val_split, class_name=category_name)
    return summary
