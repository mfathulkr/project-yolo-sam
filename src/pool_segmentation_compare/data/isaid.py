from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import gdown
from PIL import Image


def download_isaid_folder(url: str, target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = gdown.download_folder(
        url=url,
        output=str(target_dir),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    return downloaded_files or []


def download_isaid_file(url: str, target_path: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download(url=url, output=str(target_path), quiet=False, fuzzy=True)
    if downloaded is None:
        raise RuntimeError(f"Failed to download file from {url}")
    return downloaded


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def find_split_assets(split_root: Path, split: str) -> tuple[Path, Path]:
    images_dir = resolve_rgb_images_dir(split_root)

    annotation_candidates = [
        f"instancesonly_filtered_{split}.json",
        f"instances_{split}.json",
        f"iSAID_{split}.json",
        "_annotations.coco.json",
    ]

    for file_name in annotation_candidates:
        matches = sorted(split_root.rglob(file_name))
        if matches:
            return images_dir, matches[0]

    fallback_jsons = sorted(split_root.rglob("*.json"))
    if len(fallback_jsons) == 1:
        return images_dir, fallback_jsons[0]

    raise FileNotFoundError(f"Could not locate annotation JSON for split {split} under {split_root}")


def resolve_rgb_images_dir(split_root: Path) -> Path:
    direct_images_dir = split_root / "images"
    if direct_images_dir.exists():
        extract_zip_archives(direct_images_dir)
        if has_rgb_images(direct_images_dir):
            return direct_images_dir

    candidate_dirs: list[Path] = []
    for candidate in sorted(split_root.rglob("images")):
        if not candidate.is_dir():
            continue
        if "Instance_masks" in candidate.parts or "Semantic_masks" in candidate.parts:
            continue
        extract_zip_archives(candidate)
        if has_rgb_images(candidate):
            candidate_dirs.append(candidate)

    if candidate_dirs:
        return candidate_dirs[0]

    part_files = sorted(split_root.rglob("*.part"))
    if part_files:
        raise FileNotFoundError(
            "RGB image archive download is incomplete. Found partial file(s): "
            + ", ".join(str(path) for path in part_files[:3])
        )

    raise FileNotFoundError(
        f"Could not locate extracted RGB images under {split_root}. "
        "Run the dataset download with DOTA RGB images enabled."
    )


def has_rgb_images(directory: Path) -> bool:
    for extension in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        for path in directory.rglob(extension):
            lower_name = path.name.lower()
            if lower_name.endswith("_instance_id_rgb.png") or lower_name.endswith("_instance_color_rgb.png"):
                continue
            return True
    return False


def extract_zip_archives(directory: Path) -> None:
    for archive_path in sorted(directory.glob("*.zip")):
        marker = archive_path.with_suffix(archive_path.suffix + ".extracted")
        if marker.exists():
            continue

        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(directory)
        marker.write_text("ok", encoding="utf-8")


def normalize_category_name(name: str) -> str:
    normalized = name.strip().casefold().replace("_", " ").replace("-", " ")
    return " ".join(normalized.split())


def find_category_id(coco_data: dict[str, Any], category_name: str) -> int:
    target = normalize_category_name(category_name)
    for category in coco_data.get("categories", []):
        category_value = normalize_category_name(str(category.get("name", "")))
        if category_value == target:
            return int(category["id"])
    available = ", ".join(str(category.get("name", "")) for category in coco_data.get("categories", []))
    raise KeyError(f"Category not found: {category_name}. Available categories: {available}")


def filter_coco_to_single_class(
    coco_data: dict[str, Any],
    category_name: str,
    keep_all_images: bool,
) -> dict[str, Any]:
    target_category_id = find_category_id(coco_data, category_name)
    annotations = [ann for ann in coco_data.get("annotations", []) if int(ann["category_id"]) == target_category_id]
    positive_image_ids = {int(ann["image_id"]) for ann in annotations}

    if keep_all_images:
        images = list(coco_data.get("images", []))
    else:
        images = [image for image in coco_data.get("images", []) if int(image["id"]) in positive_image_ids]

    kept_image_ids = {int(image["id"]) for image in images}
    filtered_annotations = []
    for ann in annotations:
        if int(ann["image_id"]) not in kept_image_ids:
            continue
        updated = dict(ann)
        updated["category_id"] = 1
        filtered_annotations.append(updated)

    category_record = {"id": 1, "name": category_name, "supercategory": "pool"}
    return {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "images": images,
        "annotations": filtered_annotations,
        "categories": [category_record],
    }


def resolve_source_image(images_dir: Path, file_name: str) -> Path:
    direct = images_dir / file_name
    if direct.exists():
        return direct

    direct_basename = images_dir / Path(file_name).name
    if direct_basename.exists():
        return direct_basename

    matches = list(images_dir.rglob(Path(file_name).name))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not locate source image for {file_name} under {images_dir}")


def link_or_copy_file(source: Path, target: Path, method: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return

    if method == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass

    shutil.copy2(source, target)


def yolo_label_line(bbox: list[float], image_width: int, image_height: int) -> str:
    x_min, y_min, box_width, box_height = bbox
    x_center = (x_min + box_width / 2.0) / image_width
    y_center = (y_min + box_height / 2.0) / image_height
    width = box_width / image_width
    height = box_height / image_height
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def resolve_image_dimensions(image_record: dict[str, Any], source_image: Path) -> tuple[int, int]:
    width = image_record.get("width")
    height = image_record.get("height")
    if width is not None and height is not None:
        return int(width), int(height)

    with Image.open(source_image) as image_handle:
        image_width, image_height = image_handle.size
    return int(image_width), int(image_height)


def prepare_single_class_split(
    split: str,
    raw_split_root: Path,
    output_split_root: Path,
    category_name: str,
    keep_all_images: bool,
    link_method: str,
) -> dict[str, int]:
    images_dir, annotation_path = find_split_assets(raw_split_root, split)
    coco_data = load_json(annotation_path)
    filtered = filter_coco_to_single_class(coco_data, category_name, keep_all_images)

    split_images_dir = output_split_root / "images"
    split_labels_dir = output_split_root / "labels"
    split_images_dir.mkdir(parents=True, exist_ok=True)
    split_labels_dir.mkdir(parents=True, exist_ok=True)

    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in filtered["annotations"]:
        annotations_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    positive_images = 0
    written_images = 0

    for image in filtered["images"]:
        image_id = int(image["id"])
        relative_name = Path(str(image["file_name"]))
        source_image = resolve_source_image(images_dir, str(relative_name))
        image_width, image_height = resolve_image_dimensions(image, source_image)
        destination_image = split_images_dir / relative_name
        link_or_copy_file(source_image, destination_image, link_method)

        image_annotations = annotations_by_image.get(image_id, [])
        if image_annotations:
            positive_images += 1

        label_path = split_labels_dir / relative_name.with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            yolo_label_line(ann["bbox"], image_width, image_height)
            for ann in image_annotations
        ]
        label_path.write_text("\n".join(lines), encoding="utf-8")
        written_images += 1

    save_json(filtered, output_split_root / "_annotations.coco.json")
    return {
        "images": written_images,
        "positive_images": positive_images,
        "annotations": len(filtered["annotations"]),
    }


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
