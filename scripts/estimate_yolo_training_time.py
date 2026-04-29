from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam3_bbox_study.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate YOLO training time from the previous local YOLO run.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "semantic_drone_car_yolo26x.yaml")
    return parser.parse_args()


def count_prepared_train_images(config: dict) -> int | None:
    train_dir = (
        resolve_path(config["paths"]["prepared_dataset_dir"])
        / str(config["dataset"]["train_split"])
        / "images"
    )
    if not train_dir.exists():
        return None
    return sum(1 for path in train_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def estimate_train_images_from_raw(config: dict) -> int:
    if str(config["dataset"].get("name", "")).startswith("semantic_drone"):
        return estimate_semantic_drone_train_images(config)

    dataset_cfg = config["dataset"]
    raw_root = resolve_path(config["paths"]["raw_dataset_dir"])
    split = str(dataset_cfg["train_split"])
    tile_size = int(dataset_cfg["tile_size"])
    stride = int(dataset_cfg.get("stride", tile_size))
    negative_ratio = dataset_cfg.get("train_negative_ratio")
    target_names = {name.lower() for name in dataset_cfg["target_categories"]}

    annotation_path = raw_root / split / "Annotations" / f"iSAID_{split}.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories_by_id = {int(category["id"]): str(category["name"]).lower() for category in data["categories"]}
    image_paths = {
        path.name: path
        for path in (raw_root / split / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    }

    boxes_by_image_id: dict[int, list[list[float]]] = {int(image["id"]): [] for image in data["images"]}
    for annotation in data["annotations"]:
        category_name = str(annotation.get("category_name") or categories_by_id[int(annotation["category_id"])]).lower()
        if category_name not in target_names:
            continue
        boxes_by_image_id.setdefault(int(annotation["image_id"]), []).append(annotation["bbox"])

    positive_tiles = 0
    negative_tiles = 0
    for image in data["images"]:
        image_path = image_paths[str(image["file_name"])]
        with Image.open(image_path) as pil_image:
            width, height = pil_image.size
        source_boxes = boxes_by_image_id.get(int(image["id"]), [])
        for tile_y in range(0, height - tile_size + 1, stride):
            for tile_x in range(0, width - tile_size + 1, stride):
                has_target = False
                for x_min, y_min, box_width, box_height in source_boxes:
                    if (
                        x_min < tile_x + tile_size
                        and x_min + box_width > tile_x
                        and y_min < tile_y + tile_size
                        and y_min + box_height > tile_y
                    ):
                        has_target = True
                        break
                if has_target:
                    positive_tiles += 1
                else:
                    negative_tiles += 1

    if negative_ratio is None:
        return positive_tiles + negative_tiles
    return positive_tiles + min(negative_tiles, int(positive_tiles * float(negative_ratio)))


def tile_count(length: int, tile_size: int, stride: int, include_edge_tiles: bool) -> int:
    if length < tile_size:
        return 0
    starts = list(range(0, length - tile_size + 1, stride))
    last_start = length - tile_size
    if include_edge_tiles and starts[-1] != last_start:
        starts.append(last_start)
    return len(starts)


def find_semantic_drone_images_root(raw_root: Path, configured_image_dir: str | None) -> Path:
    if configured_image_dir:
        return raw_root / configured_image_dir
    for candidate in [
        "aerial_semantic_drone/images",
        "images",
        "training_set/images",
        "semantic_drone_dataset/original_images",
    ]:
        path = raw_root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Semantic Drone images directory not found under {raw_root}")


def estimate_semantic_drone_train_images(config: dict) -> int:
    dataset_cfg = config["dataset"]
    raw_root = resolve_path(config["paths"]["raw_dataset_dir"])
    tile_size = int(dataset_cfg["tile_size"])
    stride = int(dataset_cfg.get("stride", tile_size))
    include_edge_tiles = bool(dataset_cfg.get("include_edge_tiles", True))
    val_fraction = float(dataset_cfg.get("val_fraction", 0.2))

    try:
        images_root = find_semantic_drone_images_root(raw_root, dataset_cfg.get("image_dir"))
    except FileNotFoundError:
        return estimate_semantic_drone_train_images_from_reference(config, tile_size, stride, include_edge_tiles, val_fraction)

    image_paths = sorted(
        path for path in images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )
    if not image_paths:
        return estimate_semantic_drone_train_images_from_reference(config, tile_size, stride, include_edge_tiles, val_fraction)

    train_source_count = max(1, int(len(image_paths) * (1.0 - val_fraction)))
    total_tiles = 0
    for image_path in image_paths[:train_source_count]:
        with Image.open(image_path) as image:
            width, height = image.size
        total_tiles += tile_count(width, tile_size, stride, include_edge_tiles) * tile_count(
            height,
            tile_size,
            stride,
            include_edge_tiles,
        )
    return total_tiles


def estimate_semantic_drone_train_images_from_reference(
    config: dict,
    tile_size: int,
    stride: int,
    include_edge_tiles: bool,
    val_fraction: float,
) -> int:
    reference_cfg = config["training_time_reference"]
    source_images = int(reference_cfg.get("expected_source_images", 400))
    width = int(reference_cfg.get("expected_source_width", 6000))
    height = int(reference_cfg.get("expected_source_height", 4000))
    train_source_count = max(1, int(source_images * (1.0 - val_fraction)))
    tiles_per_image = tile_count(width, tile_size, stride, include_edge_tiles) * tile_count(
        height,
        tile_size,
        stride,
        include_edge_tiles,
    )
    return train_source_count * tiles_per_image


def configured_gpu_count(config: dict) -> int:
    device = config["yolo"].get("device", 0)
    if isinstance(device, int):
        return 1
    device_value = str(device).strip().lower()
    if device_value in {"cpu", "mps"}:
        return 1
    return max(1, len([part for part in device_value.split(",") if part.strip()]))


def format_duration(hours: float) -> str:
    if hours < 1:
        return f"{hours * 60:.1f} minutes"
    if hours < 48:
        return f"{hours:.1f} hours"
    return f"{hours / 24:.1f} days"


def main() -> None:
    config = load_config(parse_args().config)
    reference_cfg = config["training_time_reference"]
    baseline_results_path = resolve_path(reference_cfg["baseline_results_csv"])
    if baseline_results_path.exists():
        baseline_results = pd.read_csv(baseline_results_path)
        baseline_epochs = int(baseline_results["epoch"].max())
        baseline_seconds_per_epoch = float(baseline_results["time"].iloc[-1]) / baseline_epochs
        baseline_source = str(baseline_results_path)
    else:
        baseline_seconds_per_epoch = float(reference_cfg.get("baseline_seconds_per_epoch", 66.1))
        baseline_source = f"fallback_default_missing:{baseline_results_path}"

    train_images = count_prepared_train_images(config)
    source = "prepared dataset"
    if train_images is None:
        train_images = estimate_train_images_from_raw(config)
        source = "raw/fallback estimate"

    image_count_scale = train_images / float(reference_cfg["baseline_train_images"])
    model_scale = float(reference_cfg["target_gflops"]) / float(reference_cfg["baseline_gflops"])
    size_scale = (float(config["yolo"]["imgsz"]) / float(reference_cfg["baseline_imgsz"])) ** 2
    batch_efficiency_penalty = 1.15 if int(config["yolo"]["batch"]) < 4 else 1.0
    gpu_count = configured_gpu_count(config)
    multi_gpu_efficiency = float(reference_cfg.get("multi_gpu_efficiency", 1.0))
    parallel_speedup = gpu_count * multi_gpu_efficiency if gpu_count > 1 else 1.0

    target_seconds_per_epoch = (
        baseline_seconds_per_epoch
        * image_count_scale
        * model_scale
        * size_scale
        * batch_efficiency_penalty
        / parallel_speedup
    )
    total_hours = target_seconds_per_epoch * int(config["yolo"]["epochs"]) / 3600.0
    early_stop_hours = target_seconds_per_epoch * max(1, int(config["yolo"]["patience"])) / 3600.0

    print(f"source: {source}")
    print(f"estimated_train_images: {train_images}")
    print(f"baseline_source: {baseline_source}")
    print(f"baseline_seconds_per_epoch: {baseline_seconds_per_epoch:.1f}")
    print(f"estimated_seconds_per_epoch: {target_seconds_per_epoch:.1f}")
    print(f"configured_epochs: {config['yolo']['epochs']}")
    print(f"configured_batch: {config['yolo']['batch']}")
    print(f"configured_device: {config['yolo']['device']}")
    print(f"estimated_gpu_count: {gpu_count}")
    print(f"assumed_parallel_speedup: {parallel_speedup:.2f}x")
    print(f"full_training_estimate: {format_duration(total_hours)}")
    print(f"patience_window_estimate: {format_duration(early_stop_hours)}")
    print(f"scale_factor: {target_seconds_per_epoch / baseline_seconds_per_epoch:.1f}x")
    print(
        "formula: "
        f"images({image_count_scale:.2f}) * model_gflops({model_scale:.2f}) "
        f"* imgsz({size_scale:.2f}) / parallel({parallel_speedup:.2f})"
    )


if __name__ == "__main__":
    main()
