from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam3_bbox_study.config import load_config, resolve_path
from sam3_bbox_study.data.coco_boxes import load_ground_truth_boxes
from sam3_bbox_study.data.coco_masks import load_ground_truth_masks
from sam3_bbox_study.io_utils import load_binary_mask


CURATED_SAMPLES = [
    ("no_overlap__low_mask_area", "P2766_0016.jpg", "low total mask, separated vehicles, dense urban/industrial block"),
    ("no_overlap__high_mask_area", "P0199_0002.jpg", "many separated parked vehicles in urban campus block"),
    ("overlap__low_mask_area", "P2404_0002.jpg", "low total mask, overlapping vehicle boxes in compact urban road/building scene"),
    ("overlap__high_mask_area", "P2781_0005.jpg", "large total mask, very clear overlapping rows of parked vehicles"),
]

PIPELINES = [
    ("GT mask + GT boxes", "gt", "ground_truth", (0, 210, 75), "gt"),
    ("SAM3 text-only", "sam3_text", "sam3_text_output_dir", (40, 90, 255), None),
    ("SAM3 YOLO bbox", "yolo_sam3", "yolo_sam3_output_dir", (255, 190, 0), "yolo"),
    ("SAM3 GT bbox", "gt_box_sam3", "gt_box_sam3_output_dir", (255, 0, 220), "gt"),
    ("SAM3 hybrid YOLO bbox", "sam3_hybrid_yolo", "sam3_hybrid_yolo_output_dir", (210, 80, 255), "yolo"),
    ("RemoteSAM text-only", "remotesam_text", "remotesam_text_output_dir", (90, 190, 255), None),
    ("RingMo-SAM GT bbox", "gt_box_ringmo_sam", "gt_box_ringmo_sam_output_dir", (155, 90, 255), "gt"),
    ("RingMo-SAM YOLO bbox", "yolo_ringmo_sam", "yolo_ringmo_sam_output_dir", (255, 80, 155), "yolo"),
    ("SAM2 GT bbox", "gt_box_sam2", "gt_box_sam2_output_dir", (0, 155, 255), "gt"),
    ("SAM2 YOLO bbox", "yolo_sam2", "yolo_sam2_output_dir", (0, 190, 220), "yolo"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export hand-curated, high-visibility qualitative examples.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "isaid_vehicle_yolo26x_cpu_eval.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "isaid_vehicle_visualizations_sam3_triplet_curated",
    )
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def box_iou(left: list[float], right: list[float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def max_overlap_pair(boxes: list[list[float]]) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    best_iou = 0.0
    for i, left in enumerate(boxes):
        for j in range(i + 1, len(boxes)):
            value = box_iou(left, boxes[j])
            if value > best_iou:
                best_iou = value
                best = (i, j)
    return best if best_iou > 0 else None


def choose_crop(boxes: list[list[float]], stratum: str, image_size: tuple[int, int]) -> tuple[list[int], set[int]]:
    width, height = image_size
    highlight: set[int] = set()
    pair = max_overlap_pair(boxes) if stratum.startswith("overlap__") else None
    if pair is not None:
        highlight = set(pair)
        chosen = [boxes[pair[0]], boxes[pair[1]]]
        arr = np.asarray(chosen, dtype=float)
        x1, y1 = arr[:, 0].min(), arr[:, 1].min()
        x2, y2 = arr[:, 2].max(), arr[:, 3].max()
        side = max(260.0, max(x2 - x1, y2 - y1) * 5.0)
    elif boxes:
        centers = np.asarray([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes], dtype=float)
        radius = 180.0 if stratum.endswith("high_mask_area") else 140.0
        counts = [int(np.linalg.norm(centers - center, axis=1).min() == 0) for center in centers]
        counts = [int((np.linalg.norm(centers - center, axis=1) <= radius).sum()) for center in centers]
        best_index = int(np.argmax(counts))
        cx, cy = centers[best_index]
        side = 520.0 if stratum.endswith("high_mask_area") else 360.0
        x1, y1 = cx - side / 2, cy - side / 2
        x2, y2 = cx + side / 2, cy + side / 2
    else:
        x1, y1, x2, y2 = 0.0, 0.0, float(width), float(height)
        side = float(width)

    if pair is not None:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x1, y1, x2, y2 = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2

    x1 = max(0.0, min(x1, width - side))
    y1 = max(0.0, min(y1, height - side))
    x2 = min(float(width), x1 + side)
    y2 = min(float(height), y1 + side)
    x1 = max(0.0, x2 - side)
    y1 = max(0.0, y2 - side)
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))], highlight


def overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int = 92) -> Image.Image:
    rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    rgba[mask.astype(bool)] = [color[0], color[1], color[2], alpha]
    return Image.alpha_composite(image.convert("RGBA"), Image.fromarray(rgba, "RGBA"))


def draw_boxes(
    image: Image.Image,
    boxes: list[list[float]],
    crop: list[int] | None = None,
    highlight: set[int] | None = None,
    scale: float = 1.0,
    box_color: tuple[int, int, int, int] = (255, 35, 35, 255),
    highlight_color: tuple[int, int, int, int] = (255, 220, 0, 255),
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    x_offset = crop[0] if crop else 0
    y_offset = crop[1] if crop else 0
    highlight = highlight or set()
    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        coords = [
            (x1 - x_offset) * scale,
            (y1 - y_offset) * scale,
            (x2 - x_offset) * scale,
            (y2 - y_offset) * scale,
        ]
        color = highlight_color if index in highlight else box_color
        width = 7 if index in highlight else 4
        draw.rectangle(coords, outline=color, width=width)
    return image


def panel(
    base_image: Image.Image,
    mask: np.ndarray,
    title: str,
    color: tuple[int, int, int],
    size: tuple[int, int],
    boxes: list[list[float]] | None = None,
    crop: list[int] | None = None,
    highlight: set[int] | None = None,
    box_color: tuple[int, int, int, int] = (255, 35, 35, 255),
) -> Image.Image:
    if crop:
        image = base_image.crop(crop)
        panel_mask = mask[crop[1] : crop[3], crop[0] : crop[2]]
    else:
        image = base_image.copy()
        panel_mask = mask
    image = overlay_mask(image, panel_mask, color)
    if boxes:
        image = draw_boxes(image, boxes, crop=crop, highlight=highlight, box_color=box_color)
    image = image.convert("RGB").resize((size[0], size[1] - 46), Image.Resampling.BILINEAR)
    output = Image.new("RGB", size, "white")
    output.paste(image, (0, 46))
    draw = ImageDraw.Draw(output)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(30, 30, 30), width=2)
    draw.text((12, 10), title, fill=(0, 0, 0), font=font(25, bold=True))
    return output


def make_card(
    image_path: Path,
    gt_mask: np.ndarray,
    pred_masks: dict[str, np.ndarray],
    boxes: list[list[float]],
    boxes_by_key: dict[str, list[list[float]]],
    metadata_row: pd.Series,
    reason: str,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    crop, highlight = choose_crop(boxes, str(metadata_row["stratum"]), image.size)
    canvas = Image.new("RGB", (2820, 1800), "white")
    draw = ImageDraw.Draw(canvas)

    title = str(metadata_row["stratum"]).replace("__", " / ").replace("_", " ").upper()
    subtitle = (
        f"{metadata_row['file_name']} | objects={int(metadata_row['num_objects'])} | "
        f"mask area={float(metadata_row['mask_area_ratio']):.4f} | "
        f"max bbox IoU={float(metadata_row['max_pair_bbox_iou']):.3f}"
    )
    draw.text((38, 24), title, fill=(0, 0, 0), font=font(44, bold=True))
    draw.text((42, 80), subtitle, fill=(45, 45, 45), font=font(27))
    draw.text((42, 114), reason, fill=(80, 80, 80), font=font(23))

    full = panel(image, gt_mask, "FULL TILE: GT mask + GT boxes", (0, 210, 75), (720, 720), boxes=boxes)
    zoom = panel(
        image,
        gt_mask,
        "ZOOM: key bbox region",
        (0, 210, 75),
        (720, 720),
        boxes=boxes,
        crop=crop,
        highlight=highlight,
    )
    canvas.paste(full, (40, 165))
    canvas.paste(zoom, (805, 165))
    note = "yellow boxes = max-overlap GT pair" if highlight else "red boxes = GT boxes"
    draw.text((1565, 220), note, fill=(0, 0, 0), font=font(28, bold=True))
    draw.text((1565, 270), "cyan boxes = YOLO boxes on YOLO-bbox panels.", fill=(55, 55, 55), font=font(25))
    draw.text((1565, 318), "Large zoom makes small-object labels visible.", fill=(55, 55, 55), font=font(25))

    positions = [
        (40, 955),
        (590, 955),
        (1140, 955),
        (1690, 955),
        (2240, 955),
        (40, 1370),
        (590, 1370),
        (1140, 1370),
        (1690, 1370),
        (2240, 1370),
    ]
    for (x, y), (label, key, _, color, box_source) in zip(positions, PIPELINES, strict=True):
        mask = gt_mask if key == "gt" else pred_masks[key]
        boxes_to_draw = boxes_by_key.get(key)
        box_color = (255, 35, 35, 255) if box_source == "gt" else (0, 220, 255, 255)
        panel_highlight = highlight if box_source == "gt" else None
        comparison = panel(
            image,
            mask,
            label,
            color,
            (540, 390),
            boxes=boxes_to_draw,
            crop=crop,
            highlight=panel_highlight,
            box_color=box_color,
        )
        canvas.paste(comparison, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def load_prompt_boxes(output_dir: Path, stem: str) -> list[list[float]]:
    raw_path = output_dir / "raw" / f"{stem}.json"
    if not raw_path.exists():
        return []
    import json

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    boxes = payload.get("input_boxes", [])
    return boxes if isinstance(boxes, list) else []


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split = config["dataset"]["eval_split"]
    prepared_split_dir = resolve_path(config["paths"]["prepared_dataset_dir"]) / split
    image_dir = prepared_split_dir / "images"
    metadata = pd.read_csv(prepared_split_dir / "metadata.csv")
    gt_masks = load_ground_truth_masks(prepared_split_dir)
    boxes_by_name = load_ground_truth_boxes(prepared_split_dir)

    output_dir = args.output_dir
    hero_dir = output_dir / "hero_cards"
    hero_dir.mkdir(parents=True, exist_ok=True)
    for old_card in hero_dir.glob("*.png"):
        old_card.unlink()

    rows: list[dict[str, object]] = []
    for expected_stratum, file_name, reason in CURATED_SAMPLES:
        row = metadata[metadata["file_name"] == file_name]
        if row.empty:
            raise FileNotFoundError(f"Curated sample is not in eval metadata: {file_name}")
        metadata_row = row.iloc[0]
        if str(metadata_row["stratum"]) != expected_stratum:
            raise RuntimeError(f"{file_name} is {metadata_row['stratum']}, expected {expected_stratum}")

        gt_mask = gt_masks[file_name]
        stem = Path(file_name).stem
        gt_boxes = boxes_by_name[file_name]
        pred_masks: dict[str, np.ndarray] = {}
        boxes_by_key: dict[str, list[list[float]]] = {"gt": gt_boxes}
        for _, key, output_key, _, box_source in PIPELINES:
            if key == "gt":
                continue
            output_root = resolve_path(config["paths"][output_key])
            mask_path = output_root / "masks" / f"{stem}.png"
            pred_masks[key] = load_binary_mask(mask_path, gt_mask.shape)
            if box_source == "gt":
                boxes_by_key[key] = gt_boxes
            elif box_source == "yolo":
                boxes_by_key[key] = load_prompt_boxes(output_root, stem)

        output_path = hero_dir / f"{expected_stratum}__{stem}_hero.png"
        make_card(
            image_path=image_dir / file_name,
            gt_mask=gt_mask,
            pred_masks=pred_masks,
            boxes=gt_boxes,
            boxes_by_key=boxes_by_key,
            metadata_row=metadata_row,
            reason=reason,
            output_path=output_path,
        )

        payload = metadata_row.to_dict()
        payload["reason"] = reason
        payload["visualization"] = str(output_path.relative_to(ROOT))
        rows.append(payload)

    selected = pd.DataFrame(rows)
    selected.to_csv(output_dir / "selected_curated_samples.csv", index=False)
    print(f"Wrote curated examples: {output_dir}")


if __name__ == "__main__":
    main()
