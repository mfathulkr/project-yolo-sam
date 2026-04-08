from __future__ import annotations

import json
from pathlib import Path


def load_ground_truth_boxes(coco_dir: Path) -> dict[str, list[list[float]]]:
    annotation_path = coco_dir / "_annotations.coco.json"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))

    id_to_name = {image["id"]: image["file_name"] for image in data["images"]}
    boxes_by_name: dict[str, list[list[float]]] = {name: [] for name in id_to_name.values()}

    for annotation in data["annotations"]:
        image_name = id_to_name[annotation["image_id"]]
        x, y, w, h = annotation["bbox"]
        boxes_by_name[image_name].append([float(x), float(y), float(x + w), float(y + h)])

    return boxes_by_name
