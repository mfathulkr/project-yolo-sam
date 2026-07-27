from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from yolo_sam.io_utils import ensure_dir, list_images, save_binary_mask
from isaid_vehicle.models.segearth_ov3_local import LocalSegEarthOV3Segmenter


def run_segearth_ov3_pipeline(
    images_dir: Path,
    output_dir: Path,
    external_root: Path,
    checkpoint_path: Path,
    bpe_path: Path,
    prompts: list[str],
    device: str | int,
    confidence_threshold: float,
    prob_threshold: float,
    resolution: int,
    use_semantic_head: bool,
    use_instance_head: bool,
    use_presence_score: bool,
    limit: int | None = None,
) -> None:
    masks_dir = ensure_dir(output_dir / "masks")
    raw_dir = ensure_dir(output_dir / "raw")

    segmenter = LocalSegEarthOV3Segmenter(
        external_root=external_root,
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        device=device,
        confidence_threshold=confidence_threshold,
        resolution=resolution,
    )

    image_paths = list_images(images_dir)
    if limit is not None:
        image_paths = image_paths[: int(limit)]

    for image_path in tqdm(image_paths, desc="SegEarth-OV3 SAM3 text fusion"):
        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")
        result = segmenter.segment(
            image=image,
            prompts=prompts,
            prob_threshold=prob_threshold,
            use_semantic_head=use_semantic_head,
            use_instance_head=use_instance_head,
            use_presence_score=use_presence_score,
        )
        save_binary_mask(result.merged_mask, masks_dir / f"{image_path.stem}.png")
        raw_payload = {
            "prompts": result.prompts,
            "presence_scores": result.presence_scores,
            "instance_counts": result.instance_counts,
            "instance_score_means": result.instance_score_means,
            "score_min": float(result.score_map.min()) if result.score_map.size else 0.0,
            "score_max": float(result.score_map.max()) if result.score_map.size else 0.0,
            "score_mean": float(result.score_map.mean()) if result.score_map.size else 0.0,
            "pred_area_ratio": float(result.merged_mask.mean()) if result.merged_mask.size else 0.0,
        }
        with (raw_dir / f"{image_path.stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(raw_payload, handle, indent=2)
