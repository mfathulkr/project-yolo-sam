from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from sam3_bbox_study.io_utils import ensure_dir, list_images, save_binary_mask
from sam3_bbox_study.models.remotesam_local import LocalRemoteSAMSegmenter


def run_remotesam_text_pipeline(
    images_dir: Path,
    output_dir: Path,
    external_root: Path,
    checkpoint_path: Path,
    prompts: list[str],
    device: str | int,
    prob_threshold: float,
    use_epoc: bool = False,
    limit: int | None = None,
) -> None:
    masks_dir = ensure_dir(output_dir / "masks")
    raw_dir = ensure_dir(output_dir / "raw")

    segmenter = LocalRemoteSAMSegmenter(
        external_root=external_root,
        checkpoint_path=checkpoint_path,
        device=device,
        use_epoc=use_epoc,
    )

    image_paths = list_images(images_dir)
    if limit is not None:
        image_paths = image_paths[: int(limit)]

    for image_path in tqdm(image_paths, desc="RemoteSAM text segmentation"):
        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")
        result = segmenter.segment(
            image=image,
            prompts=prompts,
            prob_threshold=prob_threshold,
        )
        save_binary_mask(result.merged_mask, masks_dir / f"{image_path.stem}.png")
        raw_payload = {
            "prompts": result.prompts,
            "prompt_area_ratios": result.prompt_area_ratios,
            "prompt_score_means": result.prompt_score_means,
            "score_min": float(result.score_map.min()) if result.score_map.size else 0.0,
            "score_max": float(result.score_map.max()) if result.score_map.size else 0.0,
            "score_mean": float(result.score_map.mean()) if result.score_map.size else 0.0,
            "pred_area_ratio": float(result.merged_mask.mean()) if result.merged_mask.size else 0.0,
        }
        with (raw_dir / f"{image_path.stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(raw_payload, handle, indent=2)
