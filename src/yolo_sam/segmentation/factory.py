from __future__ import annotations

from pathlib import Path
from typing import Any

from yolo_sam.models.sam1_local import LocalSam1ImageSegmenter
from yolo_sam.models.sam2_local import LocalSam2ImageSegmenter
from yolo_sam.models.sam3_tracker_local import LocalSam3TrackerImageSegmenter
from yolo_sam.segmentation.box_segmenters import (
    BoxSegmenter,
    Sam1BoxSegmenter,
    Sam2BoxSegmenter,
    Sam3BoxSegmenter,
)


def create_box_segmenter(
    name: str,
    config: dict[str, Any],
    device: str | int,
    project_root: Path,
    hf_token: str | None = None,
) -> BoxSegmenter:
    if name == "sam1":
        model_id = str(config["model_id"])
        revision = str(config["revision"])
        return Sam1BoxSegmenter(
            segmenter=LocalSam1ImageSegmenter(
                model_id=model_id,
                device=device,
                torch_dtype=str(config["torch_dtype"]),
                token=hf_token,
                revision=revision,
            ),
            model_id=model_id,
            mask_threshold=float(config["mask_threshold"]),
            model_version=revision,
        )
    if name == "sam2":
        model_id = str(config["model_id"])
        revision = str(config["revision"])
        return Sam2BoxSegmenter(
            segmenter=LocalSam2ImageSegmenter(
                model_id=model_id,
                device=device,
                torch_dtype=str(config["torch_dtype"]),
                token=hf_token,
                revision=revision,
            ),
            model_id=model_id,
            mask_threshold=float(config["mask_threshold"]),
            model_version=revision,
        )
    if name == "sam3":
        inference_interface = str(config.get("inference_interface", ""))
        if inference_interface != "sam3_tracker_pvs":
            raise ValueError(
                "SAM3 bbox segmentation requires "
                "inference_interface=sam3_tracker_pvs"
            )
        model_dir = Path(str(config["model_dir"]))
        if not model_dir.is_absolute():
            model_dir = project_root / model_dir
        return Sam3BoxSegmenter(
            segmenter=LocalSam3TrackerImageSegmenter(
                model_dir=model_dir,
                device=device,
                torch_dtype=str(config["torch_dtype"]),
                hf_token=hf_token,
            ),
            model_id=str(config["model_id"]),
            mask_threshold=float(config["mask_threshold"]),
            box_batch_size=int(config.get("box_batch_size", 16)),
            model_version=str(config["checkpoint_sha256"]),
        )
    raise ValueError(f"Unknown bbox segmenter: {name}")
