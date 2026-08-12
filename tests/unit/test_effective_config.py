from __future__ import annotations

from yolo_sam.runtime.effective_config import (
    bbox_segmentation_effective_config,
    effective_config_hash,
)


def base_payload(model_config: dict[str, object]) -> dict[str, object]:
    return bbox_segmentation_effective_config(
        study_id="study",
        image_size=1024,
        dataset={
            "dataset_id": "dataset",
            "version": "v1",
            "profile_id": "profile",
            "reference_type": "human",
            "target_category": "plane",
            "area_threshold": 0.01,
        },
        model="sam1",
        model_config=model_config,
        bbox_source="gt_bbox",
        split="test",
    )


def test_effective_hash_ignores_unrelated_model_configuration() -> None:
    first = base_payload({"revision": "sam1-revision", "mask_threshold": 0.0})
    second = base_payload({"revision": "sam1-revision", "mask_threshold": 0.0})
    assert effective_config_hash(first) == effective_config_hash(second)


def test_effective_hash_changes_with_active_model_configuration() -> None:
    first = base_payload({"revision": "revision-a", "mask_threshold": 0.0})
    second = base_payload({"revision": "revision-b", "mask_threshold": 0.0})
    assert effective_config_hash(first) != effective_config_hash(second)


def test_yolo_effective_config_requires_detector_seed() -> None:
    try:
        bbox_segmentation_effective_config(
            study_id="study",
            image_size=1024,
            dataset={"dataset_id": "dataset"},
            model="sam1",
            model_config={},
            bbox_source="yolo_bbox",
            split="test",
        )
    except ValueError as exc:
        assert "detector_seed" in str(exc)
    else:
        raise AssertionError("YOLO effective config accepted a missing seed")
