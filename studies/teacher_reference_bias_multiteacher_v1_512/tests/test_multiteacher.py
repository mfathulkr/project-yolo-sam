from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from pycocotools import mask as mask_utils


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from teacher_reference_bias_multiteacher.pseudo_reference import (  # noqa: E402
    build_pseudo_reference_rows,
)
from teacher_reference_bias_multiteacher.io import portable_path  # noqa: E402
from teacher_reference_bias_multiteacher.rle_metrics import (  # noqa: E402
    binary_metrics_from_rle,
    compare_dense_and_rle,
)


def encoded(mask: np.ndarray) -> dict[str, object]:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {
        "size": [int(value) for value in rle["size"]],
        "counts": rle["counts"].decode("ascii"),
    }


def test_rle_metrics_match_dense_metrics() -> None:
    prediction = np.zeros((32, 48), dtype=bool)
    prediction[4:20, 7:30] = True
    reference = np.zeros_like(prediction)
    reference[8:25, 11:34] = True
    differences = compare_dense_and_rle(encoded(prediction), encoded(reference))
    assert max(differences.values()) == 0.0


def test_empty_prediction_metrics() -> None:
    prediction = np.zeros((12, 12), dtype=bool)
    reference = np.zeros_like(prediction)
    reference[2:5, 3:8] = True
    metrics = binary_metrics_from_rle(encoded(prediction), encoded(reference))
    assert metrics["iou"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["false_negative_pixels"] == 15


def test_teacher_identity_is_enforced() -> None:
    mask = encoded(np.ones((4, 4), dtype=bool))
    row = {
        "instance_id": "d:1:1",
        "image_id": "d:1",
        "source_scene_id": "scene",
        "stratum": "no_overlap__low_mask_area",
        "model_id": "facebook/sam2.1-hiera-large",
        "model_version": "version",
        "prompt_type": "gt_bbox",
        "prompt_source": "human_annotation",
        "run_id": "run",
        "status": "ok",
        "predicted_mask_rle": mask,
    }
    references = build_pseudo_reference_rows([row], teacher="sam2")
    assert references[0]["reference_type"] == "pseudo_sam2"
    assert references[0]["mask_rle"] == mask


def test_source_predictions_exist() -> None:
    from teacher_reference_bias_multiteacher.paths import (
        BBOX_SOURCES,
        DATASETS,
        MODELS,
        prediction_path,
    )

    for source in DATASETS.values():
        assert source.coco_path.is_file()
        for model in MODELS:
            for bbox_source in BBOX_SOURCES:
                path = prediction_path(source, model, bbox_source)
                assert path.is_file(), path
                first = json.loads(next(path.open(encoding="utf-8")))
                assert first["instance_id"].startswith(source.dataset_id + ":")


def test_manifest_paths_are_repository_relative(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    artifact = repository / "studies" / "study" / "artifact.json"
    artifact.parent.mkdir(parents=True)
    artifact.touch()
    assert portable_path(artifact, repository) == "studies/study/artifact.json"

    outside = tmp_path / "outside.json"
    outside.touch()
    with pytest.raises(ValueError, match="Repository dışındaki yol"):
        portable_path(outside, repository)
