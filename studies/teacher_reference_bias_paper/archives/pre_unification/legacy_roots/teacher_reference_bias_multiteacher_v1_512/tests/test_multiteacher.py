from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mask_utils


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
for source_root in (REPO_ROOT / "src", STUDY_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
if str(STUDY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT / "scripts"))

from teacher_reference_bias_multiteacher.pseudo_reference import (  # noqa: E402
    build_pseudo_reference_rows,
)
from teacher_reference_bias_multiteacher.comparison_report import (  # noqa: E402
    color_score,
)
from teacher_reference_bias_multiteacher.io import portable_path  # noqa: E402
from teacher_reference_bias_multiteacher.rle_metrics import (  # noqa: E402
    binary_metrics_from_rle,
    compare_dense_and_rle,
)
from build_pseudo_references import validate_source_manifest  # noqa: E402


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


def test_empty_reference_for_known_instance_is_not_success() -> None:
    empty = np.zeros((12, 12), dtype=bool)
    metrics = binary_metrics_from_rle(
        encoded(empty),
        encoded(empty),
        known_positive_instance=True,
    )
    assert metrics["iou"] == 0.0
    assert metrics["dice"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_error_count_colors_use_lower_is_better_semantics() -> None:
    clean = pd.Series({"Instances": 12_051, "Empty masks": 0})
    degraded = pd.Series({"Instances": 12_051, "Empty masks": 19})
    assert color_score("Empty masks", 0, clean) == 1.0
    assert color_score("Empty rate", "0.000", clean) == 1.0
    assert color_score("Status mismatch", 0, clean) == 1.0
    assert color_score("Empty masks", 19, degraded) < 1.0
    assert color_score("Avg IoU", "0.700", clean) == 0.7


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


def test_sam3_source_manifest_requires_completed_pvs_lineage(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"instance_id":"d:1:1"}\n', encoding="utf-8")
    digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
    source = tmp_path / "source.txt"
    source.write_text("source\n", encoding="utf-8")
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "status": "completed",
        "stage": "gt_bbox_segmentation",
        "run_id": "run",
        "config_hash": "a" * 64,
        "input_drift": [],
        "input_file_fingerprints_at_finish": {
            "source": {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": source_digest,
            }
        },
        "parameters": {
            "model": "sam3",
            "prompt_type": "gt_bbox",
            "model_config": {
                "inference_interface": "sam3_tracker_pvs",
                "mask_threshold": 0.0,
                "box_batch_size": 128,
            },
        },
        "output_file_fingerprints": {
            "predictions": {
                "path": str(predictions),
                "bytes": predictions.stat().st_size,
                "sha256": digest,
            }
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    returned_path, returned = validate_source_manifest(predictions, teacher="sam3")
    assert returned_path == manifest_path
    assert returned["run_id"] == "run"

    manifest["parameters"]["model_config"]["inference_interface"] = "sam3_pcs"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="tracker PVS"):
        validate_source_manifest(predictions, teacher="sam3")


def test_source_manifest_rejects_hash_mismatch(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{}\n', encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text("source\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "stage": "gt_bbox_segmentation",
                "config_hash": "a" * 64,
                "input_drift": [],
                "input_file_fingerprints_at_finish": {
                    "source": {
                        "path": str(source),
                        "bytes": source.stat().st_size,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                },
                "parameters": {
                    "model": "sam2",
                    "prompt_type": "gt_bbox",
                    "model_config": {},
                },
                    "output_file_fingerprints": {
                        "predictions": {
                            "path": str(predictions),
                            "bytes": predictions.stat().st_size,
                            "sha256": "0" * 64,
                        }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_source_manifest(predictions, teacher="sam2")
