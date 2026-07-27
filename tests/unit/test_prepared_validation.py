from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from yolo_sam.data.prepared_validation import (
    build_detector_training_content_manifest,
    build_prepared_content_manifest,
    validate_detector_training_content_manifest,
    validate_prepared_content_manifest,
    validate_prepared_matched_dataset,
)


def write_split(
    root: Path,
    split: str,
    *,
    scene: str,
    stratum: str | None = None,
) -> None:
    split_root = root / split
    (split_root / "images").mkdir(parents=True)
    (split_root / "labels").mkdir()
    file_name = f"{scene}_0000.png"
    (split_root / "images" / file_name).touch()
    (split_root / "labels" / f"{scene}_0000.txt").write_text(
        "0 0.15 0.15 0.1 0.1\n",
        encoding="utf-8",
    )
    annotation = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "segmentation": [[10, 10, 20, 10, 20, 20, 10, 20]],
        "area": 100,
        "bbox": [10, 10, 20, 20],
        "iscrowd": 0,
        "source_annotation_id": 100,
        "bbox_source": "human_annotation",
        "reference_type": "human",
    }
    (split_root / "_annotations.coco.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": file_name,
                        "width": 100,
                        "height": 100,
                    }
                ],
                "annotations": [annotation],
                "categories": [{"id": 1, "name": "plane"}],
            }
        ),
        encoding="utf-8",
    )
    metadata = {
        "image_id": 1,
        "file_name": file_name,
        "source_file_name": f"{scene}.png",
        "source_scene_id": scene,
        "num_objects": 1,
        "mask_area_ratio": 0.01,
        "max_pair_bbox_iou": 0.0,
    }
    if stratum is not None:
        metadata.update(
            {
                "stratum": stratum,
                "area_threshold": 0.02,
            }
        )
    pd.DataFrame([metadata]).to_csv(split_root / "metadata.csv", index=False)


class PreparedValidationTest(unittest.TestCase):
    def test_valid_non_test_splits_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_split(root, "train", scene="train_scene")
            write_split(root, "validation", scene="validation_scene")
            report = validate_prepared_matched_dataset(
                root,
                image_size=100,
                expected_test_per_stratum=1,
                overlap_threshold=0.001,
                required_splits=("train", "validation"),
            )
            self.assertTrue(report.passed, report.to_dict())

    def test_source_scene_leakage_is_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_split(root, "train", scene="shared")
            write_split(root, "validation", scene="shared")
            report = validate_prepared_matched_dataset(
                root,
                image_size=100,
                expected_test_per_stratum=1,
                overlap_threshold=0.001,
                required_splits=("train", "validation"),
            )
            self.assertFalse(report.passed)
            self.assertIn(
                "SOURCE_SCENE_SPLIT_LEAKAGE",
                {finding.code for finding in report.findings},
            )

    def test_empty_or_area_mismatched_reference_mask_is_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_split(root, "train", scene="train_scene")
            write_split(root, "validation", scene="validation_scene")
            annotation_path = (
                root / "train" / "_annotations.coco.json"
            )
            payload = json.loads(annotation_path.read_text(encoding="utf-8"))
            payload["annotations"][0]["segmentation"] = [
                [10, 10, 20, 10, 30, 10]
            ]
            annotation_path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            report = validate_prepared_matched_dataset(
                root,
                image_size=100,
                expected_test_per_stratum=1,
                overlap_threshold=0.001,
                required_splits=("train", "validation"),
            )
            codes = {finding.code for finding in report.findings}

        self.assertFalse(report.passed)
        self.assertIn("INVALID_OR_EMPTY_REFERENCE_MASK", codes)
        self.assertIn("REFERENCE_MASK_AREA_MISMATCH", codes)

    def test_content_manifest_detects_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_split(root, "train", scene="train_scene")
            manifest = build_prepared_content_manifest(
                root,
                splits=("train",),
            )
            self.assertEqual(
                validate_prepared_content_manifest(root, manifest),
                [],
            )
            label = root / "train" / "labels" / "train_scene_0000.txt"
            label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            changed_errors = validate_prepared_content_manifest(
                root,
                manifest,
            )
            self.assertTrue(
                any("hash uyuşmazlığı" in error for error in changed_errors)
            )
            label.write_text(
                "0 0.15 0.15 0.1 0.1\n",
                encoding="utf-8",
            )
            self.assertEqual(
                validate_prepared_content_manifest(root, manifest),
                [],
            )
            extra = root / "train" / "images" / "unexpected.png"
            extra.touch()
            extra_errors = validate_prepared_content_manifest(root, manifest)
            self.assertTrue(
                any(
                    "manifestte olmayan" in error
                    for error in extra_errors
                )
            )

    def test_detector_manifest_excludes_masks_and_test_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "data.yaml").write_text("path: .\n", encoding="utf-8")
            for split in ("train", "validation", "test"):
                images = root / split / "images"
                labels = root / split / "labels"
                images.mkdir(parents=True)
                labels.mkdir(parents=True)
                (images / f"{split}.png").write_bytes(split.encode())
                (labels / f"{split}.txt").write_text(
                    "0 0.5 0.5 0.1 0.1\n",
                    encoding="utf-8",
                )
                (root / split / "_annotations.coco.json").write_text(
                    "{}",
                    encoding="utf-8",
                )

            manifest = build_detector_training_content_manifest(root)
            paths = {row["path"] for row in manifest["files"]}

            self.assertEqual(
                validate_detector_training_content_manifest(root, manifest),
                [],
            )
            self.assertIn("train/images/train.png", paths)
            self.assertIn("validation/labels/validation.txt", paths)
            self.assertNotIn("test/images/test.png", paths)
            self.assertFalse(
                any(path.endswith("_annotations.coco.json") for path in paths)
            )


if __name__ == "__main__":
    unittest.main()
