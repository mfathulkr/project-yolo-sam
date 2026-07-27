from __future__ import annotations

import pickle
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from yolo_sam.data.profiles import DatasetProfile
from yolo_sam.data.provenance import (
    audit_isaid_coco_dataset,
    audit_samrs_pickle_dataset,
    source_scene_id,
)


class ProvenanceAuditTest(unittest.TestCase):
    def test_source_scene_id_removes_tile_suffix(self) -> None:
        self.assertEqual(source_scene_id("P1398_0025"), "P1398")
        self.assertEqual(source_scene_id("scene_without_tile"), "scene_without_tile")

    def test_audit_passes_for_matching_profile_and_disjoint_scenes(self) -> None:
        profile = DatasetProfile(
            profile_id="fixture",
            display_name="Fixture",
            categories=("plane",),
            reference_type="pseudo_sam1",
            annotation_format="samrs_pickle_instances",
            expected_instance_keys=frozenset({"mask", "label", "category", "rhbox"}),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            instances = root / "masks" / "ins"
            images.mkdir()
            instances.mkdir(parents=True)
            for stem in ("scene_a_0001", "scene_b_0001"):
                (images / f"{stem}.png").write_bytes(b"fixture")
                with (instances / f"{stem}.pkl").open("wb") as handle:
                    pickle.dump(
                        [
                            {
                                "mask": {"size": [1, 1], "counts": "1"},
                                "label": 0,
                                "category": "plane",
                                "rhbox": [0, 0, 1, 1],
                            }
                        ],
                        handle,
                    )
            (root / "train.txt").write_text("scene_a_0001\n", encoding="utf-8")
            (root / "valid.txt").write_text("scene_b_0001\n", encoding="utf-8")

            report = audit_samrs_pickle_dataset(root, profile)

        self.assertTrue(report.passed)
        self.assertEqual(report.instances, 2)

    def test_audit_rejects_category_mismatch_and_scene_leakage(self) -> None:
        profile = DatasetProfile(
            profile_id="fixture",
            display_name="Fixture",
            categories=("plane",),
            reference_type="pseudo_sam1",
            annotation_format="samrs_pickle_instances",
            expected_instance_keys=frozenset({"mask", "label", "category", "rhbox"}),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            instances = root / "masks" / "ins"
            images.mkdir()
            instances.mkdir(parents=True)
            for stem in ("scene_a_0001", "scene_a_0002"):
                (images / f"{stem}.png").write_bytes(b"fixture")
                with (instances / f"{stem}.pkl").open("wb") as handle:
                    pickle.dump(
                        [
                            {
                                "mask": {"size": [1, 1], "counts": "1"},
                                "label": 4,
                                "category": "ARJ21",
                                "rhbox": [0, 0, 1, 1],
                            }
                        ],
                        handle,
                    )
            (root / "train.txt").write_text("scene_a_0001\n", encoding="utf-8")
            (root / "valid.txt").write_text("scene_a_0002\n", encoding="utf-8")

            report = audit_samrs_pickle_dataset(
                root,
                profile,
                target_category="plane",
                declared_target_id=4,
            )

        codes = {finding.code for finding in report.findings}
        self.assertFalse(report.passed)
        self.assertIn("CATEGORY_PROFILE_MISMATCH", codes)
        self.assertIn("TARGET_ID_CATEGORY_MISMATCH", codes)
        self.assertIn("SOURCE_SCENE_SPLIT_LEAKAGE", codes)

    def test_authoritative_rdet_overrides_bad_pickle_names_after_exact_validation(self) -> None:
        profile = DatasetProfile(
            profile_id="fixture",
            display_name="Fixture",
            categories=("plane",),
            reference_type="pseudo_sam1",
            annotation_format="samrs_pickle_instances",
            expected_instance_keys=frozenset({"mask", "label", "category", "rhbox"}),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            instances = root / "masks" / "ins"
            rdet = root / "rbbtxts"
            images.mkdir()
            instances.mkdir(parents=True)
            rdet.mkdir()
            for stem, scene in (("scene_a_0001", "train"), ("scene_b_0001", "valid")):
                (images / f"{stem}.png").write_bytes(b"fixture")
                with (instances / f"{stem}.pkl").open("wb") as handle:
                    pickle.dump(
                        [
                            {
                                "mask": {"size": [2, 2], "counts": "4"},
                                "rbox": np.asarray(
                                    [[0, 0], [1, 0], [1, 1], [0, 1]],
                                    dtype=np.float32,
                                ),
                                "rhbox": np.asarray([0, 0, 1, 1], dtype=np.float32),
                                "label": 4,
                                "category": "ARJ21",
                            }
                        ],
                        handle,
                    )
                (rdet / f"{stem}.txt").write_text(
                    "0 0 1 0 1 1 0 1 plane 4\n",
                    encoding="utf-8",
                )
                (root / f"{scene}.txt").write_text(f"{stem}\n", encoding="utf-8")

            report = audit_samrs_pickle_dataset(
                root,
                profile,
                target_category="plane",
                declared_target_id=4,
                allow_raw_scene_overlap=True,
            )

        codes = {finding.code for finding in report.findings}
        self.assertTrue(report.passed)
        self.assertIn("AUTHORITATIVE_RDET_VALIDATED", codes)
        self.assertIn("PICKLE_CATEGORY_STRINGS_IGNORED", codes)
        self.assertEqual(report.category_mapping["4"], ["plane"])

    def test_authoritative_rdet_rejects_geometry_mismatch(self) -> None:
        profile = DatasetProfile(
            profile_id="fixture",
            display_name="Fixture",
            categories=("plane",),
            reference_type="pseudo_sam1",
            annotation_format="samrs_pickle_instances",
            expected_instance_keys=frozenset({"mask", "label", "category", "rhbox"}),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            instances = root / "masks" / "ins"
            rdet = root / "rbbtxts"
            images.mkdir()
            instances.mkdir(parents=True)
            rdet.mkdir()
            for stem, scene in (("scene_a_0001", "train"), ("scene_b_0001", "valid")):
                (images / f"{stem}.png").write_bytes(b"fixture")
                with (instances / f"{stem}.pkl").open("wb") as handle:
                    pickle.dump(
                        [
                            {
                                "mask": {"size": [2, 2], "counts": "4"},
                                "rbox": np.asarray(
                                    [[0, 0], [2, 0], [2, 2], [0, 2]],
                                    dtype=np.float32,
                                ),
                                "rhbox": np.asarray([0, 0, 2, 2], dtype=np.float32),
                                "label": 4,
                                "category": "plane",
                            }
                        ],
                        handle,
                    )
                (rdet / f"{stem}.txt").write_text(
                    "0 0 1 0 1 1 0 1 plane 4\n",
                    encoding="utf-8",
                )
                (root / f"{scene}.txt").write_text(f"{stem}\n", encoding="utf-8")

            report = audit_samrs_pickle_dataset(
                root,
                profile,
                target_category="plane",
                declared_target_id=4,
            )

        codes = {finding.code for finding in report.findings}
        self.assertFalse(report.passed)
        self.assertIn("AUTHORITATIVE_RDET_VALIDATION_FAILED", codes)

    def test_isaid_coco_audit_checks_human_annotation_structure(self) -> None:
        profile = DatasetProfile(
            profile_id="isaid_fixture",
            display_name="iSAID fixture",
            categories=("plane",),
            reference_type="human",
            annotation_format="coco_instance_segmentation",
            expected_instance_keys=frozenset(
                {"id", "image_id", "category_id", "segmentation", "bbox"}
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split, scene in (("train", "scene_a"), ("val", "scene_b")):
                annotations = root / split / "Annotations"
                images = root / split / "images"
                annotations.mkdir(parents=True)
                images.mkdir(parents=True)
                image_name = f"{scene}.png"
                (images / image_name).write_bytes(b"fixture")
                payload = {
                    "categories": [{"id": 14, "name": "plane"}],
                    "images": [{"id": 1, "file_name": image_name}],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 14,
                            "segmentation": [[0, 0, 1, 0, 1, 1]],
                            "bbox": [0, 0, 1, 1],
                        }
                    ],
                }
                (annotations / f"iSAID_{split}.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )

            report = audit_isaid_coco_dataset(root, profile)

        self.assertTrue(report.passed)
        self.assertEqual(report.instances, 2)


if __name__ == "__main__":
    unittest.main()
