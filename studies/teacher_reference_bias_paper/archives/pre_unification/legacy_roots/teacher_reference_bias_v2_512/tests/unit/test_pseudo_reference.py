from __future__ import annotations

import unittest

from teacher_reference_bias.pseudo_reference import (
    build_sam1_pseudo_reference_rows,
)

SAM1_MODEL_ID = "facebook/sam-vit-huge"
SAM1_REVISION = "87aecf0df4ce6b30cd7de76e87673c49644bdf67"


def prediction(
    *,
    model_id: str = SAM1_MODEL_ID,
    model_version: str = SAM1_REVISION,
) -> dict[str, object]:
    return {
        "instance_id": "dataset:1:1",
        "image_id": "dataset:1",
        "source_scene_id": "scene",
        "stratum": "no_overlap__low_mask_area",
        "predicted_mask_rle": {"size": [2, 2], "counts": "013"},
        "model_id": model_id,
        "model_version": model_version,
        "prompt_type": "gt_bbox",
        "prompt_source": "human_annotation",
        "status": "ok",
        "run_id": "run",
    }


class PseudoReferenceTest(unittest.TestCase):
    def test_sam1_prediction_becomes_explicit_pseudo_reference(self) -> None:
        rows = build_sam1_pseudo_reference_rows(
            [prediction()],
            expected_model_id=SAM1_MODEL_ID,
            expected_model_version=SAM1_REVISION,
        )
        self.assertEqual(rows[0]["reference_type"], "pseudo_sam1")
        self.assertEqual(rows[0]["instance_id"], "dataset:1:1")

    def test_wrong_teacher_model_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "teacher identity mismatch"):
            build_sam1_pseudo_reference_rows(
                [prediction(model_id="facebook/sam2-hiera-large")],
                expected_model_id=SAM1_MODEL_ID,
                expected_model_version=SAM1_REVISION,
            )

    def test_wrong_teacher_revision_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "teacher identity mismatch"):
            build_sam1_pseudo_reference_rows(
                [prediction(model_version="wrong-revision")],
                expected_model_id=SAM1_MODEL_ID,
                expected_model_version=SAM1_REVISION,
            )


if __name__ == "__main__":
    unittest.main()
