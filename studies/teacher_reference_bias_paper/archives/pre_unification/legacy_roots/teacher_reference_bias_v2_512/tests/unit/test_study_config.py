from __future__ import annotations

import unittest
from pathlib import Path

from yolo_sam.data.contracts import BBoxSource, ReferenceType
from teacher_reference_bias.config import (
    load_dataset_study_config,
    load_matched_study_config,
    resolved_config_hash,
)


ROOT = Path(__file__).resolve().parents[2]


class StudyConfigTest(unittest.TestCase):
    def test_primary_protocol_is_strict_and_hashable(self) -> None:
        protocol = load_matched_study_config(
            ROOT / "configs" / "protocol.yaml"
        )
        dataset = load_dataset_study_config(
            ROOT / "configs" / "datasets" / "isaid_plane.yaml"
        )
        self.assertEqual(protocol.image_size, 1024)
        self.assertEqual(protocol.study_id, "teacher_reference_bias_v2_512")
        self.assertEqual(protocol.evaluation["max_per_stratum"], 128)
        self.assertNotIn(BBoxSource.MASK_DERIVED, protocol.bbox_sources)
        self.assertEqual(set(protocol.segmenter_configs), {"sam1", "sam2", "sam3"})
        self.assertEqual(dataset.reference_type, ReferenceType.HUMAN)
        self.assertEqual(len(resolved_config_hash(protocol, dataset)), 64)


if __name__ == "__main__":
    unittest.main()
