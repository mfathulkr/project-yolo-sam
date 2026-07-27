from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from repair_legacy_detector_manifests import manifest_needs_repair


def fingerprinted_manifest(capture: str) -> dict[str, object]:
    return {
        "input_fingerprint_capture": capture,
        "input_file_fingerprints": {},
        "input_file_fingerprints_at_finish": {},
        "output_file_fingerprints": {},
    }


class LegacyDetectorManifestRepairTest(unittest.TestCase):
    def test_accepts_start_and_completed_provenance_repair(self) -> None:
        self.assertFalse(manifest_needs_repair(fingerprinted_manifest("start")))
        self.assertFalse(
            manifest_needs_repair(
                fingerprinted_manifest("provenance_repair")
            )
        )

    def test_rejects_legacy_or_incomplete_fingerprints(self) -> None:
        self.assertTrue(manifest_needs_repair({}))
        incomplete = fingerprinted_manifest("provenance_repair")
        incomplete.pop("output_file_fingerprints")
        self.assertTrue(manifest_needs_repair(incomplete))


if __name__ == "__main__":
    unittest.main()
