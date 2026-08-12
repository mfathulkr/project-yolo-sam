from __future__ import annotations

import importlib.util
import io
import tarfile
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "manage_local_assets.py"
)
SPEC = importlib.util.spec_from_file_location("manage_local_assets", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSETS)

BUNDLE_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "build_portable_bundles.py"
)
BUNDLE_SPEC = importlib.util.spec_from_file_location(
    "build_portable_bundles", BUNDLE_MODULE_PATH
)
assert BUNDLE_SPEC is not None and BUNDLE_SPEC.loader is not None
BUNDLES = importlib.util.module_from_spec(BUNDLE_SPEC)
BUNDLE_SPEC.loader.exec_module(BUNDLES)


class LocalAssetTests(unittest.TestCase):
    def test_portable_metadata_bundle_rejects_images_and_caches(self) -> None:
        self.assertFalse(
            BUNDLES.metadata_file_allowed(Path("prepared/test/images/scene.png"))
        )
        self.assertFalse(
            BUNDLES.metadata_file_allowed(Path("prepared/train/labels.cache"))
        )
        self.assertTrue(
            BUNDLES.metadata_file_allowed(
                Path("prepared/test/_annotations.coco.json")
            )
        )

    def test_portable_results_bundle_is_seed_42_only(self) -> None:
        self.assertFalse(
            BUNDLES.result_file_allowed(Path("results/train/weights/best.pt"))
        )
        self.assertFalse(
            BUNDLES.result_file_allowed(
                Path("results/detectors/example/seed_123/evaluation/metrics.json")
            )
        )
        self.assertFalse(
            BUNDLES.result_file_allowed(Path("results/train_detached.log"))
        )
        self.assertFalse(
            BUNDLES.result_file_allowed(
                Path("results/finalization/manifest.json")
            )
        )
        self.assertTrue(
            BUNDLES.result_file_allowed(
                Path("results/detectors/example/seed_42/evaluation/metrics.json")
            )
        )
        self.assertTrue(
            BUNDLES.result_file_allowed(Path("results/analysis/aggregates.csv"))
        )

    def test_detects_lfs_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "asset.pt"
            path.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:abc\n"
                "size 123\n",
                encoding="utf-8",
            )
            self.assertTrue(ASSETS.is_lfs_pointer(path))

    def test_safe_extract_rejects_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "unsafe.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                payload = b"unsafe"
                member = tarfile.TarInfo("../outside.txt")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            with self.assertRaises(ValueError):
                ASSETS.safe_extract(
                    archive_path,
                    root / "destination",
                    force=False,
                )

    def test_safe_extract_restores_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.txt"
            source.write_text("content", encoding="utf-8")
            archive_path = root / "safe.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                archive.add(source, arcname="nested/source.txt")
            destination = root / "destination"
            ASSETS.safe_extract(archive_path, destination, force=False)
            self.assertEqual(
                (destination / "nested" / "source.txt").read_text(
                    encoding="utf-8"
                ),
                "content",
            )


if __name__ == "__main__":
    unittest.main()
