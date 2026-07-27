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


class LocalAssetTests(unittest.TestCase):
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
