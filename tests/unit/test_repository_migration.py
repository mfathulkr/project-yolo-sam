from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import repository_layout_migration as migration


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class RepositoryMigrationTest(unittest.TestCase):
    def test_inventory_can_verify_audited_path_rewrites_against_originals(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "metadata.json"
            metadata.write_text('{"root": "/old/path"}\n', encoding="utf-8")
            (root / "payload.bin").write_bytes(b"unchanged")
            before = migration.inventory(root)

            original = root.parent / f"{root.name}.metadata.original"
            original.write_bytes(metadata.read_bytes())
            metadata.write_text('{"root": "/new/longer/path"}\n', encoding="utf-8")
            reconstructed = migration.inventory(
                root,
                content_overrides={metadata: original},
            )

            self.assertEqual(before.file_count, reconstructed.file_count)
            self.assertEqual(before.total_bytes, reconstructed.total_bytes)
            self.assertEqual(before.sha256, reconstructed.sha256)

    def test_rewrite_audit_rejects_unrecorded_follow_up_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = root / "current.json"
            original = root / "current.json.original"
            before = '{"root": "/old/path"}\n'
            after = '{"root": "/new/path"}\n'
            original.write_text(before, encoding="utf-8")
            current.write_text(after, encoding="utf-8")
            audit = root / "layout_migration.json"
            audit.write_text(
                json.dumps(
                    {
                        "migration_id": migration.MIGRATION_ID,
                        "status": "pass",
                        "modified_file_count": 1,
                        "files": [
                            {
                                "path": str(current),
                                "original_copy": str(original),
                                "before_sha256": sha256_text(before),
                                "after_sha256": sha256_text(after),
                                "bytes_after": len(after.encode("utf-8")),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(migration, "REWRITE_AUDIT", audit),
                patch.object(
                    migration,
                    "SUPPLEMENTAL_REWRITE_AUDIT",
                    root / "missing-supplemental.json",
                ),
            ):
                overrides = migration._audited_rewrite_overrides()
                self.assertEqual(overrides, {current: original})
                current.write_text(after + "tamper\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError,
                    "changed after rewrite",
                ):
                    migration._audited_rewrite_overrides()


if __name__ == "__main__":
    unittest.main()
