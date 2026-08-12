from __future__ import annotations

import json
import zipfile

from teacher_reference_bias_multiteacher.paths import DATASETS, REPO_ROOT


def test_sixteen_full_metric_reports_are_complete_and_portable() -> None:
    report_count = 0
    for source in DATASETS.values():
        for reference_type in source.reference_types:
            root = source.reports_root / "full_metrics" / reference_type
            slug = f"{source.experiment_id}_{reference_type}"
            markdown = root / f"{slug}_full_metric_document.md"
            docx = root / f"{slug}_full_metric_document_colored.docx"
            pdf = root / f"{slug}_full_metric_document_colored.pdf"
            for path in (markdown, docx, pdf):
                assert path.is_file() and path.stat().st_size > 1000
            with zipfile.ZipFile(docx) as archive:
                assert "word/document.xml" in archive.namelist()
            text = markdown.read_text(encoding="utf-8")
            assert "No Overlap × Low Mask Area" in text
            assert "Overlap × High Mask Area" in text
            assert "BBox mAP50-95" in text
            manifest = json.loads((root / "report_manifest.json").read_text())
            assert manifest["status"] == "completed"
            for path in (*manifest["inputs"], *manifest["outputs"]):
                assert not path.startswith("/")
                assert (REPO_ROOT / path).is_file()
            report_count += 1
    assert report_count == 16


def test_qualitative_scope_includes_all_target_instances() -> None:
    for source in DATASETS.values():
        manifest = json.loads((source.figures_root / "manifest.json").read_text())
        assert manifest["qualitative_scope"] == "all_target_instances_in_selected_images"
