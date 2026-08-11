from __future__ import annotations

import unittest
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIES_ROOT = REPO_ROOT / "studies"

EXPECTED_STUDIES = {
    "isaid_vehicle_study",
    "landcover_building_study",
    "samrs_sota_plane_study",
    "semantic_drone_car_study",
    "teacher_reference_bias_small_vehicle_v1_512",
    "teacher_reference_bias_multiteacher_v1_512",
    "teacher_reference_bias_v1",
    "teacher_reference_bias_v2_512",
}

EXPECTED_REPORTS = (
    "studies/isaid_vehicle_study/reports/"
    "isaid_vehicle_full_metric_document_colored.pdf",
    "studies/samrs_sota_plane_study/reports/"
    "samrs_sota_plane_full_metric_document_colored.pdf",
    "studies/teacher_reference_bias_v1/reports/paper/"
    "teacher_reference_bias_paper_6pages.pdf",
    "studies/teacher_reference_bias_v1/reports/full_metrics/"
    "isaid_plane/isaid_plane_full_metric_document_colored.pdf",
    "studies/teacher_reference_bias_v1/reports/full_metrics/"
    "samrs_sota_plane/samrs_sota_plane_full_metric_document_colored.pdf",
    "studies/teacher_reference_bias_v2_512/reports/full_metrics/"
    "isaid_plane_human/isaid_plane_human_full_metric_document_colored.pdf",
    "studies/teacher_reference_bias_v2_512/reports/full_metrics/"
    "isaid_plane_pseudo_sam1/"
    "isaid_plane_pseudo_sam1_full_metric_document_colored.pdf",
    "studies/teacher_reference_bias_v2_512/reports/full_metrics/"
    "samrs_sota_plane/samrs_sota_plane_full_metric_document_colored.pdf",
    "studies/teacher_reference_bias_multiteacher_v1_512/reports/full_metrics/"
    "isaid_plane_pseudo_sam2/isaid_plane_pseudo_sam2_full_metric_document_colored.pdf",
    "studies/teacher_reference_bias_multiteacher_v1_512/reports/full_metrics/"
    "isaid_plane_pseudo_sam3/isaid_plane_pseudo_sam3_full_metric_document_colored.pdf",
    "studies/teacher_reference_bias_multiteacher_v1_512/reports/full_metrics/"
    "isaid_small_vehicle_pseudo_sam2/"
    "isaid_small_vehicle_pseudo_sam2_full_metric_document_colored.pdf",
    "studies/teacher_reference_bias_multiteacher_v1_512/reports/full_metrics/"
    "isaid_small_vehicle_pseudo_sam3/"
    "isaid_small_vehicle_pseudo_sam3_full_metric_document_colored.pdf",
    "studies/teacher_reference_bias_multiteacher_v1_512/reports/teacher_comparison/"
    "sam_teacher_pseudo_reference_comparison_colored.pdf",
)

FOREIGN_STUDY_REFERENCE = re.compile(r"studies/([A-Za-z0-9_]+)")
REFERENCE_SCAN_SUFFIXES = {".json", ".py", ".toml", ".yaml", ".yml"}
REFERENCE_SCAN_DIRS = ("configs", "scripts", "src")
REFERENCE_AUDIT_ALLOWLIST = {
    (
        "teacher_reference_bias_v1",
        "scripts/snapshot_study_state.py",
    ),
    # V2 preparation consumes the SHA-256-pinned v1 tile corpus, never its
    # weights, predictions, metrics, or reports.
    (
        "teacher_reference_bias_v2_512",
        "configs/datasets/isaid_plane.yaml",
    ),
    (
        "teacher_reference_bias_v2_512",
        "configs/datasets/samrs_sota_plane.yaml",
    ),
}


def is_external_or_generated(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT)
    return any(
        part in {".git", ".venv", "__pycache__", ".pytest_cache"}
        for part in relative.parts
    )


class RepositoryLayoutTest(unittest.TestCase):
    def test_every_declared_study_has_its_own_readme(self) -> None:
        actual = {
            path.name
            for path in STUDIES_ROOT.iterdir()
            if path.is_dir()
        }
        self.assertEqual(actual, EXPECTED_STUDIES)
        for study in EXPECTED_STUDIES:
            self.assertTrue((STUDIES_ROOT / study / "README.md").is_file())

    def test_legacy_root_directories_are_absent(self) -> None:
        for name in (
            "artifacts",
            "configs",
            "data",
            "results",
            "runs",
            "scripts",
        ):
            self.assertFalse((REPO_ROOT / name).exists(), name)
        self.assertEqual(list(REPO_ROOT.glob("presentation_*")), [])

    def test_placeholder_files_are_absent(self) -> None:
        placeholders = [
            path
            for path in REPO_ROOT.rglob("*")
            if path.is_file()
            and path.name in {".gitkeep", ".keep"}
            and not is_external_or_generated(path)
        ]
        self.assertEqual(placeholders, [])

    def test_historical_and_current_reports_remain_owned_by_studies(
        self,
    ) -> None:
        for relative in EXPECTED_REPORTS:
            self.assertTrue((REPO_ROOT / relative).is_file(), relative)

    def test_active_study_owns_plan_method_and_qa_documents(self) -> None:
        for study in (
            "teacher_reference_bias_v2_512",
            "teacher_reference_bias_small_vehicle_v1_512",
        ):
            docs = STUDIES_ROOT / study / "docs"
            for name in ("EXPERIMENT_PLAN.md", "METHOD.md", "QA_CHECKLIST.md"):
                self.assertTrue((docs / name).is_file(), f"{study}/{name}")

    def test_retired_package_name_is_not_imported(self) -> None:
        offenders = []
        for root in (
            REPO_ROOT / "src",
            REPO_ROOT / "studies",
            REPO_ROOT / "tests",
            REPO_ROOT / "tools",
        ):
            for path in root.rglob("*.py"):
                if path == Path(__file__).resolve():
                    continue
                if is_external_or_generated(path):
                    continue
                text = path.read_text(encoding="utf-8")
                if (
                    "from sam3_bbox_study" in text
                    or "import sam3_bbox_study" in text
                ):
                    offenders.append(path)
        self.assertEqual(offenders, [])

    def test_runtime_files_do_not_depend_on_foreign_study_outputs(
        self,
    ) -> None:
        offenders: list[str] = []
        for study_root in sorted(STUDIES_ROOT.iterdir()):
            if not study_root.is_dir():
                continue
            study_name = study_root.name
            for directory_name in REFERENCE_SCAN_DIRS:
                directory = study_root / directory_name
                if not directory.is_dir():
                    continue
                for path in directory.rglob("*"):
                    if (
                        not path.is_file()
                        or path.suffix not in REFERENCE_SCAN_SUFFIXES
                        or is_external_or_generated(path)
                    ):
                        continue
                    relative = path.relative_to(study_root).as_posix()
                    if (
                        study_name,
                        relative,
                    ) in REFERENCE_AUDIT_ALLOWLIST:
                        continue
                    text = path.read_text(encoding="utf-8")
                    foreign = sorted(
                        {
                            match
                            for match in FOREIGN_STUDY_REFERENCE.findall(text)
                            if match != study_name
                        }
                    )
                    if foreign:
                        offenders.append(
                            f"{path.relative_to(REPO_ROOT)} -> {foreign}"
                        )
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
