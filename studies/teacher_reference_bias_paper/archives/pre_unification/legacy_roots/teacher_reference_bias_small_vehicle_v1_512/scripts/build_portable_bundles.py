from __future__ import annotations

import gzip
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Callable, Iterable


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
BUNDLE_ROOT = STUDY_ROOT / "bundles"
RESULTS_ARCHIVE = BUNDLE_ROOT / "canonical_results_without_weights.tar.gz"
METADATA_ARCHIVE = BUNDLE_ROOT / "prepared_metadata_without_images.tar.gz"
MANIFEST_PATH = BUNDLE_ROOT / "manifest.json"
DATASET_IDS = ("isaid_small_vehicle", "samrs_sota_small_vehicle")
RASTER_IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(roots: Iterable[Path], include: Callable[[Path], bool]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        files.extend(path for path in root.rglob("*") if path.is_file() and include(path))
    return sorted(set(files), key=lambda path: path.as_posix())


def write_deterministic_tar_gz(output: Path, files: Iterable[Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gz_handle:
            with tarfile.open(fileobj=gz_handle, mode="w") as archive:
                for path in files:
                    info = archive.gettarinfo(
                        str(path),
                        arcname=str(path.relative_to(REPO_ROOT)),
                    )
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)


def result_file_allowed(path: Path) -> bool:
    if path.suffix == ".pt" or "weights" in path.parts:
        return False
    if path.suffix.lower() in RASTER_IMAGE_SUFFIXES:
        return False
    if path.name.endswith(".log") or path.suffix == ".cache":
        return False
    if "post_training" in path.parts or "finalization" in path.parts:
        return False
    return all(
        not part.startswith("seed_") or part == "seed_42"
        for part in path.parts
    )


def metadata_file_allowed(path: Path) -> bool:
    return "images" not in path.parts and path.suffix != ".cache"


def asset_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    result_files = iter_files([STUDY_ROOT / "results"], result_file_allowed)
    metadata_files = iter_files(
        [STUDY_ROOT / "data" / "prepared" / dataset_id for dataset_id in DATASET_IDS],
        metadata_file_allowed,
    )
    write_deterministic_tar_gz(RESULTS_ARCHIVE, result_files)
    write_deterministic_tar_gz(METADATA_ARCHIVE, metadata_files)

    weights = [
        STUDY_ROOT
        / "results"
        / "detectors"
        / dataset_id
        / "seed_42"
        / "train"
        / "weights"
        / "best.pt"
        for dataset_id in DATASET_IDS
    ]
    for path in weights:
        if not path.is_file():
            raise FileNotFoundError(path)

    assets = [*weights, RESULTS_ARCHIVE, METADATA_ARCHIVE]
    manifest = {
        "schema_version": 1,
        "study_id": STUDY_ROOT.name,
        "detector_seed": 42,
        "contains_dataset_images": False,
        "lfs_assets": [asset_record(path) for path in assets],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(RESULTS_ARCHIVE)
    print(METADATA_ARCHIVE)
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
