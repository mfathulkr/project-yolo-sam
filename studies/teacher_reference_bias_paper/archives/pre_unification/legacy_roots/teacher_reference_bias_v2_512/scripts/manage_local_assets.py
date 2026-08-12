from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download, try_to_load_from_cache


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
BUNDLE_ROOT = STUDY_ROOT / "bundles"
MANIFEST_PATH = BUNDLE_ROOT / "manifest.json"
SAM3_ALLOW_PATTERNS = (
    "LICENSE",
    "README.md",
    "config.json",
    "merges.txt",
    "model.safetensors",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(
        b"version https://git-lfs.github.com/spec/v1\n"
    )


def load_manifest() -> dict[str, object]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported bundle manifest: {MANIFEST_PATH}")
    return payload


def asset_status(*, verify_hashes: bool) -> tuple[list[str], bool]:
    rows: list[str] = []
    passed = True
    manifest = load_manifest()
    for asset in manifest["lfs_assets"]:
        path = REPO_ROOT / str(asset["path"])
        expected_bytes = int(asset["bytes"])
        if not path.exists():
            state = "MISSING"
            passed = False
        elif is_lfs_pointer(path):
            state = "LFS_POINTER"
            passed = False
        elif path.stat().st_size != expected_bytes:
            state = f"SIZE_MISMATCH({path.stat().st_size})"
            passed = False
        elif verify_hashes and sha256_file(path) != str(asset["sha256"]):
            state = "SHA256_MISMATCH"
            passed = False
        else:
            state = "READY"
        rows.append(f"{state:24} {asset['path']}")
    return rows, passed


def runtime_status() -> list[str]:
    rows: list[str] = []
    cached_models = (
        (
            "SAM1 checkpoint",
            "facebook/sam-vit-huge",
            "87aecf0df4ce6b30cd7de76e87673c49644bdf67",
        ),
        (
            "SAM2 checkpoint",
            "facebook/sam2.1-hiera-large",
            "665f8e2ad61cf5f53d65644ff27c8ee525124610",
        ),
    )
    for label, model_id, revision in cached_models:
        cached_path = try_to_load_from_cache(
            model_id,
            "model.safetensors",
            revision=revision,
        )
        ready = isinstance(cached_path, str) and Path(cached_path).is_file()
        rows.append(f"{'READY' if ready else 'MISSING':24} {label}")
    sam3 = REPO_ROOT / "models" / "sam3_hf" / "model.safetensors"
    rows.append(f"{'READY' if sam3.exists() else 'MISSING':24} SAM3 checkpoint")
    for dataset_id in ("isaid_plane", "samrs_sota_plane"):
        images_root = (
            STUDY_ROOT / "data" / "prepared" / dataset_id / "test" / "images"
        )
        image_count = (
            sum(1 for path in images_root.iterdir() if path.is_file())
            if images_root.exists()
            else 0
        )
        state = "READY" if image_count == 512 else f"INCOMPLETE({image_count}/512)"
        rows.append(f"{state:24} {dataset_id} private test images")
    return rows


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def safe_extract(
    archive_path: Path,
    destination: Path,
    *,
    force: bool,
) -> None:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not _is_within_root(target, destination):
                raise ValueError(f"Unsafe archive member: {member.name}")
            if target.is_file() and not force:
                raise FileExistsError(
                    f"{target} already exists; pass --force to replace files"
                )
        archive.extractall(destination, members=members, filter="data")


def restore_bundles(
    bundle_names: list[str],
    destination: Path,
    *,
    force: bool,
) -> None:
    archives = {
        "canonical-results": BUNDLE_ROOT
        / "canonical_results_without_weights.tar.gz",
        "prepared-metadata": BUNDLE_ROOT
        / "prepared_metadata_without_images.tar.gz",
    }
    selected = list(archives) if "all" in bundle_names else bundle_names
    for name in selected:
        archive_path = archives[name]
        if is_lfs_pointer(archive_path):
            raise RuntimeError(f"Run `git lfs pull` first: {archive_path}")
        safe_extract(archive_path, destination, force=force)
        print(f"Restored {name} into {destination}")


def download_models(model_names: list[str]) -> None:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    configs = {
        "sam1": (
            "facebook/sam-vit-huge",
            "87aecf0df4ce6b30cd7de76e87673c49644bdf67",
        ),
        "sam2": (
            "facebook/sam2.1-hiera-large",
            "665f8e2ad61cf5f53d65644ff27c8ee525124610",
        ),
    }
    selected = ["sam1", "sam2", "sam3"] if "all" in model_names else model_names
    for name in selected:
        if name == "sam3":
            if not token:
                raise RuntimeError(
                    "SAM3 requires accepted facebook/sam3 access and HF_TOKEN"
                )
            destination = REPO_ROOT / "models" / "sam3_hf"
            snapshot_download(
                repo_id="facebook/sam3",
                local_dir=str(destination),
                token=token,
                allow_patterns=list(SAM3_ALLOW_PATTERNS),
            )
            print(f"Downloaded SAM3 to {destination}")
            continue
        model_id, revision = configs[name]
        snapshot = snapshot_download(
            repo_id=model_id,
            revision=revision,
            token=token,
        )
        print(f"Downloaded {name.upper()} to {snapshot}")


def export_private_test_images(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    image_roots: list[Path] = []
    for dataset_id in ("isaid_plane", "samrs_sota_plane"):
        images_root = (
            STUDY_ROOT
            / "data"
            / "prepared"
            / dataset_id
            / "test"
            / "images"
        )
        image_count = sum(
            1 for path in images_root.iterdir() if path.is_file()
        )
        if image_count != 512:
            raise ValueError(
                f"Expected 512 {dataset_id} test images, found {image_count}"
            )
        image_roots.append(images_root)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w") as archive:
        for images_root in image_roots:
            archive.add(
                images_root,
                arcname=images_root.relative_to(REPO_ROOT),
                recursive=True,
            )
    print(f"Wrote private test image bundle: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manage local inference assets for the canonical v2 study."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--verify-hashes", action="store_true")
    status.add_argument("--strict", action="store_true")

    restore = subparsers.add_parser("restore")
    restore.add_argument(
        "bundles",
        nargs="+",
        choices=("canonical-results", "prepared-metadata", "all"),
    )
    restore.add_argument("--destination", type=Path, default=REPO_ROOT)
    restore.add_argument("--force", action="store_true")

    models = subparsers.add_parser("download-models")
    models.add_argument(
        "models",
        nargs="+",
        choices=("sam1", "sam2", "sam3", "all"),
    )

    export = subparsers.add_parser("export-private-test-images")
    export.add_argument("--output", type=Path, required=True)

    private_restore = subparsers.add_parser("restore-private-test-images")
    private_restore.add_argument("--archive", type=Path, required=True)
    private_restore.add_argument("--destination", type=Path, default=REPO_ROOT)
    private_restore.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "status":
        asset_rows, passed = asset_status(verify_hashes=args.verify_hashes)
        print("Tracked Git/LFS assets")
        print("\n".join(asset_rows))
        print("\nRuntime-only assets")
        print("\n".join(runtime_status()))
        if args.strict and not passed:
            raise SystemExit(1)
        return
    if args.command == "restore":
        restore_bundles(
            args.bundles,
            args.destination,
            force=args.force,
        )
        return
    if args.command == "download-models":
        download_models(args.models)
        return
    if args.command == "export-private-test-images":
        export_private_test_images(args.output)
        return
    if args.command == "restore-private-test-images":
        safe_extract(
            args.archive,
            args.destination,
            force=args.force,
        )
        print(f"Restored private test images into {args.destination}")
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
