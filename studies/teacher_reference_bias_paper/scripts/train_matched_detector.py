from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.data.profiles import get_dataset_profile
from yolo_sam.data.provenance import (
    audit_isaid_coco_dataset,
    audit_samrs_pickle_dataset,
)
from yolo_sam.detection.training import DenseInstanceDetectionTrainer
from yolo_sam.runtime.manifest import (
    finish_run_manifest,
    new_run_manifest,
    write_run_manifest,
)
from teacher_reference_bias.config import (
    load_dataset_study_config,
    load_matched_study_config,
    resolved_config_hash,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train matched YOLO detectors with frozen seeds.")
    parser.add_argument(
        "--protocol",
        type=Path,
        default=STUDY_ROOT / "configs" / "protocol.yaml",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        help="Repeat to run a subset of protocol seeds. Defaults to all frozen seeds.",
    )
    parser.add_argument("--device", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run from train/weights/last.pt.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def verify_dataset(dataset_config: object) -> None:
    dataset = dataset_config
    profile = get_dataset_profile(dataset.profile_id)
    if profile.annotation_format == "coco_instance_segmentation":
        report = audit_isaid_coco_dataset(
            dataset.raw_root,
            profile,
            target_category=dataset.target_category,
        )
    else:
        target_id = next(
            (
                index
                for index, name in enumerate(profile.categories)
                if name.lower() == dataset.target_category.lower()
            ),
            None,
        )
        report = audit_samrs_pickle_dataset(
            dataset.raw_root,
            profile,
            target_category=dataset.target_category,
            declared_target_id=target_id,
            allow_raw_scene_overlap=True,
        )
    if not report.passed:
        codes = ", ".join(finding.code for finding in report.findings)
        raise ValueError(f"Dataset audit failed: {codes}")
    if dataset.version.lower() == "unverified":
        raise ValueError(f"Dataset version is unverified: {dataset.dataset_id}")


def main() -> None:
    args = parse_args()
    if args.resume and args.force:
        raise ValueError("--resume and --force are mutually exclusive")
    protocol = load_matched_study_config(args.protocol)
    dataset = load_dataset_study_config(args.dataset)
    verify_dataset(dataset)

    data_yaml = dataset.prepared_root / "data.yaml"
    detector_content_manifest = (
        dataset.prepared_root / "detector_training_content_manifest.json"
    )
    for required in (data_yaml, detector_content_manifest):
        if not required.exists():
            raise FileNotFoundError(
                f"Prepared dataset input is missing: {required}"
            )

    seeds = tuple(args.seed or protocol.detector_seeds)
    unknown_seeds = set(seeds) - set(protocol.detector_seeds)
    if unknown_seeds:
        raise ValueError(f"Seeds are not in the frozen protocol: {sorted(unknown_seeds)}")

    weights = Path(str(protocol.detector["base_weights"]))
    if not weights.is_absolute():
        weights = ROOT / weights
    if not weights.exists():
        raise FileNotFoundError(f"YOLO base weights are missing: {weights}")

    config_hash = resolved_config_hash(protocol, dataset)
    for seed in seeds:
        run_root = (
            dataset.results_root
            / "detector"
            / f"seed_{seed}"
        )
        best_weights = run_root / "train" / "weights" / "best.pt"
        last_weights = run_root / "train" / "weights" / "last.pt"
        run_id = f"{protocol.study_id}-{dataset.dataset_id}-yolo-seed-{seed}"
        manifest_path = run_root / "manifest.json"
        existing_status = None
        if manifest_path.is_file():
            try:
                existing_status = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                ).get("status")
            except (json.JSONDecodeError, OSError):
                existing_status = None
        existing_completed = existing_status == "completed"
        if args.resume:
            if existing_completed:
                print(f"Skipping completed seed {seed}: {best_weights}")
                continue
            if not last_weights.is_file():
                raise FileNotFoundError(
                    f"Cannot resume seed {seed}; missing checkpoint: {last_weights}"
                )
        elif best_weights.exists() and existing_completed and not args.force:
            print(f"Skipping completed seed {seed}: {best_weights}")
            continue
        if (
            best_weights.exists()
            and not existing_completed
            and not args.force
            and not args.resume
        ):
            raise RuntimeError(
                f"Seed {seed} has a checkpoint but no completed manifest. "
                "Inspect the interrupted run and use --force for an "
                "intentional restart."
            )

        resume_checkpoint: Path | None = None
        if args.resume:
            checkpoint_sha256 = sha256_file(last_weights)
            resume_checkpoint = (
                run_root
                / "resume_inputs"
                / f"last-{checkpoint_sha256[:16]}.pt"
            )
            resume_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if resume_checkpoint.is_file():
                if sha256_file(resume_checkpoint) != checkpoint_sha256:
                    raise ValueError(
                        f"Immutable resume checkpoint hash mismatch: {resume_checkpoint}"
                    )
            else:
                shutil.copy2(last_weights, resume_checkpoint)

        parameters = {
            **protocol.detector,
            "image_size": protocol.image_size,
            "seed": seed,
            "device": args.device,
            "workers": args.workers,
            "resume": args.resume,
            "validation_batch": int(protocol.detector["batch"]),
            "validation_batch_policy": "same_as_training",
        }
        manifest = new_run_manifest(
            project_root=ROOT,
            run_id=run_id,
            stage="train_detector",
            config_hash=config_hash,
            inputs={
                "dataset_config": str(args.dataset.resolve()),
                "data_yaml": str(data_yaml),
                "detector_training_content_manifest": str(
                    detector_content_manifest
                ),
                "base_weights": str(weights),
                **(
                    {"resume_checkpoint": str(resume_checkpoint)}
                    if resume_checkpoint is not None
                    else {}
                ),
            },
            parameters=parameters,
        )
        write_run_manifest(manifest_path, manifest)

        try:
            if args.resume:
                if resume_checkpoint is None:
                    raise RuntimeError("Resume checkpoint snapshot was not created")
                model = YOLO(str(resume_checkpoint))
                model.train(
                    trainer=DenseInstanceDetectionTrainer,
                    resume=True,
                    device=args.device,
                    workers=args.workers,
                )
            else:
                model = YOLO(str(weights))
                model.train(
                    trainer=DenseInstanceDetectionTrainer,
                    data=str(data_yaml),
                    epochs=int(protocol.detector["epochs"]),
                    imgsz=protocol.image_size,
                    batch=int(protocol.detector["batch"]),
                    patience=int(protocol.detector["patience"]),
                    optimizer=str(protocol.detector["optimizer"]),
                    workers=args.workers,
                    project=str(run_root),
                    name="train",
                    exist_ok=args.force,
                    device=args.device,
                    seed=seed,
                    deterministic=True,
                    verbose=False,
                )
        except Exception as exc:
            finish_run_manifest(manifest, status="failed", error=str(exc))
            write_run_manifest(manifest_path, manifest)
            raise

        if not best_weights.exists():
            finish_run_manifest(
                manifest,
                status="failed",
                error=f"Training returned without expected checkpoint: {best_weights}",
            )
            write_run_manifest(manifest_path, manifest)
            raise FileNotFoundError(best_weights)
        results_csv = run_root / "train" / "results.csv"
        args_yaml = run_root / "train" / "args.yaml"
        for required_output in (results_csv, args_yaml):
            if not required_output.is_file():
                finish_run_manifest(
                    manifest,
                    status="failed",
                    error=f"Expected training output is missing: {required_output}",
                )
                write_run_manifest(manifest_path, manifest)
                raise FileNotFoundError(required_output)
        manifest["outputs"] = {
            "best_weights": str(best_weights),
            "results_csv": str(results_csv),
            "args_yaml": str(args_yaml),
        }
        finish_run_manifest(manifest, status="completed")
        write_run_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
