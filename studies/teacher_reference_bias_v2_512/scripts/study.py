from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
ROOT = REPO_ROOT
for source_root in (STUDY_ROOT / "src", REPO_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from yolo_sam.data.profiles import get_dataset_profile
from yolo_sam.data.provenance import (
    DatasetAuditReport,
    audit_isaid_coco_dataset,
    audit_samrs_pickle_dataset,
    write_audit_report,
)
from yolo_sam.data.split import (
    SplitCandidate,
    grouped_stratified_split,
    write_split_manifest,
)
from teacher_reference_bias.config import (
    DatasetStudyConfig,
    MatchedStudyConfig,
    load_dataset_study_config,
    load_matched_study_config,
    resolved_config_hash,
)

DEFAULT_PROTOCOL = Path(
    "studies/teacher_reference_bias_v2_512/configs/protocol.yaml"
)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def add_protocol_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )


def add_force_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force",
        action="store_true",
        help="Intentionally replace a completed stage output.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible entry point for the matched YOLO-SAM study."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit one dataset before use.")
    audit_parser.add_argument("--dataset", type=Path, required=True)
    audit_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "studies/teacher_reference_bias_v2_512/results/dataset_audits"
        ),
    )

    preflight_parser = subparsers.add_parser(
        "preflight",
        help="Validate the protocol and every dataset before GPU work.",
    )
    preflight_parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
    )
    preflight_parser.add_argument(
        "--dataset",
        type=Path,
        action="append",
        required=True,
        help="Repeat for every dataset config in the comparison.",
    )
    preflight_parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "studies/teacher_reference_bias_v2_512/results/preflight.json"
        ),
    )

    split_parser = subparsers.add_parser(
        "split",
        help="Build a source-scene-safe split from a candidate JSON file.",
    )
    split_parser.add_argument("--candidates", type=Path, required=True)
    split_parser.add_argument("--output-dir", type=Path, required=True)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--train", type=float, default=0.70)
    split_parser.add_argument("--validation", type=float, default=0.15)
    split_parser.add_argument("--test", type=float, default=0.15)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare one matched dataset with source-scene-safe splits.",
    )
    add_protocol_argument(prepare_parser)
    prepare_parser.add_argument("--dataset", type=Path, required=True)
    add_force_argument(prepare_parser)

    validate_parser = subparsers.add_parser(
        "validate-prepared",
        help="Validate one prepared matched dataset before inference.",
    )
    add_protocol_argument(validate_parser)
    validate_parser.add_argument("--dataset", type=Path, required=True)
    validate_parser.add_argument("--output", type=Path)

    model_provenance_parser = subparsers.add_parser(
        "model-provenance",
        help="Verify pinned SAM revisions and checkpoint hashes.",
    )
    add_protocol_argument(model_provenance_parser)
    model_provenance_parser.add_argument("--output", type=Path)

    train_parser = subparsers.add_parser(
        "train-detector",
        help="Train one or more frozen YOLO seeds.",
    )
    add_protocol_argument(train_parser)
    train_parser.add_argument("--dataset", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, action="append")
    train_parser.add_argument("--device", default="0")
    train_parser.add_argument("--workers", type=int, default=4)
    add_force_argument(train_parser)

    detect_parser = subparsers.add_parser(
        "detect",
        help="Run a trained detector and compute COCO bbox metrics.",
    )
    add_protocol_argument(detect_parser)
    detect_parser.add_argument("--dataset", type=Path, required=True)
    detect_parser.add_argument("--seed", type=int, required=True)
    detect_parser.add_argument("--split", default="test")
    detect_parser.add_argument("--device", default="0")
    add_force_argument(detect_parser)

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run SAM1/2/3 with original or YOLO bbox prompts.",
    )
    add_protocol_argument(infer_parser)
    infer_parser.add_argument("--dataset", type=Path, required=True)
    infer_parser.add_argument("--model", choices=("sam1", "sam2", "sam3"), required=True)
    infer_parser.add_argument(
        "--bbox-source",
        choices=("gt_bbox", "yolo_bbox"),
        required=True,
    )
    infer_parser.add_argument("--seed", type=int)
    infer_parser.add_argument("--split", default="test")
    infer_parser.add_argument("--device", default="0")
    add_force_argument(infer_parser)

    pseudo_parser = subparsers.add_parser(
        "build-pseudo-reference",
        help="Freeze SAM1 GT-bbox predictions as a controlled pseudo-reference.",
    )
    add_protocol_argument(pseudo_parser)
    pseudo_parser.add_argument("--dataset", type=Path, required=True)
    add_force_argument(pseudo_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate one canonical prediction set against its declared references.",
    )
    add_protocol_argument(evaluate_parser)
    evaluate_parser.add_argument("--dataset", type=Path, required=True)
    evaluate_parser.add_argument(
        "--model",
        choices=("sam1", "sam2", "sam3"),
        required=True,
    )
    evaluate_parser.add_argument(
        "--bbox-source",
        choices=("gt_bbox", "yolo_bbox"),
        required=True,
    )
    evaluate_parser.add_argument("--seed", type=int)
    evaluate_parser.add_argument("--split", default="test")
    evaluate_parser.add_argument("--bootstrap-samples", type=int)
    add_force_argument(evaluate_parser)

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Compile canonical metrics, statistics, and hashes.",
    )
    add_protocol_argument(analyze_parser)
    analyze_parser.add_argument("--bootstrap-samples", type=int)

    figures_parser = subparsers.add_parser(
        "figures",
        help="Generate the canonical matched-study figures.",
    )
    add_protocol_argument(figures_parser)

    shared_parser = subparsers.add_parser(
        "shared-reference-audit",
        help="Compare SAMRS pseudo masks with independent iSAID human masks.",
    )
    add_protocol_argument(shared_parser)
    shared_parser.add_argument("--bootstrap-samples", type=int)
    shared_parser.add_argument("--bbox-match-iou", type=float, default=0.50)
    add_force_argument(shared_parser)

    return parser


def run_script(script_name: str, arguments: list[str]) -> None:
    command = [
        sys.executable,
        str(STUDY_ROOT / "scripts" / script_name),
        *arguments,
    ]
    environment = os.environ.copy()
    environment.setdefault("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def manifest_is_completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(
            "status"
        ) == "completed"
    except (json.JSONDecodeError, OSError):
        return False


def skip_completed(
    path: Path,
    *,
    force: bool,
    manifest_path: Path | None = None,
) -> bool:
    completed = path.exists()
    if manifest_path is not None:
        completed = completed and manifest_is_completed(manifest_path)
    if completed and not force:
        print(f"SKIP completed: {path}")
        return True
    return False


def protocol_and_dataset(
    protocol_path: Path,
    dataset_path: Path,
) -> tuple[Path, MatchedStudyConfig, Path, DatasetStudyConfig]:
    resolved_protocol_path = project_path(protocol_path)
    resolved_dataset_path = project_path(dataset_path)
    return (
        resolved_protocol_path,
        load_matched_study_config(resolved_protocol_path),
        resolved_dataset_path,
        load_dataset_study_config(resolved_dataset_path),
    )


def study_root(protocol: MatchedStudyConfig) -> Path:
    if protocol.study_id != STUDY_ROOT.name:
        raise ValueError(
            f"Protocol study_id {protocol.study_id!r} does not match "
            f"study directory {STUDY_ROOT.name!r}"
        )
    return STUDY_ROOT / "results"


def run_dataset_audit(
    dataset: DatasetStudyConfig,
    output_dir: Path,
) -> DatasetAuditReport:
    profile = get_dataset_profile(dataset.profile_id)
    if profile.annotation_format == "samrs_pickle_instances":
        declared_target_id = next(
            (
                index
                for index, category in enumerate(profile.categories)
                if category.lower() == dataset.target_category.lower()
            ),
            None,
        )
        report = audit_samrs_pickle_dataset(
            root=dataset.raw_root,
            profile=profile,
            target_category=dataset.target_category,
            declared_target_id=declared_target_id,
            allow_raw_scene_overlap=True,
        )
    elif profile.annotation_format == "coco_instance_segmentation":
        report = audit_isaid_coco_dataset(
            root=dataset.raw_root,
            profile=profile,
            target_category=dataset.target_category,
        )
    else:
        raise ValueError(f"Unsupported annotation format: {profile.annotation_format}")

    dataset_output = output_dir / dataset.dataset_id
    write_audit_report(
        report,
        json_path=dataset_output / "audit.json",
        markdown_path=dataset_output / "audit.md",
    )
    return report


def audit_command(args: argparse.Namespace) -> int:
    dataset = load_dataset_study_config(project_path(args.dataset))
    output_dir = project_path(args.output_dir)
    report = run_dataset_audit(dataset, output_dir)
    print(f"{dataset.dataset_id}: {'PASS' if report.passed else 'FAIL'}")
    for finding in report.findings:
        print(f"[{finding.severity.upper()}] {finding.code}: {finding.message}")
    return 0 if report.passed else 2


def preflight_command(args: argparse.Namespace) -> int:
    protocol_path = project_path(args.protocol)
    protocol = load_matched_study_config(protocol_path)
    datasets = [
        load_dataset_study_config(project_path(path))
        for path in args.dataset
    ]
    if len(datasets) < 2:
        raise ValueError("Matched preflight requires at least two datasets")

    target_categories = {dataset.target_category.lower() for dataset in datasets}
    blocking: list[dict[str, str]] = []
    if len(target_categories) != 1:
        blocking.append(
            {
                "code": "TARGET_CATEGORY_MISMATCH",
                "message": f"Datasets use different target categories: {sorted(target_categories)}",
            }
        )

    output_path = project_path(args.output)
    audit_root = output_path.parent / "dataset_audits"
    dataset_rows = []
    for dataset in datasets:
        report = run_dataset_audit(dataset, audit_root)
        if not report.passed:
            blocking.append(
                {
                    "code": "DATASET_AUDIT_FAILED",
                    "message": f"{dataset.dataset_id} failed provenance audit",
                }
            )
        if dataset.version.strip().lower() == "unverified":
            blocking.append(
                {
                    "code": "UNVERIFIED_DATASET_VERSION",
                    "message": f"{dataset.dataset_id} is explicitly marked unverified",
                }
            )
        dataset_rows.append(
            {
                "dataset_id": dataset.dataset_id,
                "version": dataset.version,
                "target_category": dataset.target_category,
                "reference_type": dataset.reference_type.value,
                "profile_id": dataset.profile_id,
                "raw_root": str(dataset.raw_root),
                "config_hash": resolved_config_hash(protocol, dataset),
                "audit_passed": report.passed,
                "audit_findings": [asdict(finding) for finding in report.findings],
            }
        )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not blocking else "fail",
        "protocol": {
            "path": str(protocol_path),
            "study_id": protocol.study_id,
            "image_size": protocol.image_size,
            "split_fractions": protocol.split_fractions,
            "detector_seeds": protocol.detector_seeds,
            "segmenters": protocol.segmenters,
            "segmenter_configs": protocol.segmenter_configs,
            "bbox_sources": [source.value for source in protocol.bbox_sources],
            "evaluation": protocol.evaluation,
        },
        "datasets": dataset_rows,
        "blocking_findings": blocking,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Preflight: {payload['status'].upper()}")
    print(output_path)
    for finding in blocking:
        print(f"[ERROR] {finding['code']}: {finding['message']}")
    return 0 if not blocking else 2


def split_command(args: argparse.Namespace) -> int:
    candidate_path = project_path(args.candidates)
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidates = [
        SplitCandidate(
            image_id=str(row["image_id"]),
            source_scene_id=str(row["source_scene_id"]),
            stratum=str(row["stratum"]),
            instance_count=int(row.get("instance_count", 1)),
        )
        for row in payload
    ]
    rows = grouped_stratified_split(
        candidates,
        split_fractions={
            "train": args.train,
            "validation": args.validation,
            "test": args.test,
        },
        seed=args.seed,
    )
    output_dir = project_path(args.output_dir)
    write_split_manifest(
        rows,
        csv_path=output_dir / "split_manifest.csv",
        json_path=output_dir / "split_manifest.json",
    )
    print(output_dir)
    return 0


def prepare_command(args: argparse.Namespace) -> int:
    protocol_path, _, dataset_path, dataset = protocol_and_dataset(
        args.protocol,
        args.dataset,
    )
    completion_path = dataset.prepared_root / "data.yaml"
    if skip_completed(completion_path, force=args.force):
        return 0
    script_by_profile = {
        "isaid": "prepare_matched_512_from_master.py",
        "samrs_sota": "prepare_matched_512_from_master.py",
    }
    try:
        script_name = script_by_profile[dataset.profile_id]
    except KeyError as exc:
        raise ValueError(
            f"No matched preparation entry point for {dataset.profile_id}"
        ) from exc
    command_args = [
        "--protocol",
        str(protocol_path),
        "--dataset",
        str(dataset_path),
    ]
    if args.force:
        command_args.append("--force")
    run_script(script_name, command_args)
    return 0


def validate_prepared_command(args: argparse.Namespace) -> int:
    protocol_path, protocol, dataset_path, dataset = protocol_and_dataset(
        args.protocol,
        args.dataset,
    )
    output_path = (
        project_path(args.output)
        if args.output
        else study_root(protocol)
        / "audits"
        / f"{dataset.dataset_id}_prepared_dataset.json"
    )
    run_script(
        "validate_prepared_matched_dataset.py",
        [
            "--protocol",
            str(protocol_path),
            "--dataset",
            str(dataset_path),
            "--output",
            str(output_path),
        ],
    )
    return 0


def model_provenance_command(args: argparse.Namespace) -> int:
    protocol = load_matched_study_config(project_path(args.protocol))
    output_path = (
        project_path(args.output)
        if args.output is not None
        else study_root(protocol) / "audits" / "segmenter_provenance.json"
    )
    run_script(
        "audit_segmenter_provenance.py",
        [
            "--protocol",
            str(project_path(args.protocol)),
            "--output",
            str(output_path),
        ],
    )
    return 0


def train_detector_command(args: argparse.Namespace) -> int:
    protocol_path, protocol, dataset_path, dataset = protocol_and_dataset(
        args.protocol,
        args.dataset,
    )
    seeds = tuple(args.seed or protocol.detector_seeds)
    incomplete = [
        seed
        for seed in seeds
        if args.force
        or not (
            (
                study_root(protocol)
                / "detectors"
                / dataset.dataset_id
                / f"seed_{seed}"
                / "train"
                / "weights"
                / "best.pt"
            ).exists()
            and manifest_is_completed(
                study_root(protocol)
                / "detectors"
                / dataset.dataset_id
                / f"seed_{seed}"
                / "manifest.json"
            )
        )
    ]
    if not incomplete:
        print(f"SKIP completed detector seeds: {seeds}")
        return 0
    command_args = [
        "--protocol",
        str(protocol_path),
        "--dataset",
        str(dataset_path),
        "--device",
        str(args.device),
        "--workers",
        str(args.workers),
    ]
    for seed in incomplete:
        command_args.extend(["--seed", str(seed)])
    if args.force:
        command_args.append("--force")
    run_script("train_matched_detector.py", command_args)
    return 0


def detect_command(args: argparse.Namespace) -> int:
    protocol_path, protocol, dataset_path, dataset = protocol_and_dataset(
        args.protocol,
        args.dataset,
    )
    detector_root = (
        study_root(protocol)
        / "detectors"
        / dataset.dataset_id
        / f"seed_{args.seed}"
        / "evaluation"
    )
    splits = ("validation", "test") if args.split == "test" else (args.split,)
    for split in splits:
        metrics_path = detector_root / split / "metrics.json"
        threshold_path = (
            detector_root
            / "validation"
            / "selected_confidence_threshold.json"
        )
        complete = metrics_path.exists() and manifest_is_completed(
            detector_root / split / "manifest.json"
        )
        if split == "validation":
            complete = complete and threshold_path.exists()
        if complete and not args.force:
            print(f"SKIP completed output: {metrics_path}")
            continue
        command_args = [
            "--protocol",
            str(protocol_path),
            "--dataset",
            str(dataset_path),
            "--seed",
            str(args.seed),
            "--split",
            split,
            "--device",
            str(args.device),
        ]
        if args.force:
            command_args.append("--force")
        run_script("run_matched_detector.py", command_args)
    return 0


def _prediction_root(
    protocol: MatchedStudyConfig,
    dataset: DatasetStudyConfig,
    *,
    model: str,
    bbox_source: str,
    seed: int | None,
) -> Path:
    root = (
        study_root(protocol)
        / "predictions"
        / dataset.dataset_id
        / model
        / bbox_source
    )
    if bbox_source == "yolo_bbox":
        if seed is None:
            raise ValueError("--seed is required for yolo_bbox")
        root /= f"seed_{seed}"
    elif seed is not None:
        raise ValueError("--seed is only valid for yolo_bbox")
    return root


def infer_command(args: argparse.Namespace) -> int:
    protocol_path, protocol, dataset_path, dataset = protocol_and_dataset(
        args.protocol,
        args.dataset,
    )
    output_root = _prediction_root(
        protocol,
        dataset,
        model=args.model,
        bbox_source=args.bbox_source,
        seed=args.seed,
    )
    if skip_completed(
        output_root / "predictions.jsonl",
        force=args.force,
        manifest_path=output_root / "manifest.json",
    ):
        return 0
    command_args = [
        "--protocol",
        str(protocol_path),
        "--dataset",
        str(dataset_path),
        "--model",
        args.model,
        "--split",
        args.split,
    ]
    if args.bbox_source == "gt_bbox":
        script_name = "run_matched_gt_bbox_segmentation.py"
        command_args.extend(["--device", str(args.device)])
    else:
        script_name = "run_matched_yolo_bbox_segmentation.py"
        command_args.extend(
            [
                "--seed",
                str(args.seed),
                "--segmenter-device",
                str(args.device),
            ]
        )
    if args.force:
        command_args.append("--force")
    run_script(script_name, command_args)
    return 0


def build_pseudo_reference_command(args: argparse.Namespace) -> int:
    _, protocol, _, dataset = protocol_and_dataset(
        args.protocol,
        args.dataset,
    )
    if dataset.reference_type.value != "human":
        raise ValueError(
            "Controlled SAM1 pseudo-reference requires a human-reference dataset"
        )
    sam1_config = protocol.segmenter_configs["sam1"]
    predictions_path = (
        _prediction_root(
            protocol,
            dataset,
            model="sam1",
            bbox_source="gt_bbox",
            seed=None,
        )
        / "predictions.jsonl"
    )
    output_path = (
        study_root(protocol)
        / "references"
        / dataset.dataset_id
        / "sam1_gt_bbox_pseudo.jsonl"
    )
    if skip_completed(
        output_path,
        force=args.force,
        manifest_path=output_path.with_suffix(".manifest.json"),
    ):
        return 0
    command_args = [
        "--predictions",
        str(predictions_path),
        "--output",
        str(output_path),
        "--expected-model-id",
        str(sam1_config["model_id"]),
        "--expected-model-version",
        str(sam1_config["revision"]),
        "--config-hash",
        resolved_config_hash(protocol, dataset),
        "--run-id",
        f"{protocol.study_id}-{dataset.dataset_id}-sam1-pseudo-reference",
    ]
    if args.force:
        command_args.append("--force")
    run_script("build_controlled_sam1_pseudo_references.py", command_args)
    return 0


def evaluate_command(args: argparse.Namespace) -> int:
    protocol_path, protocol, _, dataset = protocol_and_dataset(
        args.protocol,
        args.dataset,
    )
    del protocol_path
    prediction_root = _prediction_root(
        protocol,
        dataset,
        model=args.model,
        bbox_source=args.bbox_source,
        seed=args.seed,
    )
    pseudo_path = (
        study_root(protocol)
        / "references"
        / dataset.dataset_id
        / "sam1_gt_bbox_pseudo.jsonl"
    )
    use_dual_reference = (
        dataset.reference_type.value == "human" and pseudo_path.exists()
    )
    mode_name = args.bbox_source
    if args.bbox_source == "yolo_bbox":
        if args.seed is None:
            raise ValueError("--seed is required for yolo_bbox")
        mode_name = f"yolo_bbox/seed_{args.seed}"
        if use_dual_reference:
            mode_name += "_dual_reference"
    elif use_dual_reference:
        mode_name = "gt_bbox_dual_reference"
    output_root = (
        study_root(protocol)
        / "evaluation"
        / dataset.dataset_id
        / args.model
        / mode_name
    )
    if skip_completed(
        output_root / "summary_instance.csv",
        force=args.force,
        manifest_path=output_root / "manifest.json",
    ):
        return 0
    command_args = [
        "--dataset-id",
        dataset.dataset_id,
        "--coco",
        str(dataset.prepared_root / args.split / "_annotations.coco.json"),
        "--coco-reference-type",
        dataset.reference_type.value,
        "--predictions",
        str(prediction_root / "predictions.jsonl"),
        "--output-dir",
        str(output_root),
        "--bootstrap-samples",
        str(args.bootstrap_samples or protocol.evaluation["bootstrap_samples"]),
        "--bootstrap-seed",
        str(protocol.split_seed),
        "--config-hash",
        resolved_config_hash(protocol, dataset),
        "--run-id",
        (
            f"{protocol.study_id}-{dataset.dataset_id}-{args.model}-"
            f"{mode_name.replace('/', '-')}-evaluation"
        ),
    ]
    if use_dual_reference:
        command_args.extend(["--pseudo-references", str(pseudo_path)])
    if args.bbox_source == "yolo_bbox":
        command_args.extend(
            [
                "--unmatched-predictions",
                str(prediction_root / "unmatched_detector_predictions.jsonl"),
            ]
        )
    run_script("evaluate_matched_predictions.py", command_args)
    return 0


def analyze_command(args: argparse.Namespace) -> int:
    protocol_path = project_path(args.protocol)
    protocol = load_matched_study_config(protocol_path)
    run_script(
        "compile_matched_study_results.py",
        [
            "--study-root",
            str(study_root(protocol)),
            "--bootstrap-samples",
            str(args.bootstrap_samples or protocol.evaluation["bootstrap_samples"]),
            "--bootstrap-seed",
            str(protocol.split_seed),
        ],
    )
    return 0


def figures_command(args: argparse.Namespace) -> int:
    protocol = load_matched_study_config(project_path(args.protocol))
    run_script(
        "generate_matched_study_figures.py",
        ["--study-root", str(study_root(protocol))],
    )
    return 0


def shared_reference_audit_command(args: argparse.Namespace) -> int:
    protocol = load_matched_study_config(project_path(args.protocol))
    output_root = study_root(protocol) / "analysis" / "shared_human_reference_audit"
    if skip_completed(
        output_root / "model_reference_inflation_ci.json",
        force=args.force,
        manifest_path=output_root / "manifest.json",
    ):
        return 0
    command_args = [
        "--study-root",
        str(study_root(protocol)),
        "--bbox-match-iou",
        str(args.bbox_match_iou),
        "--bootstrap-samples",
        str(args.bootstrap_samples or protocol.evaluation["bootstrap_samples"]),
        "--bootstrap-seed",
        str(protocol.split_seed),
    ]
    if args.force:
        command_args.append("--force")
    run_script("audit_shared_isaid_samrs_references.py", command_args)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "audit":
        return audit_command(args)
    if args.command == "preflight":
        return preflight_command(args)
    if args.command == "split":
        return split_command(args)
    if args.command == "prepare":
        return prepare_command(args)
    if args.command == "validate-prepared":
        return validate_prepared_command(args)
    if args.command == "model-provenance":
        return model_provenance_command(args)
    if args.command == "train-detector":
        return train_detector_command(args)
    if args.command == "detect":
        return detect_command(args)
    if args.command == "infer":
        return infer_command(args)
    if args.command == "build-pseudo-reference":
        return build_pseudo_reference_command(args)
    if args.command == "evaluate":
        return evaluate_command(args)
    if args.command == "analyze":
        return analyze_command(args)
    if args.command == "figures":
        return figures_command(args)
    if args.command == "shared-reference-audit":
        return shared_reference_audit_command(args)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
