from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
EXPERIMENT_CONFIGS = {
    name: STUDY_ROOT / "experiments" / name / "config.yaml"
    for name in (
        "isaid_plane",
        "isaid_small_vehicle",
        "samrs_plane",
        "samrs_small_vehicle",
    )
}
MASTER_CONFIGS = {
    name: STUDY_ROOT / "experiments" / name / "master_config.yaml"
    for name in EXPERIMENT_CONFIGS
}
PROTOCOLS = {
    "canonical": STUDY_ROOT / "configs" / "protocol.yaml",
    "local_8gb": STUDY_ROOT / "configs" / "protocol.local_8gb.yaml",
}


def run_script(name: str, arguments: list[str]) -> None:
    subprocess.run(
        [sys.executable, str(STUDY_ROOT / "scripts" / name), *arguments],
        cwd=REPO_ROOT,
        check=True,
    )


def dataset_arguments(experiment: str, protocol: Path) -> list[str]:
    return [
        "--protocol",
        str(protocol),
        "--dataset",
        str(EXPERIMENT_CONFIGS[experiment]),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Teacher-reference-bias paper study kanonik CLI."
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROTOCOLS),
        default="canonical",
        help="canonical veya 8 GB VRAM için local_8gb protokolü.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_master = commands.add_parser("prepare-master")
    prepare_master.add_argument(
        "--experiment", choices=tuple(EXPERIMENT_CONFIGS), required=True
    )
    prepare_master.add_argument("--force", action="store_true")

    prepare_matched = commands.add_parser("prepare-matched")
    prepare_matched.add_argument(
        "--experiment", choices=tuple(EXPERIMENT_CONFIGS), required=True
    )
    prepare_matched.add_argument("--force", action="store_true")

    validate_data = commands.add_parser("validate-data")
    validate_data.add_argument("--experiment", choices=tuple(EXPERIMENT_CONFIGS))

    train = commands.add_parser("train-detector")
    train.add_argument("--experiment", choices=tuple(EXPERIMENT_CONFIGS), required=True)
    train.add_argument("--device", default="0")
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--force", action="store_true")

    detect = commands.add_parser("detect")
    detect.add_argument("--experiment", choices=tuple(EXPERIMENT_CONFIGS), required=True)
    detect.add_argument("--device", default="0")
    detect.add_argument("--force", action="store_true")

    infer = commands.add_parser("infer")
    infer.add_argument("--experiment", choices=tuple(EXPERIMENT_CONFIGS), required=True)
    infer.add_argument("--model", choices=("sam1", "sam2", "sam3"), required=True)
    infer.add_argument("--bbox-source", choices=("gt_bbox", "yolo_bbox"), required=True)
    infer.add_argument("--device", default="0")
    infer.add_argument("--force", action="store_true")

    commands.add_parser("references")
    commands.add_parser("evaluate")
    commands.add_parser("reports")
    commands.add_parser("validate")
    commands.add_parser("postprocess")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol = PROTOCOLS[args.profile]
    if args.command == "prepare-master":
        script = (
            "prepare_master_isaid.py"
            if args.experiment.startswith("isaid_")
            else "prepare_master_samrs.py"
        )
        options = [
            "--protocol",
            str(protocol),
            "--dataset",
            str(MASTER_CONFIGS[args.experiment]),
        ]
        if args.force:
            options.append("--force")
        run_script(script, options)
        return
    if args.command == "prepare-matched":
        options = dataset_arguments(args.experiment, protocol)
        if args.force:
            options.append("--force")
        run_script("prepare_matched_512_from_master.py", options)
        return
    if args.command == "validate-data":
        experiments = (
            (args.experiment,) if args.experiment else tuple(EXPERIMENT_CONFIGS)
        )
        for experiment in experiments:
            run_script(
                "validate_prepared_matched_dataset.py",
                dataset_arguments(experiment, protocol),
            )
        return
    if args.command == "train-detector":
        options = dataset_arguments(args.experiment, protocol) + [
            "--seed",
            "42",
            "--device",
            args.device,
            "--workers",
            str(args.workers),
        ]
        if args.resume:
            options.append("--resume")
        if args.force:
            options.append("--force")
        run_script("train_matched_detector.py", options)
        return
    if args.command == "detect":
        options = dataset_arguments(args.experiment, protocol) + [
            "--seed",
            "42",
            "--device",
            args.device,
        ]
        if args.force:
            options.append("--force")
        run_script("run_matched_detector.py", options)
        return
    if args.command == "infer":
        script = (
            "run_matched_gt_bbox_segmentation.py"
            if args.bbox_source == "gt_bbox"
            else "run_matched_yolo_bbox_segmentation.py"
        )
        device_argument = (
            "--device" if args.bbox_source == "gt_bbox" else "--segmenter-device"
        )
        options = dataset_arguments(args.experiment, protocol) + [
            "--model",
            args.model,
            device_argument,
            args.device,
        ]
        if args.bbox_source == "yolo_bbox":
            options.extend(("--seed", "42"))
        if args.force:
            options.append("--force")
        run_script(script, options)
        return
    if args.command == "references":
        run_script("build_references.py", [])
        return
    if args.command == "evaluate":
        run_script("evaluate_reference_cubes.py", [])
        run_script("compile_experiment_analyses.py", [])
        return
    if args.command in {"reports", "postprocess"}:
        run_script("generate_experiment_figures.py", [])
        run_script("write_full_metric_reports.py", [])
        run_script("write_cross_analysis_reports.py", [])
        run_script("generate_paper_assets.py", [])
        if args.command == "reports":
            return
    if args.command in {"validate", "postprocess"}:
        run_script("validate_paper_study.py", [])


if __name__ == "__main__":
    main()
