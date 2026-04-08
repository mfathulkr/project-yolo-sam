from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pool_segmentation_compare.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export key experiment assets for presentation use.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "experiment.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "presentation_sam3_bbox_study",
    )
    return parser.parse_args()


def bootstrap_ci(diff: np.ndarray, seed: int = 0, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, size=(n_boot, len(diff)), replace=True)
    means = samples.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_triplet_metrics(
    triplet_df: pd.DataFrame,
    negative_images: list[str],
    mask_dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pretty = {
        "iou_text_sam3": "SAM3 text-only",
        "iou_yolo_sam3": "YOLO + SAM3",
        "iou_gt_box_sam3": "GT bbox + SAM3",
    }

    summary_rows: list[dict[str, object]] = []
    for column, label in pretty.items():
        scores = triplet_df[column]
        positive_nonempty = 0
        for image_name in triplet_df["image"]:
            mask = np.array(Image.open(mask_dirs[column] / f"{Path(image_name).stem}.png"))
            positive_nonempty += int((mask > 0).any())

        negative_fp = 0
        for image_name in negative_images:
            mask = np.array(Image.open(mask_dirs[column] / f"{Path(image_name).stem}.png"))
            negative_fp += int((mask > 0).any())

        summary_rows.append(
            {
                "pipeline": label,
                "mean_iou": float(scores.mean()),
                "median_iou": float(scores.median()),
                "zero_iou_images": int((scores == 0).sum()),
                "iou_ge_0_50": int((scores >= 0.5).sum()),
                "iou_ge_0_75": int((scores >= 0.75).sum()),
                "iou_ge_0_90": int((scores >= 0.9).sum()),
                "positive_nonempty_images": positive_nonempty,
                "negative_false_positive_images": negative_fp,
            }
        )

    pair_rows: list[dict[str, object]] = []
    pair_specs = [
        ("iou_yolo_sam3", "iou_text_sam3"),
        ("iou_gt_box_sam3", "iou_yolo_sam3"),
        ("iou_gt_box_sam3", "iou_text_sam3"),
    ]
    for left, right in pair_specs:
        diff = (triplet_df[left] - triplet_df[right]).to_numpy()
        low, high = bootstrap_ci(diff)
        pair_rows.append(
            {
                "comparison": f"{pretty[left]} - {pretty[right]}",
                "mean_difference": float(diff.mean()),
                "ci95_low": low,
                "ci95_high": high,
                "left_wins": int((triplet_df[left] > triplet_df[right]).sum()),
                "ties": int((triplet_df[left] == triplet_df[right]).sum()),
                "right_wins": int((triplet_df[right] > triplet_df[left]).sum()),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(pair_rows)


def summarize_yolo_runs(old_results: Path, new_results: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, path in [
        ("YOLOv8n 512x512 50ep", old_results),
        ("YOLOv8s 640x640 100ep", new_results),
    ]:
        df = pd.read_csv(path)
        best_map50 = df.loc[df["metrics/mAP50(B)"].idxmax()]
        best_map5095 = df.loc[df["metrics/mAP50-95(B)"].idxmax()]
        best_recall = df.loc[df["metrics/recall(B)"].idxmax()]
        rows.append(
            {
                "run": label,
                "duration_sec": float(df["time"].iloc[-1]),
                "best_mAP50": float(best_map50["metrics/mAP50(B)"]),
                "best_mAP50_95": float(best_map5095["metrics/mAP50-95(B)"]),
                "best_recall": float(best_recall["metrics/recall(B)"]),
                "best_precision": float(df["metrics/precision(B)"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_mean_iou(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary_df["pipeline"], summary_df["mean_iou"], color=["#3b82f6", "#f59e0b", "#10b981"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean IoU")
    ax.set_title("Triplet Comparison on Positive Validation Tiles")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_thresholds(summary_df: pd.DataFrame, output_path: Path) -> None:
    labels = summary_df["pipeline"].tolist()
    x = np.arange(len(labels))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, summary_df["iou_ge_0_50"], width, label="IoU >= 0.50")
    ax.bar(x, summary_df["iou_ge_0_75"], width, label="IoU >= 0.75")
    ax.bar(x + width, summary_df["iou_ge_0_90"], width, label="IoU >= 0.90")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Image Count")
    ax.set_title("High-IoU Counts")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_failures(summary_df: pd.DataFrame, output_path: Path) -> None:
    labels = summary_df["pipeline"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, summary_df["zero_iou_images"], width, label="Zero-IoU positives")
    ax.bar(x + width / 2, summary_df["negative_false_positive_images"], width, label="Negative FP images")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Image Count")
    ax.set_title("Failure Profile")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_yolo_runs(yolo_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(yolo_df))
    width = 0.25
    ax.bar(x - width, yolo_df["best_mAP50"], width, label="best mAP50")
    ax.bar(x, yolo_df["best_mAP50_95"], width, label="best mAP50-95")
    ax.bar(x + width, yolo_df["best_recall"], width, label="best recall")
    ax.set_xticks(x)
    ax.set_xticklabels(yolo_df["run"], rotation=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("YOLO Detector Improvement")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_takeaways(summary_df: pd.DataFrame, pairwise_df: pd.DataFrame, yolo_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Slide Takeaways",
        "",
        "## Core result",
        "",
        "- `GT bbox + SAM3` is the clear upper bound and best-performing pipeline.",
        "- `YOLO + SAM3` improves over `SAM3 text-only`, but the gain is modest.",
        "- The remaining gap between `YOLO + SAM3` and `GT bbox + SAM3` points to localization quality, not SAM3 mask quality alone.",
        "",
        "## Key numbers",
        "",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['pipeline']}: mean IoU `{row['mean_iou']:.4f}`, "
            f"zero-IoU `{int(row['zero_iou_images'])}`, "
            f"IoU>=0.75 `{int(row['iou_ge_0_75'])}`"
        )

    lines.extend(
        [
            "",
            "## Pairwise interpretation",
            "",
        ]
    )
    for _, row in pairwise_df.iterrows():
        lines.append(
            f"- {row['comparison']}: mean diff `{row['mean_difference']:.4f}`, "
            f"95% CI `[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]`"
        )

    lines.extend(
        [
            "",
            "## Detector message",
            "",
            f"- Strong YOLO run reached best mAP50 `{yolo_df.iloc[1]['best_mAP50']:.4f}` and best recall `{yolo_df.iloc[1]['best_recall']:.4f}`.",
            "- Because `YOLO + SAM3` still trails `GT bbox + SAM3`, the experiment supports the claim that box quality matters.",
            "",
            "## Presentation angle",
            "",
            "- SAM3 text-only is not weak; it can localize from text alone.",
            "- But explicit geometry still matters for hard or ambiguous cases.",
            "- The experiment directly supports the argument that bounding-box prompting remains important even with SAM3.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    out_root = args.output_dir
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    notes_dir = out_root / "notes"
    literature_dir = out_root / "literature"
    cases_dir = figures_dir / "sample_cases"
    for directory in [tables_dir, figures_dir, notes_dir, literature_dir, cases_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    triplet_dir = resolve_path(config["paths"]["sam3_triplet_metrics_dir"])
    triplet_df = pd.read_csv(triplet_dir / "per_image_iou_sam3_triplet.csv")

    val_images = sorted((resolve_path(config["paths"]["prepared_dataset_dir"]) / config["dataset"]["eval_split"] / "images").glob("*.jpg"))
    positive_images = set(triplet_df["image"])
    negative_images = [path.name for path in val_images if path.name not in positive_images]

    mask_dirs = {
        "iou_text_sam3": resolve_path(config["paths"]["sam3_text_output_dir"]) / "masks",
        "iou_yolo_sam3": resolve_path(config["paths"]["yolo_sam3_output_dir"]) / "masks",
        "iou_gt_box_sam3": resolve_path(config["paths"]["gt_box_sam3_output_dir"]) / "masks",
    }

    summary_df, pairwise_df = summarize_triplet_metrics(triplet_df, negative_images, mask_dirs)
    yolo_df = summarize_yolo_runs(
        ROOT / "runs" / "yolo_building" / "train" / "results.csv",
        ROOT / "runs" / "yolo_building_s640" / "train" / "results.csv",
    )

    summary_df.to_csv(tables_dir / "triplet_summary_slide_ready.csv", index=False)
    pairwise_df.to_csv(tables_dir / "pairwise_differences.csv", index=False)
    yolo_df.to_csv(tables_dir / "yolo_detector_comparison.csv", index=False)
    shutil.copy2(triplet_dir / "summary_sam3_triplet.csv", tables_dir / "summary_sam3_triplet.csv")
    shutil.copy2(triplet_dir / "per_image_iou_sam3_triplet.csv", tables_dir / "per_image_iou_sam3_triplet.csv")

    plot_mean_iou(summary_df, figures_dir / "mean_iou_triplet.png")
    plot_thresholds(summary_df, figures_dir / "high_iou_counts.png")
    plot_failures(summary_df, figures_dir / "failure_profile.png")
    plot_yolo_runs(yolo_df, figures_dir / "yolo_detector_comparison.png")

    vis_dir = resolve_path(config["paths"]["sam3_triplet_visualizations_dir"])
    for image_path in sorted(vis_dir.glob("*.png"))[:12]:
        shutil.copy2(image_path, cases_dir / image_path.name)

    write_takeaways(summary_df, pairwise_df, yolo_df, notes_dir / "slide_takeaways.md")

    readme = [
        "# Presentation Workspace",
        "",
        "This folder mirrors the most important outputs from the code repository for slide preparation.",
        "",
        "## Contents",
        "",
        "- `tables/`: slide-ready CSV tables and raw experiment summaries",
        "- `figures/`: bar charts and copied qualitative examples",
        "- `notes/slide_takeaways.md`: short speaking points based on local results",
        "- `literature/`: literature notes and missing-paper tracking",
        "",
        "Generated from:",
        f"- `{ROOT}`",
        "",
    ]
    (out_root / "README.md").write_text("\n".join(readme), encoding="utf-8")


if __name__ == "__main__":
    main()
