from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sam3_bbox_study.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export stratified iSAID experiment assets for presentation use.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "isaid_vehicle_yolo26x_cpu_eval.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "presentation_isaid_vehicle_sam3_study",
    )
    return parser.parse_args()


def plot_stratified_metric(summary: pd.DataFrame, metric: str, output_path: Path) -> None:
    pivot = summary.pivot(index="stratum", columns="pipeline_label", values=metric)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#252525", "#666666", "#9a9a9a", "#c7c7c7", "#e2e2e2"]
    pivot.plot(kind="bar", ax=ax, color=colors[: len(pivot.columns)])
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("")
    ax.set_title(f"{metric.replace('_', ' ').title()} by Bbox-Overlap / Mask-Area Stratum")
    ax.grid(axis="y", alpha=0.25)
    ax.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_takeaways(overall: pd.DataFrame, by_stratum: pd.DataFrame, pairwise: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Stratified iSAID Vehicle Study",
        "",
        "## Setup",
        "",
        "- Dataset: iSAID overhead urban imagery, filtered to Small_Vehicle and Large_Vehicle as one `vehicle` class.",
        "- Evaluation split is balanced across bbox-overlap and target-mask-area strata.",
        "- Pipelines: `SAM3 text-only`, `YOLO26x + SAM3`, `GT bbox + SAM3`, and `YOLO26x + SAM2`.",
        "",
        "## Overall metrics",
        "",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"- {row['pipeline_label']}: mean IoU `{row['mean_iou']:.4f}`, "
            f"mean Dice `{row['mean_dice']:.4f}`, mean recall `{row['mean_recall']:.4f}`"
        )

    lines.extend(["", "## Stratified message", ""])
    best_rows = by_stratum.sort_values(["stratum", "mean_iou"], ascending=[True, False]).groupby("stratum").head(1)
    for _, row in best_rows.iterrows():
        lines.append(f"- `{row['stratum']}` best mean IoU: {row['pipeline_label']} `{row['mean_iou']:.4f}`")

    if not pairwise.empty:
        lines.extend(["", "## Pairwise deltas", ""])
        for _, row in pairwise.iterrows():
            lines.append(
                f"- `{row['stratum']}` {row['comparison']}: mean diff `{row['mean_difference']:.4f}` "
                f"95% CI `[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]`"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics_dir = resolve_path(config["paths"]["sam3_triplet_metrics_dir"])

    overall = pd.read_csv(metrics_dir / "summary_overall_stratified.csv")
    by_stratum = pd.read_csv(metrics_dir / "summary_by_stratum.csv")
    pairwise = pd.read_csv(metrics_dir / "pairwise_iou_by_stratum.csv")

    out_root = args.output_dir
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    notes_dir = out_root / "notes"
    cases_dir = figures_dir / "sample_cases"
    for directory in [tables_dir, figures_dir, notes_dir, cases_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    for file_name in [
        "per_image_stratified_metrics.csv",
        "summary_overall_stratified.csv",
        "summary_by_stratum.csv",
        "pairwise_iou_by_stratum.csv",
    ]:
        shutil.copy2(metrics_dir / file_name, tables_dir / file_name)

    plot_stratified_metric(by_stratum, "mean_iou", figures_dir / "mean_iou_by_stratum.png")
    plot_stratified_metric(by_stratum, "mean_dice", figures_dir / "mean_dice_by_stratum.png")
    write_takeaways(overall, by_stratum, pairwise, notes_dir / "slide_takeaways.md")

    vis_dir = resolve_path(config["paths"]["sam3_triplet_visualizations_dir"])
    if vis_dir.exists():
        for image_path in sorted(vis_dir.glob("*.png")):
            shutil.copy2(image_path, cases_dir / image_path.name)

    (out_root / "README.md").write_text(
        "\n".join(
            [
                "# Stratified Aerial Object Presentation Workspace",
                "",
                "- `tables/`: stratified CSV metrics",
                "- `figures/`: metric plots and qualitative examples",
                "- `notes/slide_takeaways.md`: short speaking points",
                "",
                f"Generated from `{ROOT}`.",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
