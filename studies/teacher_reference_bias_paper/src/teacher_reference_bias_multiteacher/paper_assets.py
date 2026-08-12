from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

from .paths import DATASETS, MODELS, REFERENCES, REPO_ROOT, STRATA


REFERENCE_SHORT = {
    "human": "Human",
    "published_samrs_reference": "Published\nSAMRS",
    "reproduced_pseudo_sam1": "Reproduced\nSAM1",
    "pseudo_sam1": "SAM1 pseudo",
    "pseudo_sam2": "SAM2 pseudo",
    "pseudo_sam3": "SAM3 pseudo",
}
EXPERIMENT_SHORT = {
    "isaid_plane": "iSAID Plane",
    "isaid_small_vehicle": "iSAID Small Vehicle",
    "samrs_plane": "SAMRS Plane",
    "samrs_small_vehicle": "SAMRS Small Vehicle",
}
MODEL_COLORS = {"sam1": "#0072B2", "sam2": "#009E73", "sam3": "#D55E00"}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, root: Path, stem: str) -> list[Path]:
    outputs: list[Path] = []
    for extension, dpi in (("png", 260), ("pdf", 300)):
        path = root / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        outputs.append(path)
    plt.close(figure)
    return outputs


def study_design_figure(root: Path) -> list[Path]:
    configure_style()
    figure, axis = plt.subplots(figsize=(12.2, 5.0))
    axis.set_xlim(0, 12.2)
    axis.set_ylim(0, 5.0)
    axis.axis("off")
    columns = (
        (0.2, 4.05, 2.0, 0.62, "2 dataset families", "iSAID human control\nSAMRS SAM-derived labels"),
        (2.7, 4.05, 2.0, 0.62, "2 target classes", "Plane\nSmall Vehicle"),
        (5.2, 4.05, 2.0, 0.62, "Frozen predictions", "SAM1 / SAM2 / SAM3\nGT bbox + YOLO bbox"),
        (7.7, 4.05, 2.0, 0.62, "4 references", "Human/published baseline\nSAM1 / SAM2 / SAM3 pseudo"),
        (10.2, 4.05, 1.8, 0.62, "Evaluation", "Instance metrics\npaired deltas"),
    )
    for x, y, width, height, title, subtitle in columns:
        axis.add_patch(
            Rectangle((x, y), width, height, facecolor="#EAF2F8", edgecolor="#1F4E79", linewidth=1.2)
        )
        axis.text(x + width / 2, y + height * 0.67, title, ha="center", va="center", fontweight="bold")
        axis.text(x + width / 2, y - 0.15, subtitle, ha="center", va="top", fontsize=8)
    for left, right in zip(columns[:-1], columns[1:], strict=True):
        axis.add_patch(
            FancyArrowPatch(
                (left[0] + left[2] + 0.03, 4.36),
                (right[0] - 0.03, 4.36),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#444444",
                linewidth=1.0,
            )
        )
    rows = (
        ("iSAID Plane", "5,447 instances", "Independent human reference"),
        ("iSAID Small Vehicle", "12,051 instances", "Independent human reference"),
        ("SAMRS Plane", "3,713 instances", "Published SAM-derived reference"),
        ("SAMRS Small Vehicle", "7,659 instances", "Published SAM-derived reference"),
    )
    for index, row in enumerate(rows):
        y = 2.72 - index * 0.59
        axis.add_patch(Rectangle((0.7, y), 10.8, 0.44, facecolor="#F8F9F9", edgecolor="#B0B0B0", linewidth=0.7))
        axis.text(1.0, y + 0.22, row[0], va="center", fontweight="bold")
        axis.text(4.25, y + 0.22, row[1], va="center")
        axis.text(6.5, y + 0.22, "512 images = 4 × 128 strata", va="center")
        axis.text(9.0, y + 0.22, row[2], va="center", fontsize=8)
    axis.text(
        6.1,
        0.18,
        "Key control: predictions are frozen; only the evaluation reference changes. GT self-reference diagonal is an identity control, not performance.",
        ha="center",
        fontsize=8.5,
        fontweight="bold",
    )
    return save_figure(figure, root, "figure_1_study_design")


def combined_matrix_figure(root: Path) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.1), squeeze=False)
    image = None
    for axis, (experiment_id, source) in zip(axes.flat, DATASETS.items(), strict=True):
        aggregates = pd.read_csv(source.analysis_root / "aggregate_metrics.csv")
        selected = aggregates[
            (aggregates["stratum"] == "overall")
            & (aggregates["bbox_source"] == "yolo_bbox")
        ]
        pivot = (
            selected.pivot(index="model", columns="reference_type", values="mean_iou")
            .loc[list(MODELS), list(source.reference_types)]
        )
        image = axis.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1)
        for y in range(3):
            for x in range(4):
                value = float(pivot.iloc[y, x])
                axis.text(
                    x,
                    y,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.32 or value > 0.82 else "black",
                    fontweight="bold",
                )
        axis.set_xticks(
            range(4),
            [REFERENCE_SHORT[name] for name in source.reference_types],
            rotation=18,
            ha="right",
        )
        axis.set_yticks(range(3), [model.upper() for model in MODELS])
        axis.set_title(EXPERIMENT_SHORT[experiment_id])
        axis.set_xlabel("Evaluation reference")
        axis.set_ylabel("Evaluated model")
    figure.suptitle("YOLO-bbox Overall Avg IoU: model–reference matrices", fontsize=13, fontweight="bold")
    figure.subplots_adjust(left=0.07, right=0.91, bottom=0.12, top=0.90, wspace=0.28, hspace=0.55)
    color_axis = figure.add_axes((0.93, 0.16, 0.014, 0.68))
    figure.colorbar(image, cax=color_axis, label="Avg IoU")
    return save_figure(figure, root, "figure_2_model_reference_iou_matrix")


def isaid_effect_figure(root: Path) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.1), sharey=True)
    for axis, experiment_id in zip(
        axes, ("isaid_plane", "isaid_small_vehicle"), strict=True
    ):
        source = DATASETS[experiment_id]
        effects = pd.read_csv(source.analysis_root / "paired_reference_effects.csv")
        selected = effects[effects["bbox_source"] == "yolo_bbox"]
        x = np.arange(3)
        offsets = {"pseudo_sam1": -0.20, "pseudo_sam2": 0.0, "pseudo_sam3": 0.20}
        for reference_type, teacher in zip(
            ("pseudo_sam1", "pseudo_sam2", "pseudo_sam3"), MODELS, strict=True
        ):
            rows = selected[selected["comparison_reference"] == reference_type].set_index("model").loc[list(MODELS)]
            values = rows["delta_iou"].to_numpy(float)
            errors = np.vstack(
                (
                    values - rows["delta_ci_lower"].to_numpy(float),
                    rows["delta_ci_upper"].to_numpy(float) - values,
                )
            )
            axis.errorbar(
                x + offsets[reference_type],
                values,
                yerr=errors,
                fmt="o",
                capsize=3,
                color=MODEL_COLORS[teacher],
                label=f"{teacher.upper()} pseudo",
            )
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_xticks(x, [model.upper() for model in MODELS])
        axis.set_title(EXPERIMENT_SHORT[experiment_id])
        axis.set_xlabel("Evaluated model")
        axis.set_ylabel("Pseudo − human Avg IoU")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("iSAID human-controlled reference effects (95% scene-clustered CI)", fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return save_figure(figure, root, "figure_3_isaid_reference_effect_with_ci")


def samrs_integrity_figure(root: Path) -> list[Path]:
    configure_style()
    experiments = ("samrs_plane", "samrs_small_vehicle")
    agreement_values: list[float] = []
    yolo_gaps: list[float] = []
    for experiment_id in experiments:
        source = DATASETS[experiment_id]
        agreement = pd.read_csv(source.analysis_root / "reference_agreement.csv")
        row = agreement[
            (agreement["reference_a"] == "published_samrs_reference")
            & (agreement["reference_b"] == "reproduced_pseudo_sam1")
        ].iloc[0]
        agreement_values.append(float(row["mean_instance_iou"]))
        aggregates = pd.read_csv(source.analysis_root / "aggregate_metrics.csv")
        overall = aggregates[
            (aggregates["stratum"] == "overall")
            & (aggregates["bbox_source"] == "yolo_bbox")
            & (aggregates["model"] == "sam1")
        ].set_index("reference_type")
        yolo_gaps.append(
            float(overall.loc["reproduced_pseudo_sam1", "mean_iou"])
            - float(overall.loc["published_samrs_reference", "mean_iou"])
        )
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    labels = [EXPERIMENT_SHORT[name] for name in experiments]
    bars = axes[0].bar(labels, agreement_values, color="#0072B2", width=0.55)
    axes[0].set_ylim(0.94, 1.002)
    axes[0].set_ylabel("Mean instance IoU")
    axes[0].set_title("Published SAMRS vs reproduced SAM1")
    for bar, value in zip(bars, agreement_values, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.3f}", ha="center", fontweight="bold")
    bars = axes[1].bar(labels, yolo_gaps, color="#009E73", width=0.55)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_ylabel("Reproduced − published Avg IoU")
    axes[1].set_title("SAM1 YOLO-bbox score gap")
    for bar, value in zip(bars, yolo_gaps, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.0003, f"{value:+.3f}", ha="center", fontweight="bold")
    figure.suptitle("SAMRS reference integrity: published labels are SAM1-like, not human GT", fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return save_figure(figure, root, "figure_4_samrs_reference_integrity")


def stratified_affinity_figure(root: Path) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 7.2), sharey=True)
    for axis, (experiment_id, source) in zip(axes.flat, DATASETS.items(), strict=True):
        aggregates = pd.read_csv(source.analysis_root / "aggregate_metrics.csv")
        baseline = source.reference_types[0]
        own_reference = {
            "sam1": "pseudo_sam1" if source.dataset_family == "isaid" else "reproduced_pseudo_sam1",
            "sam2": "pseudo_sam2",
            "sam3": "pseudo_sam3",
        }
        selected = aggregates[aggregates["bbox_source"] == "yolo_bbox"]
        x = np.arange(4)
        for model_index, model in enumerate(MODELS):
            values: list[float] = []
            for stratum in STRATA[1:]:
                rows = selected[(selected["model"] == model) & (selected["stratum"] == stratum)].set_index("reference_type")
                values.append(
                    float(rows.loc[own_reference[model], "mean_iou"])
                    - float(rows.loc[baseline, "mean_iou"])
                )
            axis.plot(x, values, marker="o", color=MODEL_COLORS[model], label=model.upper())
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_xticks(x, ["No/Low", "No/High", "Overlap/Low", "Overlap/High"], rotation=16)
        axis.set_title(EXPERIMENT_SHORT[experiment_id])
        axis.set_ylabel("Own-reference − baseline Avg IoU")
        axis.grid(axis="y", alpha=0.22)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("YOLO-bbox own-reference effect across overlap × mask-area strata", fontsize=12, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return save_figure(figure, root, "figure_5_stratified_self_reference_effect")


def design_table() -> pd.DataFrame:
    rows = []
    for experiment_id, source in DATASETS.items():
        rows.append(
            {
                "Experiment": EXPERIMENT_SHORT[experiment_id],
                "Target": source.target_label,
                "Images": 512,
                "Instances": source.instance_count,
                "Strata": "4 × 128",
                "Baseline reference": REFERENCE_SHORT[source.reference_types[0]].replace("\n", " "),
                "Independent human": "yes" if source.dataset_family == "isaid" else "no",
            }
        )
    return pd.DataFrame(rows)


def control_table() -> pd.DataFrame:
    rows = []
    for experiment_id, source in DATASETS.items():
        aggregates = pd.read_csv(source.analysis_root / "aggregate_metrics.csv")
        baseline = source.reference_types[0]
        selected = aggregates[
            (aggregates["stratum"] == "overall")
            & (aggregates["bbox_source"] == "yolo_bbox")
            & (aggregates["reference_type"] == baseline)
        ]
        for _, row in selected.iterrows():
            rows.append(
                {
                    "Experiment": EXPERIMENT_SHORT[experiment_id],
                    "Model": str(row["model"]).upper(),
                    "Baseline reference": REFERENCE_SHORT[baseline].replace("\n", " "),
                    "Avg IoU": float(row["mean_iou"]),
                    "Avg Dice": float(row["mean_dice"]),
                    "Avg Precision": float(row["mean_precision"]),
                    "Avg Recall": float(row["mean_recall"]),
                }
            )
    return pd.DataFrame(rows)


def own_effect_table() -> pd.DataFrame:
    rows = []
    for experiment_id, source in DATASETS.items():
        effects = pd.read_csv(source.analysis_root / "paired_reference_effects.csv")
        own_reference = {
            "sam1": "pseudo_sam1" if source.dataset_family == "isaid" else "reproduced_pseudo_sam1",
            "sam2": "pseudo_sam2",
            "sam3": "pseudo_sam3",
        }
        selected = effects[effects["bbox_source"] == "yolo_bbox"]
        for model in MODELS:
            row = selected[
                (selected["model"] == model)
                & (selected["comparison_reference"] == own_reference[model])
            ].iloc[0]
            rows.append(
                {
                    "Experiment": EXPERIMENT_SHORT[experiment_id],
                    "Model": model.upper(),
                    "Baseline IoU": float(row["baseline_mean_iou"]),
                    "Own-reference IoU": float(row["comparison_mean_iou"]),
                    "Delta": float(row["delta_iou"]),
                    "CI lower": float(row["delta_ci_lower"]),
                    "CI upper": float(row["delta_ci_upper"]),
                }
            )
    return pd.DataFrame(rows)


def samrs_integrity_table() -> pd.DataFrame:
    rows = []
    for experiment_id in ("samrs_plane", "samrs_small_vehicle"):
        source = DATASETS[experiment_id]
        agreement = pd.read_csv(source.analysis_root / "reference_agreement.csv")
        row = agreement[
            (agreement["reference_a"] == "published_samrs_reference")
            & (agreement["reference_b"] == "reproduced_pseudo_sam1")
        ].iloc[0]
        rows.append(
            {
                "Experiment": EXPERIMENT_SHORT[experiment_id],
                "Published vs reproduced SAM1 IoU": float(row["mean_instance_iou"]),
                "Instances": int(row["instance_count"]),
                "Interpretation": "SAM1-like published reference; not human GT",
            }
        )
    return pd.DataFrame(rows)


def detector_table() -> pd.DataFrame:
    frames = []
    for experiment_id, source in DATASETS.items():
        row = pd.read_csv(source.analysis_root / "detector_summary.csv").iloc[0]
        frames.append(
            {
                "Experiment": EXPERIMENT_SHORT[experiment_id],
                "Seed": 42,
                "BBox mAP50": float(row["bbox_AP50_mean"]),
                "BBox mAP75": float(row["bbox_AP75_mean"]),
                "BBox mAP90": float(row["bbox_AP90_mean"]),
                "BBox mAP50-95": float(row["bbox_AP50_95_mean"]),
                "Precision@0.50": float(row["precision_at_bbox_iou50_mean"]),
                "Recall@0.50": float(row["recall_at_bbox_iou50_mean"]),
            }
        )
    return pd.DataFrame(frames)


def strata_table() -> pd.DataFrame:
    rows = []
    for experiment_id, source in DATASETS.items():
        aggregates = pd.read_csv(source.analysis_root / "aggregate_metrics.csv")
        baseline = source.reference_types[0]
        own_reference = {
            "sam1": "pseudo_sam1" if source.dataset_family == "isaid" else "reproduced_pseudo_sam1",
            "sam2": "pseudo_sam2",
            "sam3": "pseudo_sam3",
        }
        selected = aggregates[aggregates["bbox_source"] == "yolo_bbox"]
        for stratum in STRATA[1:]:
            for model in MODELS:
                condition = selected[(selected["stratum"] == stratum) & (selected["model"] == model)].set_index("reference_type")
                rows.append(
                    {
                        "Experiment": EXPERIMENT_SHORT[experiment_id],
                        "Stratum": stratum,
                        "Model": model.upper(),
                        "Baseline IoU": float(condition.loc[baseline, "mean_iou"]),
                        "Own-reference IoU": float(condition.loc[own_reference[model], "mean_iou"]),
                        "Delta": float(condition.loc[own_reference[model], "mean_iou"] - condition.loc[baseline, "mean_iou"]),
                    }
                )
    return pd.DataFrame(rows)


def _latex_escape(value: object) -> str:
    text = str(value)
    for source, target in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(source, target)
    return text


def write_table(frame: pd.DataFrame, root: Path, stem: str) -> list[Path]:
    csv_path = root / f"{stem}.csv"
    tex_path = root / f"{stem}.tex"
    frame.to_csv(csv_path, index=False, float_format="%.6f")
    headers = " & ".join(_latex_escape(column) for column in frame.columns)
    rows = [
        " & ".join(_latex_escape(value) for value in row) + r" \\"
        for row in frame.itertuples(index=False, name=None)
    ]
    tex_path.write_text(
        "\n".join(
            [
                r"\begin{tabular}{" + "l" * len(frame.columns) + "}",
                r"\toprule",
                headers + r" \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return [csv_path, tex_path]


def generate_assets(study_root: Path) -> dict[str, list[Path]]:
    assets_root = study_root / "paper_writing" / "assets"
    figures_root = assets_root / "figures"
    tables_root = assets_root / "tables"
    figures_root.mkdir(parents=True, exist_ok=True)
    tables_root.mkdir(parents=True, exist_ok=True)
    for path in (*figures_root.glob("*"), *tables_root.glob("*")):
        if path.is_file():
            path.unlink()
    figures = [
        *study_design_figure(figures_root),
        *combined_matrix_figure(figures_root),
        *isaid_effect_figure(figures_root),
        *samrs_integrity_figure(figures_root),
        *stratified_affinity_figure(figures_root),
    ]
    tables = [
        *write_table(design_table(), tables_root, "table_1_experimental_design"),
        *write_table(control_table(), tables_root, "table_2_baseline_reference_results"),
        *write_table(own_effect_table(), tables_root, "table_3_own_reference_effect"),
        *write_table(samrs_integrity_table(), tables_root, "table_4_samrs_reference_integrity"),
        *write_table(detector_table(), tables_root, "table_5_detector_control"),
        *write_table(strata_table(), tables_root, "table_s1_stratified_reference_effect"),
    ]
    inputs = [
        source.analysis_root / filename
        for source in DATASETS.values()
        for filename in (
            "aggregate_metrics.csv",
            "paired_reference_effects.csv",
            "reference_agreement.csv",
            "detector_summary.csv",
        )
    ]

    def record(path: Path) -> dict[str, object]:
        return {
            "path": path.resolve().relative_to(REPO_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest_path = assets_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "status": "completed",
                "scope": "four_experiments_no_cross_dataset_pooling",
                "inputs": [record(path) for path in inputs],
                "outputs": [record(path) for path in (*figures, *tables)],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"figures": figures, "tables": tables, "manifest": [manifest_path]}
