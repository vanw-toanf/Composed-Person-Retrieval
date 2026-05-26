#!/usr/bin/env python
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


IMAGE_CORRUPTIONS = {
    "blur",
    "gaussian_noise",
    "jpeg_compression",
    "brightness_contrast",
    "random_occlusion",
}
TEXT_PERTURBATIONS = {
    "synonym",
    "paraphrase",
    "typo",
    "word_deletion",
    "color_swap",
    "object_swap",
}
MODALITY_CONFLICTS = {
    "wrong_text",
    "wrong_image",
    "noisy_text",
}
METRICS = ["R@1", "R@5", "R@10", "R@50"]


def group_of(corruption_type: str) -> str:
    if corruption_type == "clean":
        return "clean"
    if corruption_type in IMAGE_CORRUPTIONS:
        return "image"
    if corruption_type in TEXT_PERTURBATIONS:
        return "text"
    if corruption_type in MODALITY_CONFLICTS:
        return "conflict"
    return "unknown"


def format_html_table(frame: pd.DataFrame, columns=None) -> str:
    out = frame.copy()
    if columns is not None:
        out = out[columns]
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: f"{value:.3f}")
    return out.to_html(index=False, escape=False)


def plot_r1_curves(df: pd.DataFrame, clean: pd.Series, group_name: str, output_dir: Path):
    sub = df[df["group"] == group_name]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for corruption_type, group in sub.groupby("corruption_type"):
        group = group.sort_values("severity")
        ax.plot(group["severity"], group["R@1"], marker="o", linewidth=2, label=corruption_type)
    ax.axhline(clean["R@1"], color="black", linestyle="--", linewidth=1.2, label="clean R@1")
    ax.set_title(f"R@1 vs severity - {group_name}")
    ax.set_xlabel("Severity")
    ax.set_ylabel("R@1 (%)")
    ax.set_xticks(sorted(sub["severity"].unique()))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{group_name}_r1_curves.png", dpi=160)
    plt.close(fig)


def plot_r1_drop_heatmap(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
    plot_df = df[df["corruption_type"] != "clean"].pivot(
        index="corruption_type", columns="severity", values="R@1_drop"
    )
    order = summary.sort_values("mean_R@1_drop", ascending=False)["corruption_type"].tolist()
    plot_df = plot_df.loc[order]
    fig, ax = plt.subplots(figsize=(8, max(5, len(plot_df) * 0.35)))
    image = ax.imshow(plot_df.values, aspect="auto", cmap="Reds")
    ax.set_title("R@1 absolute drop from clean")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Corruption type")
    ax.set_xticks(range(len(plot_df.columns)))
    ax.set_xticklabels(plot_df.columns)
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            ax.text(j, i, f"{plot_df.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="R@1 drop")
    fig.tight_layout()
    fig.savefig(output_dir / "r1_drop_heatmap.png", dpi=160)
    plt.close(fig)


def plot_worst_drop_bar(summary: pd.DataFrame, output_dir: Path):
    bar = summary.sort_values("worst_R@1_drop", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(bar) * 0.32)))
    ax.barh(bar["corruption_type"], bar["worst_R@1_drop"], color="#c44e52")
    ax.set_title("Worst-case R@1 drop by corruption")
    ax.set_xlabel("R@1 drop from clean")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "worst_r1_drop_bar.png", dpi=160)
    plt.close(fig)


def build_summary(df: pd.DataFrame, clean: pd.Series) -> pd.DataFrame:
    rows = []
    for corruption_type, group in df[df["corruption_type"] != "clean"].groupby("corruption_type"):
        row = {"group": group_of(corruption_type), "corruption_type": corruption_type}
        for metric in METRICS:
            row[f"mean_{metric}"] = group[metric].mean()
            row[f"worst_{metric}"] = group[metric].min()
            row[f"mean_{metric}_drop"] = group[f"{metric}_drop"].mean()
            worst_idx = group[metric].idxmin()
            row[f"worst_{metric}_severity"] = int(df.loc[worst_idx, "severity"])
            row[f"worst_{metric}_drop"] = clean[metric] - df.loc[worst_idx, metric]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["group", "mean_R@1"])


def write_report(df: pd.DataFrame, summary: pd.DataFrame, group_summary: pd.DataFrame, clean: pd.Series, output_dir: Path):
    worst = df[df["corruption_type"] != "clean"].sort_values("R@1").head(10)
    best = df[df["corruption_type"] != "clean"].sort_values("R@1", ascending=False).head(10)
    style = """
<style>
body { font-family: Arial, sans-serif; margin: 28px; color: #222; }
h1, h2 { margin-bottom: 8px; }
table { border-collapse: collapse; margin: 14px 0 28px; font-size: 14px; }
th, td { border: 1px solid #ddd; padding: 7px 9px; text-align: right; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
th { background: #f3f5f7; }
img { max-width: 100%; margin: 12px 0 28px; border: 1px solid #ddd; }
.metric { display:inline-block; margin-right:18px; padding:10px 12px; background:#f7f7f7; border:1px solid #ddd; }
</style>
"""
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>FAFA Robustness Report</title>{style}</head><body>
<h1>FAFA Robustness Report</h1>
<div class="metric"><b>Clean R@1</b><br>{clean['R@1']:.3f}</div>
<div class="metric"><b>Clean R@5</b><br>{clean['R@5']:.3f}</div>
<div class="metric"><b>Clean R@10</b><br>{clean['R@10']:.3f}</div>
<div class="metric"><b>Clean R@50</b><br>{clean['R@50']:.3f}</div>
<h2>Group Summary</h2>
{format_html_table(group_summary)}
<h2>Worst 10 Cases by R@1</h2>
{format_html_table(worst, ['group','corruption_type','severity','R@1','R@5','R@10','R@50','R@1_drop'])}
<h2>Best 10 Perturbed Cases by R@1</h2>
{format_html_table(best, ['group','corruption_type','severity','R@1','R@5','R@10','R@50','R@1_drop'])}
<h2>R@1 Curves</h2>
<img src="robustness_report_assets/image_r1_curves.png">
<img src="robustness_report_assets/text_r1_curves.png">
<img src="robustness_report_assets/conflict_r1_curves.png">
<h2>R@1 Drop Heatmap</h2>
<img src="robustness_report_assets/r1_drop_heatmap.png">
<h2>Worst Drop Bar</h2>
<img src="robustness_report_assets/worst_r1_drop_bar.png">
<h2>Per-Corruption Summary</h2>
{format_html_table(summary)}
</body></html>"""
    (output_dir / "robustness_report.html").write_text(html, encoding="utf-8")


def analyze(input_csv: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "robustness_report_assets"
    assets_dir.mkdir(exist_ok=True)

    df = pd.read_csv(input_csv)
    df["severity"] = pd.to_numeric(df["severity"], errors="coerce").astype(int)
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["group"] = df["corruption_type"].map(group_of)

    clean_rows = df[df["corruption_type"] == "clean"]
    if clean_rows.empty:
        raise ValueError("The robustness CSV must contain a clean baseline row.")
    clean = clean_rows.iloc[0]
    for metric in METRICS:
        df[f"{metric}_drop"] = clean[metric] - df[metric]
        df[f"{metric}_relative_drop_pct"] = (clean[metric] - df[metric]) / clean[metric] * 100

    summary = build_summary(df, clean)
    group_summary = df[df["corruption_type"] != "clean"].groupby("group").agg(
        cases=("corruption_type", "count"),
        mean_R1=("R@1", "mean"),
        worst_R1=("R@1", "min"),
        mean_R1_drop=("R@1_drop", "mean"),
        worst_R1_drop=("R@1_drop", "max"),
        mean_R5=("R@5", "mean"),
        mean_R10=("R@10", "mean"),
        mean_R50=("R@50", "mean"),
    ).reset_index().sort_values("mean_R1")

    summary.to_csv(output_dir / "robustness_summary.csv", index=False)
    group_summary.to_csv(output_dir / "robustness_group_summary.csv", index=False)
    for group_name in ["image", "text", "conflict"]:
        plot_r1_curves(df, clean, group_name, assets_dir)
    plot_r1_drop_heatmap(df, summary, assets_dir)
    plot_worst_drop_bar(summary, assets_dir)
    write_report(df, summary, group_summary, clean, output_dir)

    print(f"Clean R@1: {clean['R@1']:.3f}")
    print(group_summary.to_string(index=False))
    print(f"\nReport: {output_dir / 'robustness_report.html'}")
    print(f"Summary CSV: {output_dir / 'robustness_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze FAFA robustness CSV and create plots/report.")
    parser.add_argument("--input", type=Path, required=True, help="Path to robustness_results_*.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for report files")
    args = parser.parse_args()
    output_dir = args.output_dir or args.input.parent
    analyze(args.input, output_dir)


if __name__ == "__main__":
    main()
