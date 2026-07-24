#!/usr/bin/env python3
"""Build the report's evidence-bearing figures from frozen raw JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "repro" / "evidence" / "full_grid_raw.json"
KS = np.array([2, 3, 4, 5])
SEEDS = [604, 1337, 20260719]
COLORS = {
    "ours": "#087E8B",
    "mogfn": "#FF5A5F",
    "hngfn": "#F2B134",
    "ensemble": "#6C5CE7",
    "paper": "#22333B",
    "high": "#087E8B",
    "low": "#E4572E",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#172554",
            "axes.titlecolor": "#0F172A",
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "legend.frameon": False,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )


def save(fig: plt.Figure, output: Path, name: str) -> None:
    path = output / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def headline(raw: dict, output: Path) -> None:
    aggregates = raw["summary"]["aggregates"]
    paper = raw["summary"]["paper_table"]
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    offsets = {"ours": -0.24, "mogfn": 0.0, "hngfn": 0.24}
    widths = 0.21
    labels = {"ours": "Routing by Reaching", "mogfn": "MOGFN", "hngfn": "HN-GFN"}
    for method in ("ours", "mogfn", "hngfn"):
        means = np.array([aggregates[method][str(k)]["mean"] for k in KS])
        errors = np.array(
            [
                [
                    means[i] - aggregates[method][str(k)]["ci95_t"][0]
                    for i, k in enumerate(KS)
                ],
                [
                    aggregates[method][str(k)]["ci95_t"][1] - means[i]
                    for i, k in enumerate(KS)
                ],
            ]
        )
        ax.bar(
            KS + offsets[method],
            means,
            widths,
            color=COLORS[method],
            label=f"Observed {labels[method]}",
            alpha=0.92,
            yerr=errors,
            capsize=3,
            error_kw={"elinewidth": 1.2},
        )
    ax.plot(
        KS,
        [paper["ours"][str(k)] for k in KS],
        color=COLORS["paper"],
        linestyle="--",
        marker="D",
        linewidth=1.8,
        label="Paper: ours = 0.003",
    )
    ax.set(
        xlabel="Number of objectives (k)",
        ylabel="Terminal-distribution L1 error ↓",
        title="The method still wins, but the exact 0.003 point is outside every 95% CI",
        xticks=KS,
    )
    ax.set_ylim(0, 0.068)
    ax.legend(ncol=2, loc="upper left")
    ax.text(
        0.99,
        0.96,
        "32×32 grid · 20k steps · 3 seeds · 128 preferences/k",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color="#475569",
        fontsize=9,
    )
    save(fig, output, "headline-table1.png")


def seed_robustness(raw: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), sharey=True)
    labels = {"ours": "Routing by Reaching", "mogfn": "MOGFN", "hngfn": "HN-GFN"}
    for ax, method in zip(axes, ("ours", "mogfn", "hngfn"), strict=True):
        for index, seed in enumerate(SEEDS):
            means = [
                float(np.mean(raw["table1"][str(seed)][method][str(k)])) for k in KS
            ]
            ax.plot(
                KS,
                means,
                marker="o",
                linewidth=1.8,
                color=COLORS[method],
                alpha=0.55 + index * 0.2,
                label=str(seed),
            )
        ax.set_title(labels[method])
        ax.set_xticks(KS)
        ax.set_xlabel("Objectives (k)")
    axes[0].set_ylabel("Mean L1 error ↓")
    axes[-1].legend(title="Seed", loc="upper left")
    fig.suptitle(
        "The comparative ordering is stable across all three deterministic seeds",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    save(fig, output, "seed-robustness.png")


def reaching_ablation(raw: dict, output: Path) -> None:
    ours = []
    ensemble = []
    labels = []
    for seed in SEEDS:
        for k in KS:
            ours.append(float(np.mean(raw["table1"][str(seed)]["ours"][str(k)])))
            ensemble.append(
                float(np.mean(raw["table1"][str(seed)]["ensemble"][str(k)]))
            )
            labels.append(int(k))
    fig, ax = plt.subplots(figsize=(7.5, 6.1))
    scatter = ax.scatter(
        ours,
        ensemble,
        c=labels,
        cmap="viridis",
        s=82,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    limit = max(ensemble) * 1.08
    ax.plot([0, limit], [0, limit], linestyle="--", color="#64748B", linewidth=1.2)
    ax.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="Full routing policy L1 ↓",
        ylabel="Without reaching weights L1 ↓",
        title="Every paired ablation is substantially worse without reaching weights",
    )
    colorbar = fig.colorbar(scatter, ax=ax, ticks=KS)
    colorbar.set_label("Objectives (k)")
    ax.text(
        0.04,
        0.95,
        "12/12 seed×k pairs improve by ≥0.05",
        transform=ax.transAxes,
        va="top",
        color="#172554",
        fontweight="bold",
    )
    save(fig, output, "reaching-ablation.png")


def distortion_primary(raw: dict, output: Path) -> None:
    primary_names = ("harmonic_mean_circle12", "contrast_circle12")
    rows = [
        row for row in raw["distortion"] if row["custom_dist"] in primary_names
    ]
    rows.sort(key=lambda row: (primary_names.index(row["custom_dist"]), row["seed"]))
    labels = [
        ("Harmonic" if "harmonic" in row["custom_dist"] else "Contrast")
        + f"\n{row['seed']}"
        for row in rows
    ]
    x = np.arange(len(rows))
    high = [row["high_g_median_relative_deviation"] for row in rows]
    low = [row["low_g_median_relative_deviation"] for row in rows]
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.bar(x - 0.18, high, 0.36, label="High-G decile", color=COLORS["high"])
    ax.bar(x + 0.18, low, 0.36, label="Bottom half of G", color=COLORS["low"])
    ax.set(
        xticks=x,
        xticklabels=labels,
        ylabel="Median relative deviation from 1/Zₘ ↓",
        title="Primary Figure 3 compositions: distortion is smaller where G is high",
    )
    ax.axhline(0.30, color="#475569", linestyle=":", linewidth=1.3, label="0.30 gate")
    ax.legend(ncol=3, loc="upper left")
    save(fig, output, "distortion-primary.png")


def distortion_stress(raw: dict, output: Path) -> None:
    rows = raw["distortion"]
    primary_names = {"harmonic_mean_circle12", "contrast_circle12"}
    low = np.array([row["low_g_median_relative_deviation"] for row in rows])
    high = np.array([row["high_g_median_relative_deviation"] for row in rows])
    primary = np.array([row["custom_dist"] in primary_names for row in rows])
    fig, ax = plt.subplots(figsize=(7.6, 6.3))
    ax.scatter(
        low[~primary],
        high[~primary],
        color="#94A3B8",
        s=55,
        alpha=0.8,
        label="Figure A6 stress settings",
    )
    ax.scatter(
        low[primary],
        high[primary],
        color=COLORS["high"],
        edgecolors="white",
        linewidths=0.8,
        s=92,
        label="Primary Figure 3 settings",
        zorder=3,
    )
    limit = max(float(low.max()), float(high.max())) * 1.04
    ax.plot([0, limit], [0, limit], linestyle="--", color="#E4572E", linewidth=1.3)
    ax.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="Bottom-half median relative deviation",
        ylabel="High-G-decile median relative deviation",
        title="31/36 stress audits fall below the equal-deviation line",
    )
    ax.legend(loc="upper left")
    ax.text(
        0.97,
        0.04,
        "Five appendix rows reverse direction;\nall six primary rows pass.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#475569",
    )
    save(fig, output, "distortion-stress.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    raw = json.loads(RAW.read_text())
    headline(raw, args.output)
    seed_robustness(raw, args.output)
    reaching_ablation(raw, args.output)
    distortion_primary(raw, args.output)
    distortion_stress(raw, args.output)
    manifest = {
        path.name: path.stat().st_size
        for path in sorted(args.output.glob("*.png"))
    }
    (args.output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"output": str(args.output), "figures": manifest}, indent=2))
    if len(manifest) != 5 or any(size < 10_000 for size in manifest.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
