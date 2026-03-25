#!/usr/bin/env python3
"""
plot_results.py — Generates comparison plots from experiment.c training runs.

Reads an episodes CSV and summary CSV produced by metric_listener.py and
generates plots comparing hyperparameters: learning rate, hidden neurons,
hidden layers, hidden activation, and output activation.

Each plot shows mean ± std across all architectures sharing that hyperparameter
value, with one curve per unique value.

Usage:
    python plot_results.py --episodes episodes_20260318_103102.csv --summary summary_20260318_103102.csv
    python plot_results.py --episodes ep.csv --summary sum.csv --metric val_acc --output plots/
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as patheffects

# ── Arch name parser ──────────────────────────────────────────────────────────

ACT_LABELS = {"sig": "sigmoid", "relu": "relu", "lin": "linear"}


def parse_activations(name: str) -> tuple:
    """Extract (act_hidden, act_output) from arch_name like reluosig_l2x8_lr0.1."""
    m = re.match(r"^(\w+?)o(\w+?)_l", name)
    if not m:
        return "unknown", "unknown"
    return ACT_LABELS.get(m.group(1), m.group(1)), ACT_LABELS.get(m.group(2), m.group(2))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(ep_path: str, sum_path: str) -> tuple:
    ep = pd.read_csv(ep_path, on_bad_lines="skip")
    sm = pd.read_csv(sum_path, on_bad_lines="skip")

    # Parse activations into columns
    for df in (ep, sm):
        parsed = df["arch_name"].apply(parse_activations)
        df["act_hidden"] = parsed.apply(lambda x: x[0])
        df["act_output"] = parsed.apply(lambda x: x[1])

    # Filter out degenerate runs (inf losses)
    sm = sm.replace([np.inf, -np.inf], np.nan)
    bad_archs = sm[sm["final_val_loss"].isna()]["arch_idx"].unique()
    sm = sm[~sm["arch_idx"].isin(bad_archs)]
    ep = ep[~ep["arch_idx"].isin(bad_archs)]

    # Ensure numeric
    for col in ("train_loss", "val_loss", "train_acc", "val_acc"):
        ep[col] = pd.to_numeric(ep[col], errors="coerce")
    for col in ("final_train_loss", "final_val_loss", "final_train_acc", "final_val_acc"):
        sm[col] = pd.to_numeric(sm[col], errors="coerce")

    sm.dropna(subset=["final_val_loss", "final_val_acc"], inplace=True)

    return ep, sm


# ── Plot helpers ──────────────────────────────────────────────────────────────

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def sort_key(val):
    """Sort numerics numerically, strings alphabetically."""
    try:
        return (0, float(val))
    except (ValueError, TypeError):
        return (1, str(val))


def plot_episode_curves(ep: pd.DataFrame, group_col: str, metric: str,
                        ax: plt.Axes, title: str):
    """
    Plot mean ± std training curves for `metric` over episodes,
    one curve per unique value of `group_col`.
    """
    groups = sorted(ep[group_col].unique(), key=sort_key)

    for i, val in enumerate(groups):
        subset = ep[ep[group_col] == val]

        # Pivot: rows=episode, columns=arch_idx, values=metric
        pivot = subset.pivot_table(index="episode", columns="arch_idx",
                                   values=metric, aggfunc="first")
        mean = pivot.mean(axis=1)
        std = pivot.std(axis=1)
        episodes = mean.index

        color = COLORS[i % len(COLORS)]
        label = f"{val}"
        ax.plot(episodes, mean, color=color, label=label, linewidth=1.5)
        ax.fill_between(episodes, mean - std, mean + std,
                        color=color, alpha=0.15)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_summary_bars(sm: pd.DataFrame, group_col: str, metric: str,
                      ax: plt.Axes, title: str):
    """
    Bar chart of mean ± std of a final metric, one bar per unique value
    of `group_col`.
    """
    groups = sorted(sm[group_col].unique(), key=sort_key)
    means, stds, labels = [], [], []

    for val in groups:
        subset = sm[sm[group_col] == val][metric]
        means.append(subset.mean())
        stds.append(subset.std())
        labels.append(str(val))

    x = np.arange(len(labels))
    colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]

    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").replace("final ", "").title())
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    # Value annotations
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.001,
                f"{m:.4f}", ha="center", va="bottom", fontsize=7)


# ── Main plotting ─────────────────────────────────────────────────────────────

HYPERPARAMS = [
    ("lr",           "Learning Rate"),
    ("hidden",       "Hidden Neurons"),
    ("hidden_layers","Hidden Layers"),
    ("act_hidden",   "Hidden Activation"),
    ("act_output",   "Output Activation"),
]


def generate_plots(ep: pd.DataFrame, sm: pd.DataFrame, output_dir: Path,
                   metric: str = "val_loss"):
    """Generate one figure per hyperparameter with 3 subplots each."""
    output_dir.mkdir(parents=True, exist_ok=True)

    final_metric = f"final_{metric}"
    # Determine if higher is better (for acc) or lower (for loss)
    is_acc = "acc" in metric

    for col, label in HYPERPARAMS:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Effect of {label}", fontsize=14, fontweight="bold")

        # 1) Training curves: val_loss
        plot_episode_curves(ep, col, "val_loss", axes[0],
                            f"Val Loss by {label}")

        # 2) Training curves: val_acc
        plot_episode_curves(ep, col, "val_acc", axes[1],
                            f"Val Accuracy by {label}")
        axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        # 3) Summary bars for the chosen metric
        plot_summary_bars(sm, col, final_metric, axes[2],
                          f"Final {metric.replace('_', ' ').title()} by {label}")
        if is_acc:
            axes[2].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

        plt.tight_layout()
        out_path = output_dir / f"compare_{col}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")

    # ── Combined heatmap: mean final val_loss by (hidden_layers × hidden) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Architecture Grid: Hidden Layers × Hidden Neurons",
                 fontsize=14, fontweight="bold")

    for ax, metric_name, cmap in [
        (axes[0], "final_val_loss", "viridis_r"),
        (axes[1], "final_val_acc",  "viridis"),
    ]:
        pivot = sm.pivot_table(index="hidden_layers", columns="hidden",
                               values=metric_name, aggfunc="mean")
        im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Hidden Neurons")
        ax.set_ylabel("Hidden Layers")
        title = metric_name.replace("final_", "").replace("_", " ").title()
        ax.set_title(f"Mean {title}")
        fig.colorbar(im, ax=ax, shrink=0.8)

        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    fmt = f"{val:.3f}" if "loss" in metric_name else f"{val:.1%}"
                    ax.text(j, i, fmt, ha="center", va="center",
                            fontsize=7, color="white",
                            path_effects=[
                                patheffects.withStroke(
                                    linewidth=2, foreground="black"
                                )
                            ])

    plt.tight_layout()
    out_path = output_dir / "heatmap_arch_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Plot training results from experiment.c runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--episodes", "-e", required=True, help="Path to episodes CSV")
    p.add_argument("--summary", "-s", required=True, help="Path to summary CSV")
    p.add_argument("--metric", "-m", default="val_loss",
                   choices=["val_loss", "val_acc", "train_loss", "train_acc"],
                   help="Primary metric for summary bar charts")
    p.add_argument("--output", "-o", default="plots", help="Output directory")
    args = p.parse_args()

    for path in (args.episodes, args.summary):
        if not Path(path).exists():
            print(f"ERROR: {path} not found")
            sys.exit(1)

    print("Loading data...")
    ep, sm = load_data(args.episodes, args.summary)
    print(f"  Episodes: {len(ep)} rows, {ep['arch_idx'].nunique()} architectures")
    print(f"  Summary:  {len(sm)} rows (after filtering degenerate runs)")
    print(f"  Hyperparameter values found:")
    for col, label in HYPERPARAMS:
        vals = sorted(sm[col].unique(), key=sort_key)
        print(f"    {label}: {vals}")

    print(f"\nGenerating plots...")
    generate_plots(ep, sm, Path(args.output), args.metric)
    print(f"\nDone! All plots saved to {args.output}/")


if __name__ == "__main__":
    main()
