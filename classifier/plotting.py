"""Charts. Each function returns the Figure; the notebook decides whether
to show it or save it (fig.savefig)."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BAR, LINE = "steelblue", "firebrick"


def _draw_accuracy_by_bucket(ax1, summary: pd.DataFrame, annotate: bool, target: Optional[float]):
    x = summary["bucket_label"].astype(str)
    ax1.bar(x, summary["n_items"], color=BAR, alpha=0.7)
    ax1.set_xlabel("confidence bucket")
    ax1.set_ylabel("items", color=BAR)
    ax1.tick_params(axis="y", labelcolor=BAR)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax2 = ax1.twinx()
    ax2.plot(x, summary["accuracy"], color=LINE, marker="o", linewidth=2)
    ax2.set_ylabel("accuracy", color=LINE)
    ax2.tick_params(axis="y", labelcolor=LINE)
    ax2.set_ylim(0, 1.05)
    if annotate:
        for xi, yi in zip(x, summary["accuracy"]):
            if pd.notna(yi):
                ax2.annotate(f"{yi:.0%}", (xi, yi), textcoords="offset points", xytext=(0, 7),
                             ha="center", fontsize=8, color=LINE)
    if target is not None:
        ax2.axhline(target, linestyle="--", color="gray", linewidth=1)


def _draw_coverage(ax, summary: pd.DataFrame, target: Optional[float], annotate: bool):
    ax.plot(summary["cumulative_coverage"], summary["cumulative_accuracy"], marker="o", color=LINE)
    if annotate:
        for _, r in summary.iterrows():
            if pd.notna(r["cumulative_coverage"]):
                ax.annotate(str(r["bucket_label"]), (r["cumulative_coverage"], r["cumulative_accuracy"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)
    if target is not None:
        ax.axhline(target, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("coverage (share of items at or above the bucket)")
    ax.set_ylabel("accuracy of those items")
    ax.set_ylim(0, 1.05)


def _grid(n, ncols, per_panel):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(per_panel[0] * ncols, per_panel[1] * nrows),
                             squeeze=False)
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    return fig, axes


def plot_accuracy_by_bucket(summary: pd.DataFrame, title: str = "Full path: accuracy and volume by confidence",
                            target_accuracy: Optional[float] = None):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _draw_accuracy_by_bucket(ax, summary, True, target_accuracy)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_accuracy_by_bucket_grid(summaries: Dict[int, pd.DataFrame], levels: List[str], ncols: int = 2,
                                 target_accuracy: Optional[float] = None):
    fig, axes = _grid(len(summaries), ncols, (7.5, 4.6))
    for i, d in enumerate(sorted(summaries)):
        ax = axes[i // ncols][i % ncols]
        _draw_accuracy_by_bucket(ax, summaries[d], False, target_accuracy)
        ax.set_title(f"Correct through {levels[d - 1]}", fontsize=10)
    fig.suptitle("Accuracy and volume by confidence bucket, per depth")
    fig.tight_layout()
    return fig


def plot_coverage_curve(summary: pd.DataFrame, target_accuracy: Optional[float] = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_coverage(ax, summary, target_accuracy, True)
    ax.set_title("Full path: accuracy vs coverage")
    fig.tight_layout()
    return fig


def plot_coverage_curve_grid(summaries: Dict[int, pd.DataFrame], levels: List[str], ncols: int = 2,
                             target_accuracy: Optional[float] = None):
    fig, axes = _grid(len(summaries), ncols, (6.2, 4.2))
    for i, d in enumerate(sorted(summaries)):
        ax = axes[i // ncols][i % ncols]
        _draw_coverage(ax, summaries[d], target_accuracy, False)
        ax.set_title(f"Correct through {levels[d - 1]}", fontsize=10)
    fig.suptitle("Accuracy vs coverage, per depth")
    fig.tight_layout()
    return fig


def _draw_violins(ax, sub: pd.DataFrame):
    labels = (sub.drop_duplicates("bucket_label").sort_values("bucket_left")["bucket_label"].tolist())
    vals = {lab: sub.loc[(sub["bucket_label"] == lab) & sub["accuracy"].notna(), "accuracy"].to_numpy()
            for lab in labels}
    pos = [i for i, lab in enumerate(labels) if len(vals[lab]) >= 3]
    if pos:
        ax.violinplot([vals[labels[i]] for i in pos], positions=pos, showmeans=True, widths=0.7)
    for i, lab in enumerate(labels):       # points always; a violin needs >= 3 splits
        ax.scatter(np.full(len(vals[lab]), i), vals[lab], s=12, zorder=3, color="#333333", alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("accuracy")
    ax.grid(axis="y", alpha=0.3)


def plot_bucket_violins_by_level(buckets: pd.DataFrame, levels: List[str], ncols: int = 2):
    """Spread of accuracy across splits within each bucket, one panel per depth."""
    depths = sorted(buckets["depth"].unique())
    fig, axes = _grid(len(depths), ncols, (7.5, 4.2))
    for i, d in enumerate(depths):
        ax = axes[i // ncols][i % ncols]
        _draw_violins(ax, buckets[buckets["depth"] == d])
        ax.set_title(f"Correct through {levels[d - 1]}", fontsize=10)
    fig.suptitle("Accuracy per confidence bucket, across splits")
    fig.tight_layout()
    return fig


def plot_bucket_violins(buckets: pd.DataFrame, depth: int):
    fig, ax = plt.subplots(figsize=(11, 5))
    _draw_violins(ax, buckets[buckets["depth"] == depth])
    ax.set_title("Full path: accuracy per confidence bucket, across splits")
    fig.tight_layout()
    return fig


def plot_per_level_accuracy(m: dict, levels: List[str]):
    """Each level on its own, and correct-through-depth, from evaluation.metrics()."""
    lv = [m[f"level_accuracy {c}"] for c in levels]
    ct = [m[f"correct_through {c}"] for c in levels]
    x = np.arange(len(levels))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - 0.2, lv, 0.4, label="level alone", color=BAR)
    ax.bar(x + 0.2, ct, 0.4, label="correct through level", color=LINE, alpha=0.8)
    for xi, a, b in zip(x, lv, ct):
        ax.annotate(f"{a:.1%}", (xi - 0.2, a), ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{b:.1%}", (xi + 0.2, b), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy")
    ax.legend(loc="lower left")
    ax.set_title("Accuracy by hierarchy level")
    fig.tight_layout()
    return fig


def plot_top_confusions(confusions: pd.DataFrame, level: str, n: int = 15):
    top = confusions.head(n).iloc[::-1]
    a = f"actual {level.replace(' ID', ' Name')}"
    p = f"predicted {level.replace(' ID', ' Name')}"
    if a in top.columns:
        labels = top[a].astype(str) + "  ->  " + top[p].astype(str)
    else:
        labels = top[level].astype(str) + "  ->  " + top[f"Predicted {level}"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(top) + 1.5))
    ax.barh(labels, top["n"], color=BAR)
    ax.set_xlabel("misclassified items")
    ax.set_title(f"Most frequent confusions, {level}")
    fig.tight_layout()
    return fig


def plot_training_curves(history: pd.DataFrame, levels: List[str]):
    """Loss and per-level validation accuracy by epoch; the restored epoch marked."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    a1.plot(history["epoch"], history["train_loss"], label="train")
    if "val_loss" in history:
        a1.plot(history["epoch"], history["val_loss"], label="validation")
    a1.set_xlabel("epoch")
    a1.set_ylabel("loss")
    a1.legend()
    prefix = "val_acc" if f"val_acc {levels[0]}" in history else "train_acc"
    for c in levels:
        a2.plot(history["epoch"], history[f"{prefix} {c}"], label=c)
    a2.set_xlabel("epoch")
    a2.set_ylabel(f"{prefix.replace('_', ' ')}")
    a2.legend(fontsize=8)
    best = history.attrs.get("best_epoch")
    if best is None and "best" in history:
        flagged = history.loc[history["best"].astype(bool), "epoch"]
        best = int(flagged.iloc[-1]) if len(flagged) else None
    if best:
        for ax in (a1, a2):
            ax.axvline(best, linestyle="--", color="gray", linewidth=1)
    fig.tight_layout()
    return fig
