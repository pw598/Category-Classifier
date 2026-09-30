"""Accuracy measures, overall, by level, by split and by confidence bucket.

A predictions frame here carries 'Predicted <level>' and 'L<d> Confidence'
columns (from hierarchy.score_paths) alongside the true <level> columns,
and optionally a 'split' column identifying the fold/repeat.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def correct_through(pred: pd.DataFrame, levels: List[str]) -> np.ndarray:
    """(n, n_levels) bool: prediction right at every level down to d."""
    hit = np.column_stack([pred[f"Predicted {c}"].astype(str).to_numpy() == pred[c].astype(str).to_numpy()
                           for c in levels])
    return np.logical_and.accumulate(hit, axis=1)


def metrics(pred: pd.DataFrame, levels: List[str]) -> Dict[str, float]:
    """Path accuracy, correct-through-each-depth, and each level on its own."""
    ct = correct_through(pred, levels)
    out = {"n": int(len(pred)), "path_accuracy": float(ct[:, -1].mean())}
    for d, c in enumerate(levels):
        out[f"correct_through {c}"] = float(ct[:, d].mean())
    for c in levels:
        out[f"level_accuracy {c}"] = float((pred[f"Predicted {c}"].astype(str) == pred[c].astype(str)).mean())
    if "readable" in pred.columns:
        out["share_unreadable"] = float((~pred["readable"].astype(bool)).mean())
    return out


def metrics_by_split(pred: pd.DataFrame, levels: List[str], split_col: str = "split") -> pd.DataFrame:
    """One row per split, then mean and std across splits."""
    rows = [{"split": k, **metrics(g, levels)} for k, g in pred.groupby(split_col)]
    table = pd.DataFrame(rows)
    num = table.drop(columns="split")
    summary = pd.DataFrame([{"split": "mean", **num.mean().to_dict()},
                            {"split": "std", **num.std(ddof=1).to_dict()}])
    return pd.concat([table, summary], ignore_index=True)


def _edges(scores: np.ndarray, width: float, n_quantile: Optional[int]) -> np.ndarray:
    if n_quantile:
        edges = np.unique(np.quantile(scores, np.linspace(0, 1, n_quantile + 1)))
        if len(edges) >= 3:
            return edges
        print("[buckets] scores too concentrated for quantile buckets; using uniform width")
    return np.arange(0, 1.0 + width / 2, width)


def bucket_accuracy(pred: pd.DataFrame, levels: List[str], split_col: Optional[str] = "split",
                    bucket_width: float = 0.05, n_quantile_buckets: Optional[int] = None) -> pd.DataFrame:
    """Accuracy per confidence bucket, per depth and split (long format).

    At depth d the score is 'L<d> Confidence' and correct means right down
    to d. Each depth is bucketed on its own score distribution, so read
    each depth separately rather than comparing bands across depths.
    """
    ct = correct_through(pred, levels)
    split = pred[split_col].to_numpy() if split_col and split_col in pred.columns else np.zeros(len(pred), int)
    rows = []
    for d in range(1, len(levels) + 1):
        scores = pred[f"L{d} Confidence"].to_numpy(dtype=float)
        band = pd.cut(scores, bins=_edges(scores, bucket_width, n_quantile_buckets), include_lowest=True)
        per = (pd.DataFrame({"bucket": band, "correct": ct[:, d - 1], "split": split})
               .groupby(["split", "bucket"], observed=False)
               .agg(n_items=("correct", "size"), accuracy=("correct", "mean")).reset_index())
        per["depth"] = d
        rows.append(per)
    out = pd.concat(rows, ignore_index=True)
    out["bucket_label"] = out["bucket"].astype(str)
    out["bucket_left"] = out["bucket"].map(lambda b: b.left).astype(float)
    return out


def bucket_summary(buckets: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Pooled across splits, one table per depth, with cumulative coverage.

    cumulative_coverage / cumulative_accuracy read from the top bucket
    down: accepting everything at or above this bucket covers this share
    of items at this accuracy.
    """
    out = {}
    for depth, g in buckets.groupby("depth"):
        g = g.assign(_correct=g["n_items"] * g["accuracy"].fillna(0))
        per = (g.groupby(["bucket_left", "bucket_label"], observed=True)
               .agg(n_items=("n_items", "sum"), _correct=("_correct", "sum"),
                    mean_split_accuracy=("accuracy", "mean"), std_split_accuracy=("accuracy", "std"))
               .reset_index().sort_values("bucket_left").reset_index(drop=True))
        per["accuracy"] = per["_correct"] / per["n_items"].replace(0, np.nan)
        total = per["n_items"].sum()
        per["share_of_items"] = per["n_items"] / total if total else np.nan
        rev = per.iloc[::-1]
        per["cumulative_coverage"] = (rev["n_items"].cumsum() / total).iloc[::-1].to_numpy()
        per["cumulative_accuracy"] = (rev["_correct"].cumsum()
                                      / rev["n_items"].cumsum().replace(0, np.nan)).iloc[::-1].to_numpy()
        out[int(depth)] = per.drop(columns="_correct")
    return out


def top_confusions(pred: pd.DataFrame, level: str, n: int = 20,
                   lookup: Optional[pd.DataFrame] = None, name_col: Optional[str] = None) -> pd.DataFrame:
    """The most frequent actual -> predicted mix-ups at one level."""
    wrong = pred[pred[f"Predicted {level}"].astype(str) != pred[level].astype(str)]
    out = (wrong.groupby([level, f"Predicted {level}"]).size().rename("n")
           .sort_values(ascending=False).head(n).reset_index())
    if lookup is not None and name_col and name_col in lookup.columns:
        names = lookup[[level, name_col]].drop_duplicates(level).set_index(level)[name_col]
        out.insert(1, f"actual {name_col}", out[level].map(names))
        out.insert(3, f"predicted {name_col}", out[f"Predicted {level}"].map(names))
    return out
