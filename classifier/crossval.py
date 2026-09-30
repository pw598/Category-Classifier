"""Cross-validation that can be spread over several sessions.

A run lives in its own directory, and every split writes its results
there the moment it finishes:

    outputs/cv_<stamp>/
        settings.json          method, splits, seed, and the data it ran on
        model_config.json      the configuration being validated
        assignment.csv         row, ID, and split_<k> = 1 where the row is
                               in split k's test set
        split_<k>/predictions.csv, history.csv, done.json

To continue a run in a later session, point notebook 03 at the run
directory: finished splits are skipped, and the data is checked against
assignment.csv row by row, so a changed cleaned file or blacklist is
caught instead of silently mixing two datasets. A split interrupted part
way through is simply re-run.

Two methods:
    kfold     each row is in exactly one test set; every row gets one
              out-of-sample prediction
    holdout   n_splits independent random draws of test_size; a row can
              appear in several test sets or none, but each bucket gets
              an independent accuracy estimate per split (the violins)
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import time
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .io import load_json, save_json
from .model import CategoryClassifier, ModelConfig, validation_split


def make_splits(labels: pd.Series, method: str = "kfold", n_splits: int = 5,
                test_size: float = 0.2, stratify: bool = True, seed: int = 42) -> np.ndarray:
    """(n_rows, n_splits) int8, 1 where a row is in that split's test set."""
    from sklearn.model_selection import KFold, ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
    y = labels.astype(str).to_numpy()
    n = len(y)
    if stratify:
        smallest = pd.Series(y).value_counts().min()
        need = n_splits if method == "kfold" else 2
        if smallest < need:
            raise ValueError(
                f"the smallest class has {smallest} rows; a stratified {method} split needs "
                f"{need}. Raise min_path_count in notebook 01, or set STRATIFY = False.")
    if method == "kfold":
        splitter = (StratifiedKFold if stratify else KFold)(n_splits=n_splits, shuffle=True, random_state=seed)
    elif method == "holdout":
        splitter = (StratifiedShuffleSplit if stratify else ShuffleSplit)(
            n_splits=n_splits, test_size=test_size, random_state=seed)
    else:
        raise ValueError(f"method must be 'kfold' or 'holdout', got {method!r}")
    out = np.zeros((n, n_splits), dtype=np.int8)
    for k, (_, test_idx) in enumerate(splitter.split(np.zeros(n), y if stratify else None)):
        out[test_idx, k] = 1
    return out


def start_run(run_dir: str, df: pd.DataFrame, cfg: ModelConfig, method: str, n_splits: int,
              test_size: float = 0.2, stratify: bool = True, seed: int = 42,
              nn_val_frac: float = 0.05, save_split_models: bool = False,
              extra: Optional[dict] = None) -> None:
    """Fix the splits and settings for a new run. Nothing is trained here."""
    if os.path.exists(os.path.join(run_dir, "settings.json")):
        raise FileExistsError(f"{run_dir} already holds a run; resume it instead")
    os.makedirs(run_dir, exist_ok=True)
    splits = make_splits(df[cfg.level_columns[-1]], method, n_splits, test_size, stratify, seed)
    assignment = pd.DataFrame({"row": np.arange(len(df)), cfg.id_col: df[cfg.id_col].astype(str).to_numpy()})
    for k in range(n_splits):
        assignment[f"split_{k}"] = splits[:, k]
    assignment.to_csv(os.path.join(run_dir, "assignment.csv"), index=False)
    cfg.save(os.path.join(run_dir, "model_config.json"))
    save_json({"method": method, "n_splits": n_splits, "test_size": test_size if method == "holdout" else None,
               "stratify": stratify, "seed": seed, "nn_val_frac": nn_val_frac,
               "save_split_models": save_split_models, "n_rows": len(df), **(extra or {})},
              os.path.join(run_dir, "settings.json"))
    print(f"[cv] new run in {run_dir}: {method}, {n_splits} splits over {len(df):,} rows")


def load_run(run_dir: str, df: pd.DataFrame):
    """(settings, cfg, assignment), after checking df is the data the run started on."""
    settings = load_json(os.path.join(run_dir, "settings.json"))
    cfg = ModelConfig.load(os.path.join(run_dir, "model_config.json"))
    assignment = pd.read_csv(os.path.join(run_dir, "assignment.csv"), dtype={cfg.id_col: str},
                             keep_default_na=False)
    if len(assignment) != len(df) or not np.array_equal(
            assignment[cfg.id_col].to_numpy(), df[cfg.id_col].astype(str).to_numpy()):
        raise ValueError(
            f"the data does not match the run in {run_dir} ({len(df):,} rows now, "
            f"{len(assignment):,} when it started, or the IDs differ). Load the same cleaned "
            "file with the same blacklist and exclusions the run began with.")
    return settings, cfg, assignment


def status(run_dir: str) -> pd.DataFrame:
    settings = load_json(os.path.join(run_dir, "settings.json"))
    rows = []
    for k in range(settings["n_splits"]):
        path = os.path.join(run_dir, f"split_{k}", "done.json")
        rows.append({"split": k, "done": os.path.exists(path), **(load_json(path) if os.path.exists(path) else {})})
    return pd.DataFrame(rows)


def run_splits(run_dir: str, df: pd.DataFrame, splits: Optional[Iterable[int]] = None,
               verbose: bool = True) -> pd.DataFrame:
    """Train and score the requested splits (default: every unfinished one).

    Each finished split is written to disk before the next starts, so the
    run can stop between any two splits and continue in another session.
    """
    settings, cfg, assignment = load_run(run_dir, df)
    done = set(status(run_dir).query("done")["split"])
    todo = [k for k in (range(settings["n_splits"]) if splits is None else splits) if k not in done]
    if not todo:
        print("[cv] nothing to do: every requested split is finished")
    df = df.reset_index(drop=True)
    for k in todo:
        split_dir = os.path.join(run_dir, f"split_{k}")
        os.makedirs(split_dir, exist_ok=True)
        test_mask = assignment[f"split_{k}"].to_numpy() == 1
        train, test = df[~test_mask], df[test_mask]
        val = None
        if cfg.model_type == "nn" and settings["nn_val_frac"]:
            train, val = validation_split(train, settings["nn_val_frac"], cfg.level_columns[-1],
                                          settings["seed"] + k)
        print(f"\n[cv] split {k}: train {len(train):,}"
              + (f", validation {len(val):,}" if val is not None else "") + f", test {len(test):,}")
        t0 = time.time()
        clf = CategoryClassifier(cfg).fit(train, val, checkpoint_dir=os.path.join(split_dir, "checkpoint"),
                                          verbose=verbose)
        pred = clf.predict(test[cfg.text_col], verbose=verbose)
        pred.insert(0, "row", np.flatnonzero(test_mask))
        pred.insert(1, "split", k)
        pred.to_csv(os.path.join(split_dir, "predictions.csv"), index=False)
        record = {"seconds": round(time.time() - t0, 1), "n_train": len(train),
                  "n_val": 0 if val is None else len(val), "n_test": int(test_mask.sum()),
                  "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        if clf.history is not None:
            clf.history.to_csv(os.path.join(split_dir, "history.csv"), index=False)
            record["best_epoch"] = clf.history.attrs.get("best_epoch")
            record["epochs_run"] = len(clf.history)
        if settings.get("save_split_models"):
            clf.save(os.path.join(split_dir, "model"))
        shutil.rmtree(os.path.join(split_dir, "checkpoint"), ignore_errors=True)
        save_json(record, os.path.join(split_dir, "done.json"))     # written last: marks the split complete
        print(f"[cv] split {k} done in {record['seconds'] / 60:.1f} min")
        del clf, pred, train, test, val
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return status(run_dir)


def collect(run_dir: str, df: pd.DataFrame, allow_partial: bool = False) -> pd.DataFrame:
    """Every split's test predictions, joined to the truth, in one frame."""
    settings, cfg, _ = load_run(run_dir, df)
    st = status(run_dir)
    if not st["done"].all() and not allow_partial:
        raise RuntimeError(f"splits not finished: {st.loc[~st['done'], 'split'].tolist()} "
                           "(pass allow_partial=True to look at what is done)")
    parts = [pd.read_csv(os.path.join(run_dir, f"split_{k}", "predictions.csv"), dtype=str,
                         keep_default_na=False) for k in st.loc[st["done"], "split"]]
    pred = pd.concat(parts, ignore_index=True)
    for c in pred.columns:
        if c.endswith("Confidence") or c.endswith("Level Prob") or c in ("Path Log Evidence",):
            pred[c] = pred[c].astype(float)
    pred["row"] = pred["row"].astype(int)
    pred["split"] = pred["split"].astype(int)
    pred["n_active_features"] = pred["n_active_features"].astype(int)
    pred["readable"] = pred["readable"].map({"True": True, "False": False})
    keep = [cfg.id_col, cfg.text_col] + list(cfg.level_columns) + [c for c in cfg.name_columns if c in df.columns]
    truth = df.reset_index(drop=True)[keep]
    return pd.concat([truth.iloc[pred["row"]].reset_index(drop=True), pred], axis=1)


def median_best_epoch(run_dir: str) -> Optional[int]:
    """Median best epoch across finished NN splits, for fixed-epoch final training."""
    st = status(run_dir)
    if "best_epoch" not in st:
        return None
    vals = st.loc[st["done"], "best_epoch"].dropna()
    return int(round(float(vals.median()))) if len(vals) else None
