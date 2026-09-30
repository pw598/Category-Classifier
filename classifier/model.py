"""The model configuration and the classifier that ties the pieces together.

CategoryClassifier runs the same steps for both model types:

    prune rare tokens  ->  vectorize  ->  per-level probabilities  ->  paths
    (training rows)        (fit on         NB: one SparseMultinomialNB per level
                            training)      NN: one multi-head network

and saves/loads as a directory. A ModelConfig is JSON-serialisable, so
the configuration chosen in notebook 02 or 03 is exactly what notebook 04
trains, and what notebook 05 predicts with.
"""

from __future__ import annotations

import gc
import json
import os
import platform
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from .features import VectorizerSettings, build_keep_vocab, prune
from .hierarchy import LabelHierarchy, score_paths
from .nb import NBSettings, SparseMultinomialNB

LEVEL_COLUMNS = ["Level 1 ID", "Level 2 ID", "Level 3 ID", "Level 4 ID"]
NAME_COLUMNS = ["Level 1 Name", "Level 2 Name", "Level 3 Name", "Level 4 Name"]


@dataclass
class ModelConfig:
    model_type: str = "nb"                          # "nb" | "nn"
    text_col: str = "FullDesc"
    id_col: str = "Vallen ID"
    level_columns: List[str] = field(default_factory=lambda: list(LEVEL_COLUMNS))
    name_columns: List[str] = field(default_factory=lambda: list(NAME_COLUMNS))
    min_token_count: int = 1                        # 1 = off; counted on training rows only
    vectorizer: VectorizerSettings = field(default_factory=VectorizerSettings)
    nb: NBSettings = field(default_factory=NBSettings)
    nn: dict = field(default_factory=dict)          # NNSettings as a dict (torch-free here)
    path_mode: str = "independent"                  # "independent" | "conditional"
    normalise: bool = True                          # normalise path scores over valid paths
    score_batch_size: int = 2000                    # rows per path-scoring block (memory)
    predict_chunk_size: int = 50000                 # rows per prediction chunk (memory)
    # provenance, filled in by the notebooks
    cleaning_stamp: Optional[str] = None
    blacklist_file: Optional[str] = None
    source: Optional[str] = None                    # which run produced this config
    notes: str = ""

    def nn_settings(self):
        from .nn import NNSettings
        return NNSettings.from_dict({**NNSettings().to_dict(), **self.nn})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vectorizer"] = self.vectorizer.to_dict()
        d["nb"] = self.nb.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        d["vectorizer"] = VectorizerSettings.from_dict(d["vectorizer"])
        d["nb"] = NBSettings(**d["nb"])
        return cls(**d)

    def save(self, path) -> None:
        from .io import save_json
        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path) -> "ModelConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def validation_split(df: pd.DataFrame, frac: float, stratify_col: Optional[str], seed: int = 42):
    """(train, validation). Stratified on stratify_col where every class has
    two or more rows to spare; otherwise a plain random split."""
    from sklearn.model_selection import train_test_split
    strat = None
    if stratify_col:
        counts = df[stratify_col].value_counts()
        if counts.min() >= 2 and int(round(frac * len(df))) >= len(counts):
            strat = df[stratify_col]
        else:
            print("[split] validation split not stratified (classes too small for the fraction)")
    tr, va = train_test_split(df, test_size=frac, random_state=seed, stratify=strat)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


class CategoryClassifier:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.keep_vocab = None
        self.vectorizer = None
        self.hierarchy: Optional[LabelHierarchy] = None
        self.models = None          # NB: list of SparseMultinomialNB
        self.net = None             # NN: SparseInputMLP
        self.history: Optional[pd.DataFrame] = None

    # -- features -----------------------------------------------------------

    def _texts(self, texts: pd.Series) -> pd.Series:
        return prune(pd.Series(texts).reset_index(drop=True), self.keep_vocab)

    def transform(self, texts: pd.Series):
        return self.vectorizer.transform(self._texts(texts)).astype(np.float32).tocsr()

    # -- fitting ------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None,
            checkpoint_dir: Optional[str] = None, verbose: bool = True) -> "CategoryClassifier":
        """Fit on train_df. val_df is used by the NN only (early stopping and
        checkpointing); for NB it is ignored."""
        cfg, levels = self.cfg, self.cfg.level_columns
        train_df = train_df.reset_index(drop=True)
        texts = train_df[cfg.text_col].fillna("").astype(str)

        if cfg.min_token_count > 1:
            self.keep_vocab = build_keep_vocab(texts, cfg.min_token_count)
            if verbose:
                print(f"[fit] kept {len(self.keep_vocab):,} tokens seen in >= "
                      f"{cfg.min_token_count} training descriptions")
        self.vectorizer = cfg.vectorizer.build()
        X = self.vectorizer.fit_transform(self._texts(texts)).astype(np.float32).tocsr()
        if verbose:
            print(f"[fit] {X.shape[0]:,} rows x {X.shape[1]:,} features, "
                  f"{X.nnz / max(X.shape[0], 1):.1f} non-zeros per row")

        self.hierarchy = LabelHierarchy.from_frame(train_df, levels)
        y = self.hierarchy.encode(train_df)
        if verbose:
            print(self.hierarchy.describe().to_string(index=False))
            print(f"[fit] {self.hierarchy.n_paths:,} valid paths")

        if cfg.model_type == "nb":
            self.models = []
            for d in range(len(levels)):
                m = SparseMultinomialNB(**cfg.nb.to_dict()).fit(X, y[:, d], self.hierarchy.n_classes(d))
                self.models.append(m)
            if verbose:
                mb = sum(m.nbytes() for m in self.models) / 1e6
                print(f"[fit] Naive Bayes fitted, {mb:,.0f} MB")
        elif cfg.model_type == "nn":
            from . import nn
            s = cfg.nn_settings()
            self.net = nn.SparseInputMLP(
                X.shape[1], [self.hierarchy.n_classes(d) for d in range(len(levels))],
                hidden_dims=s.hidden_dims, dropout=s.dropout, batch_norm=s.batch_norm,
                sparse_grad=s.sparse_grad, feature_dropout=s.feature_dropout)
            X_val = y_val = None
            if val_df is not None and len(val_df):
                X_val = self.transform(val_df[cfg.text_col])
                y_val = self.hierarchy.encode(val_df.reset_index(drop=True))
            self.history = nn.fit(self.net, X, y, s, X_val, y_val, checkpoint_dir=checkpoint_dir,
                                  level_names=levels, verbose=verbose)
        else:
            raise ValueError(f"model_type must be 'nb' or 'nn', got {cfg.model_type!r}")
        del X
        gc.collect()
        return self

    # -- prediction ---------------------------------------------------------

    def level_probs(self, X) -> List[np.ndarray]:
        if self.cfg.model_type == "nb":
            return [m.predict_proba(X) for m in self.models]
        from . import nn
        s = self.cfg.nn_settings()
        return nn.predict_proba(self.net, X, s.eval_batch_size, s.prefer_gpu)

    def predict(self, texts: pd.Series, chunk_size: Optional[int] = None,
                verbose: bool = True) -> pd.DataFrame:
        """One row per input: predicted path, confidences, readability.

        Chunked so that memory is bounded by the chunk, not the input: the
        per-level probability matrices are rows x classes, the largest
        allocation in prediction.
        """
        texts = pd.Series(texts).reset_index(drop=True)
        chunk = chunk_size or self.cfg.predict_chunk_size
        parts = []
        for start in range(0, len(texts), chunk):
            X = self.transform(texts.iloc[start:start + chunk])
            probs = self.level_probs(X)
            frame = score_paths(self.hierarchy, probs, mode=self.cfg.path_mode,
                                normalise=self.cfg.normalise, batch_size=self.cfg.score_batch_size)
            frame["n_active_features"] = X.getnnz(axis=1)
            parts.append(frame)
            del X, probs
            gc.collect()
            if verbose and len(texts) > chunk:
                print(f"[predict] {min(start + chunk, len(texts)):,} / {len(texts):,}")
        if not parts:
            raise ValueError("nothing to predict")
        out = pd.concat(parts, ignore_index=True)
        out["readable"] = out["n_active_features"] > 0
        return out

    # -- persistence --------------------------------------------------------

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.cfg.save(os.path.join(directory, "model_config.json"))
        joblib.dump({"keep_vocab": self.keep_vocab, "vectorizer": self.vectorizer,
                     "hierarchy": self.hierarchy}, os.path.join(directory, "features.joblib"),
                    compress=3)
        if self.cfg.model_type == "nb":
            joblib.dump(self.models, os.path.join(directory, "nb_models.joblib"), compress=3)
        else:
            import torch
            torch.save({"init_kwargs": self.net.init_kwargs, "state_dict": self.net.state_dict()},
                       os.path.join(directory, "nn_model.pt"))
        if self.history is not None:
            self.history.to_csv(os.path.join(directory, "history.csv"), index=False)

    @classmethod
    def load(cls, directory: str) -> "CategoryClassifier":
        clf = cls(ModelConfig.load(os.path.join(directory, "model_config.json")))
        feats = joblib.load(os.path.join(directory, "features.joblib"))
        clf.keep_vocab, clf.vectorizer, clf.hierarchy = (feats["keep_vocab"], feats["vectorizer"],
                                                         feats["hierarchy"])
        if clf.cfg.model_type == "nb":
            clf.models = joblib.load(os.path.join(directory, "nb_models.joblib"))
        else:
            import torch
            from .nn import SparseInputMLP
            blob = torch.load(os.path.join(directory, "nn_model.pt"), map_location="cpu")
            clf.net = SparseInputMLP(**blob["init_kwargs"])
            clf.net.load_state_dict(blob["state_dict"])
            clf.net.eval()
        hist = os.path.join(directory, "history.csv")
        if os.path.exists(hist):
            clf.history = pd.read_csv(hist)
        return clf


def versions() -> dict:
    import scipy
    import sklearn
    out = {"python": platform.python_version(), "pandas": pd.__version__,
           "numpy": np.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}
    try:
        import torch
        out["torch"] = torch.__version__
    except ImportError:
        pass
    return out


# ---------------------------------------------------------------------------
# The deployable bundle: model + cleaner + configs + ID->name lookup
# ---------------------------------------------------------------------------

def save_bundle(directory: str, clf: CategoryClassifier, cleaner, cleaning_config: dict,
                lookup: Optional[pd.DataFrame] = None, manifest: Optional[dict] = None) -> None:
    """Everything prediction needs, in one directory, so notebook 05 cannot
    clean, prune or vectorize differently from training."""
    from .io import save_json
    clf.save(directory)
    joblib.dump(cleaner, os.path.join(directory, "cleaner.joblib"), compress=3)
    save_json(cleaning_config, os.path.join(directory, "cleaning_config.json"))
    if lookup is not None:
        lookup.to_csv(os.path.join(directory, "hierarchy_lookup.csv"), index=False)
    save_json({**(manifest or {}), "versions": versions()}, os.path.join(directory, "manifest.json"))


def load_bundle(directory: str) -> dict:
    from .io import load_json
    lookup_path = os.path.join(directory, "hierarchy_lookup.csv")
    return {
        "clf": CategoryClassifier.load(directory),
        "cleaner": joblib.load(os.path.join(directory, "cleaner.joblib")),
        "cleaning_config": load_json(os.path.join(directory, "cleaning_config.json")),
        "lookup": (pd.read_csv(lookup_path, dtype=str, keep_default_na=False)
                   if os.path.exists(lookup_path) else None),
        "manifest": load_json(os.path.join(directory, "manifest.json")),
    }


def hierarchy_lookup(df: pd.DataFrame, level_columns: List[str], name_columns: List[str]) -> pd.DataFrame:
    """Distinct ID/name rows, for turning predicted IDs back into names."""
    cols = [c for c in list(level_columns) + list(name_columns) if c in df.columns]
    return df[cols].drop_duplicates().reset_index(drop=True)


def attach_names(pred: pd.DataFrame, lookup: pd.DataFrame, level_columns: List[str],
                 name_columns: List[str], prefix: str = "Predicted ") -> pd.DataFrame:
    """Add 'Predicted <Level N Name>' next to each predicted ID."""
    out = pred.copy()
    for id_col, name_col in zip(level_columns, name_columns):
        if id_col not in lookup.columns or name_col not in lookup.columns:
            continue
        m = lookup[[id_col, name_col]].drop_duplicates(id_col).set_index(id_col)[name_col]
        pos = out.columns.get_loc(prefix + id_col) + 1
        out.insert(pos, prefix + name_col, out[prefix + id_col].map(m))
    return out
