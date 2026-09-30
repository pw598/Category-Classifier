"""The label hierarchy, and turning per-level probabilities into paths.

Both models produce one probability distribution per level. A path's
score combines them, and only paths seen in training can be predicted,
so the four levels of an answer always agree with each other.

    independent   score(path) = prod_d p_d(c_d)
    conditional   score(path) = p_1(c_1) * prod_{d>1} p_d(c_d) / sum of p_d
                  over c_d's siblings -- each level renormalised within the
                  parent the path passes through

Scores are normalised over the valid paths (unless normalise=False), and
the confidence at depth d is the mass of every path sharing the predicted
path's prefix down to d. It can only fall with depth.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

EPS = 1e-12


class LabelHierarchy:
    """Classes per level and the valid paths, built from training rows."""

    def __init__(self, level_columns: Sequence[str], classes: List[np.ndarray], paths: np.ndarray):
        self.level_columns = list(level_columns)
        self.classes = classes                      # per level, sorted string labels
        self.paths = paths                          # (n_paths, n_levels) int codes
        self._index = [{c: i for i, c in enumerate(cl)} for cl in classes]
        self._build_structures()

    @classmethod
    def from_frame(cls, df: pd.DataFrame, level_columns: Sequence[str]) -> "LabelHierarchy":
        labels = df[list(level_columns)].astype(str)
        classes = [np.array(sorted(labels[c].unique())) for c in level_columns]
        codes = np.column_stack([
            pd.Categorical(labels[c], categories=classes[i]).codes
            for i, c in enumerate(level_columns)
        ]).astype(np.int64)
        paths = np.unique(codes, axis=0)
        return cls(level_columns, classes, paths)

    def _build_structures(self) -> None:
        n_paths, L = self.paths.shape
        # prefix id of every path at every depth, and a (paths x prefixes)
        # indicator for summing path mass into prefixes
        self.prefix_ids, self.prefix_sum = [], []
        for d in range(L):
            _, inv = np.unique(self.paths[:, :d + 1], axis=0, return_inverse=True)
            inv = inv.ravel()
            self.prefix_ids.append(inv)
            self.prefix_sum.append(sparse.csr_matrix(
                (np.ones(n_paths, dtype=np.float64), (np.arange(n_paths), inv)),
                shape=(n_paths, inv.max() + 1)))
        # parent of each class, for conditional mode. A class under two
        # parents (a data error in a tree) keeps the first.
        self.parent_of = [None]
        for d in range(1, L):
            parent = np.full(self.n_classes(d), -1, dtype=np.int64)
            for child, par in self.paths[:, [d, d - 1]][::-1]:
                parent[child] = par
            self.parent_of.append(parent)
        n_multi = sum(
            int((pd.DataFrame(self.paths[:, [d - 1, d]]).drop_duplicates()[1]
                 .value_counts() > 1).sum()) for d in range(1, L))
        if n_multi:
            print(f"[hierarchy] WARNING: {n_multi} class(es) sit under more than one "
                  "parent; conditional mode uses one parent for each")

    @property
    def n_levels(self) -> int:
        return len(self.level_columns)

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    def n_classes(self, d: int) -> int:
        return len(self.classes[d])

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """(n_rows, n_levels) codes; -1 for a label unseen in training."""
        return np.column_stack([
            df[c].astype(str).map(self._index[d]).fillna(-1).to_numpy(dtype=np.int64)
            for d, c in enumerate(self.level_columns)
        ])

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame({"level": self.level_columns,
                             "classes": [self.n_classes(d) for d in range(self.n_levels)],
                             "prefixes": [m.shape[1] for m in self.prefix_sum]})


def score_paths(H: LabelHierarchy, level_probs: List[np.ndarray], mode: str = "independent",
                normalise: bool = True, batch_size: int = 2000) -> pd.DataFrame:
    """The best valid path per row, with confidences.

    Columns: 'Predicted <level>' per level, 'L<d> Confidence' (prefix mass
    through depth d), 'L<d> Level Prob' (that level's own probability for
    the class on the predicted path), and 'Path Log Evidence' (log of the
    unnormalised best score; very low values mean every path is
    implausible, whatever the normalised confidence says).
    """
    if mode not in ("independent", "conditional"):
        raise ValueError(f"mode must be 'independent' or 'conditional', got {mode!r}")
    L, n = H.n_levels, level_probs[0].shape[0]

    # conditional: sparse (classes_d x classes_{d-1}) parent indicator
    parent_mats = [None] + [
        sparse.csr_matrix((np.ones(H.n_classes(d)), (np.arange(H.n_classes(d)), np.maximum(H.parent_of[d], 0))),
                          shape=(H.n_classes(d), H.n_classes(d - 1)))
        for d in range(1, L)]

    best = np.empty(n, dtype=np.int64)
    conf = np.empty((n, L), dtype=np.float32)
    lvl = np.empty((n, L), dtype=np.float32)
    evidence = np.empty(n, dtype=np.float64)

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        logs = np.zeros((stop - start, H.n_paths), dtype=np.float64)
        for d in range(L):
            p = np.asarray(level_probs[d][start:stop], dtype=np.float64)
            if mode == "conditional" and d > 0:
                sibling_total = np.asarray((parent_mats[d].T @ p.T).T)
                p = p / (sibling_total[:, H.parent_of[d]] + EPS)
            logs += np.log(p[:, H.paths[:, d]] + EPS)

        top = logs.argmax(axis=1)
        top_log = logs[np.arange(len(top)), top]
        if normalise:
            w = np.exp(logs - top_log[:, None])
            w /= w.sum(axis=1, keepdims=True)
        else:
            w = np.exp(logs)
        rows = np.arange(len(top))
        for d in range(L):
            mass = np.asarray((H.prefix_sum[d].T @ w.T).T)     # (b, n_prefixes)
            conf[start:stop, d] = mass[rows, H.prefix_ids[d][top]]
            lvl[start:stop, d] = np.asarray(level_probs[d][start:stop])[rows, H.paths[top, d]]
        best[start:stop], evidence[start:stop] = top, top_log

    out = {f"Predicted {c}": H.classes[d][H.paths[best, d]] for d, c in enumerate(H.level_columns)}
    for d in range(L):
        out[f"L{d + 1} Confidence"] = conf[:, d]
    for d in range(L):
        out[f"L{d + 1} Level Prob"] = lvl[:, d]
    out["Path Log Evidence"] = evidence
    return pd.DataFrame(out)
