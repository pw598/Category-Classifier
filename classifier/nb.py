"""Multinomial Naive Bayes that stays sparse.

sklearn's MultinomialNB stores two dense (n_classes x n_features)
float64 arrays. At 3,500 leaf classes and 125,000 features that is
7 GB for one level. The same model can be held sparsely, because

    log P(f | c) = log(N_cf + a) - log(N_c + a*F)
                 = log(a) - log(N_c + a*F) + log1p(N_cf / a)

and the last term is zero wherever class c never saw feature f. So the
joint log-likelihood for a row x is

    x @ S.T  +  sum(x) * (log a - log(N_c + a*F))  +  log prior(c)

with S = log1p(N / a) holding only the observed (class, feature) pairs.
The arithmetic is identical to sklearn's; memory scales with the number
of observed pairs rather than classes x features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
from scipy import sparse


@dataclass
class NBSettings:
    alpha: float = 0.001                  # additive smoothing
    fit_prior: bool = True                # learn class frequencies; False = uniform
    class_prior: Optional[List[float]] = None

    def to_dict(self):
        return asdict(self)


class SparseMultinomialNB:
    def __init__(self, alpha: float = 0.001, fit_prior: bool = True,
                 class_prior: Optional[List[float]] = None):
        self.alpha, self.fit_prior, self.class_prior = float(alpha), fit_prior, class_prior

    def fit(self, X, y: np.ndarray, n_classes: int) -> "SparseMultinomialNB":
        X = sparse.csr_matrix(X, dtype=np.float64)
        n, F = X.shape
        keep = y >= 0                                    # -1 = unknown label
        rows = np.flatnonzero(keep)
        Y = sparse.csr_matrix((np.ones(len(rows)), (y[keep], rows)), shape=(n_classes, n))
        counts = (Y @ X).tocsr()                         # (classes x features), sparse
        counts.eliminate_zeros()

        self.n_features_ = F
        self.class_count_ = np.bincount(y[keep], minlength=n_classes).astype(np.float64)
        self.log_denominator_ = np.log(np.asarray(counts.sum(axis=1)).ravel() + self.alpha * F)
        S = counts.copy()
        S.data = np.log1p(S.data / self.alpha)
        self.S_T_ = S.T.tocsr()                          # (features x classes)
        self.row_term_ = (np.log(self.alpha) - self.log_denominator_).astype(np.float64)

        if self.class_prior is not None:
            prior = np.asarray(self.class_prior, dtype=np.float64)
            if len(prior) != n_classes:
                raise ValueError(f"class_prior has {len(prior)} values for {n_classes} classes")
            self.class_log_prior_ = np.log(prior)
        elif self.fit_prior:
            with np.errstate(divide="ignore"):
                self.class_log_prior_ = np.log(self.class_count_) - np.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))
        return self

    def joint_log_likelihood(self, X) -> np.ndarray:
        X = sparse.csr_matrix(X, dtype=np.float64)
        row_sum = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
        jll = (X @ self.S_T_).toarray()
        jll += row_sum[:, None] * self.row_term_[None, :]
        jll += self.class_log_prior_[None, :]
        return jll

    def predict_proba(self, X) -> np.ndarray:
        jll = self.joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)
        p = np.exp(jll)
        p /= p.sum(axis=1, keepdims=True)
        return p.astype(np.float32)

    def nbytes(self) -> int:
        m = self.S_T_
        return m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
