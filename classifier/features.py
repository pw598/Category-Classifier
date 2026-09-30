"""Features: rare-token pruning and the sparse vectorizers.

Both are fitted on training rows only. Counting tokens across the whole
dataset would let held-out rows vote on which of their own tokens
survive, and the resulting score would be optimistic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .cleaning import tokenize


# ---------------------------------------------------------------------------
# Rare-token pruning (min_token_count)
# ---------------------------------------------------------------------------

def build_keep_vocab(texts: Iterable[str], min_count: int) -> set:
    """Tokens appearing in at least `min_count` descriptions."""
    counts: Counter = Counter()
    for t in texts:
        counts.update(set(tokenize(t)))
    return {tok for tok, n in counts.items() if n >= min_count}


def prune(texts: pd.Series, keep: Optional[set]) -> pd.Series:
    """Keep only tokens in `keep`, in order. None means no pruning."""
    if keep is None:
        return texts.fillna("").astype(str)
    return texts.fillna("").astype(str).map(lambda t: " ".join(w for w in tokenize(t) if w in keep))


# ---------------------------------------------------------------------------
# Order-insensitive bigrams
# ---------------------------------------------------------------------------

def fold_bigrams(tokens: List[str]) -> List[str]:
    """Sort the two terms of every bigram; unigrams and trigrams untouched."""
    out = []
    for tok in tokens:
        parts = tok.split(" ")
        if len(parts) == 2 and parts[0] > parts[1]:
            tok = f"{parts[1]} {parts[0]}"
        out.append(tok)
    return out


class UnorderedBigramCountVectorizer(CountVectorizer):
    def build_analyzer(self) -> Callable:
        base = super().build_analyzer()
        return lambda doc: fold_bigrams(base(doc))


class UnorderedBigramTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self) -> Callable:
        base = super().build_analyzer()
        return lambda doc: fold_bigrams(base(doc))


@dataclass
class VectorizerSettings:
    """Every vectorizer parameter, stated. int min_df is a document count;
    float max_df is a proportion."""

    kind: str = "count"                  # "count" | "tfidf"
    ngram_range: Tuple[int, int] = (1, 2)
    unordered_bigrams: bool = True       # 'ball valve' == 'valve ball'
    min_df: int = 3
    max_df: float = 1.0
    max_features: Optional[int] = None
    stop_words: Optional[str] = None     # "english" drops e.g. FRONT, BACK, BOTTOM, FILL
    binary: bool = True
    lowercase: bool = True
    token_pattern: str = r"(?u)\b\w\w+\b"
    # tfidf only
    sublinear_tf: bool = True
    norm: Optional[str] = "l2"
    use_idf: bool = True
    smooth_idf: bool = True

    def build(self):
        if isinstance(self.min_df, float) and self.min_df >= 1.0:
            raise ValueError(f"min_df={self.min_df} is a float, i.e. a proportion; use an int count")
        common = dict(ngram_range=tuple(self.ngram_range), min_df=self.min_df, max_df=self.max_df,
                      max_features=self.max_features, stop_words=self.stop_words,
                      binary=self.binary, lowercase=self.lowercase,
                      token_pattern=self.token_pattern, dtype=np.float32)
        if self.kind == "count":
            cls = UnorderedBigramCountVectorizer if self.unordered_bigrams else CountVectorizer
            return cls(**common)
        if self.kind == "tfidf":
            cls = UnorderedBigramTfidfVectorizer if self.unordered_bigrams else TfidfVectorizer
            return cls(sublinear_tf=self.sublinear_tf, norm=self.norm, use_idf=self.use_idf,
                       smooth_idf=self.smooth_idf, **common)
        raise ValueError(f"kind must be 'count' or 'tfidf', got {self.kind!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "VectorizerSettings":
        d = dict(d)
        d["ngram_range"] = tuple(d["ngram_range"])
        return cls(**d)
