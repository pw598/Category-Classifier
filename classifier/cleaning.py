"""Description cleaning: the recipe, and everything it reads.

Order of operations inside Cleaner.clean(), each step switchable:

    1. strip the leading token when it is not an English word (vendor
       codes and part numbers: 'QJX8A TORQUE HEAD')
    2. the regex pass, read from a pattern file in cleaning_files/patterns
    3. expand abbreviations          (abbrevs_to_expand.csv)
    4. normalize misspellings        (to_normalize.csv)
    5. filter to known tokens        (good_tokens.csv, and optionally
                                      English words / expansion words /
                                      normalization words)

Marker tokens (VND_, CAT_, PLINE_, UNSPSC2_ ...) are attached *after*
cleaning, by Cleaner.attach_markers, so the regex pass and the token
filter never see them.

The Cleaner is pickled with every model, reference lists and word list
included, so prediction cleans exactly as training did even if the files
on disk change later.
"""

from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# The step names, in the order they run. Used by trace() and describe().
STEPS = ("strip_leading_code", "regex", "expand_abbrevs",
         "normalize_misspellings", "filter_to_known_tokens")


# ---------------------------------------------------------------------------
# Reading the files
# ---------------------------------------------------------------------------

def load_term_list(path) -> List[str]:
    """One term per line after a header line; uppercased, blanks dropped."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as fh:
        if path.suffix.lower() == ".csv":
            rows = list(csv.reader(fh))[1:]
            values = [r[0] for r in rows if r]
        else:
            values = fh.read().splitlines()[1:]
    return [v.strip().upper() for v in values if v.strip()]


def load_reference_lists(directory) -> Dict[str, List[str]]:
    """Every .txt/.csv in `directory` as {file stem: [TERMS]}."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"reference-list directory not found: {directory}")
    lists = {p.stem: load_term_list(p) for p in sorted(directory.iterdir())
             if p.suffix.lower() in (".txt", ".csv")}
    if not lists:
        raise ValueError(f"no .txt or .csv lists in {directory}")
    return lists


def load_pairs(path) -> Dict[str, str]:
    """Two-column CSV with a header -> {FIND: replacement}.

    Keys are uppercased (descriptions are uppercase by the time these run);
    values keep their case, which does not matter downstream because the
    vectorizer lowercases. The first spelling of a duplicate key wins.
    """
    out: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            key = re.sub(r"\s+", " ", row[0].strip()).upper()
            out.setdefault(key, row[1].strip())
    if not out:
        raise ValueError(f"{path} contains no find/replace pairs")
    return out


def load_english_vocabulary() -> set:
    """NLTK's English word list, lowercased. Installs nltk if it is absent."""
    try:
        import nltk
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "nltk"])
        import nltk
    try:
        from nltk.corpus import words
        words.words()
    except LookupError:
        nltk.download("words", quiet=True)
        from nltk.corpus import words
    return {w.lower() for w in words.words()}


# ---------------------------------------------------------------------------
# Pattern files
# ---------------------------------------------------------------------------

_SECTION = re.compile(r"^\[(delete|replace)\]$")
_PLACEHOLDER = re.compile(r"<<(\w+)(:escaped)?>>")


def parse_pattern_file(path) -> List[Tuple[str, str, Optional[str]]]:
    """A pattern file as [(kind, pattern, replacement)], placeholders intact.

    kind is 'delete' (replacement None) or 'replace'. See
    cleaning_files/patterns/original.txt for the format.
    """
    steps, kind = [], None
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _SECTION.match(line)
        if m:
            kind = m.group(1)
            continue
        if kind is None:
            raise ValueError(f"{path}:{lineno}: pattern before any [delete]/[replace] header")
        if kind == "delete":
            steps.append(("delete", line, None))
        else:
            if "=>" not in line:
                raise ValueError(f"{path}:{lineno}: [replace] line needs '<regex> => \"text\"'")
            pattern, repl = line.rsplit("=>", 1)
            steps.append(("replace", pattern.strip(), ast.literal_eval(repl.strip())))
    if not steps:
        raise ValueError(f"{path} defines no patterns")
    return steps


def resolve_patterns(steps, lists: Dict[str, List[str]]) -> List[Tuple[str, str, Optional[str]]]:
    """Substitute <<list>> placeholders and check every pattern compiles."""
    def sub(m):
        name, escaped = m.group(1), m.group(2)
        if name not in lists:
            raise KeyError(f"pattern refers to <<{name}>>, but there is no list "
                           f"'{name}'. Lists available: {sorted(lists)}")
        terms = lists[name]
        return "|".join(re.escape(t) for t in terms) if escaped else "|".join(terms)

    out = []
    for kind, pattern, repl in steps:
        resolved = _PLACEHOLDER.sub(sub, pattern)
        try:
            re.compile(resolved)
        except re.error as exc:
            raise ValueError(f"pattern does not compile ({exc}): {pattern}") from exc
        out.append((kind, resolved, repl))
    return out


def apply_patterns(series: pd.Series, steps) -> pd.Series:
    out = series.astype(str)
    for kind, pattern, repl in steps:
        out = out.str.replace(pattern, "" if kind == "delete" else repl, regex=True)
    return out


# ---------------------------------------------------------------------------
# Individual steps
# ---------------------------------------------------------------------------

def first_word_is_real(text: str, vocabulary: set) -> bool:
    """False for a leading token that is <=2 characters, has a digit, or is
    not an English word -- i.e. a vendor code or part number."""
    if not isinstance(text, str) or not text.strip():
        return False
    first = text.split(maxsplit=1)[0].strip(".,;:!?\"'()[]{}").lower()
    if len(first) <= 2 or any(ch.isdigit() for ch in first):
        return False
    return first in vocabulary


def build_replacer(pairs: Dict[str, str]):
    """A function text -> text applying {FIND: replacement}, whole tokens only.

    One pass, so a replacement is never re-matched by a later entry and
    the result does not depend on the row order of the CSV. Simple
    alphanumeric keys are looked up per token (fast even at 20k+ entries);
    keys with spaces or punctuation go through one alternation, longest
    first, fenced so they cannot match inside a longer word.
    """
    simple = re.compile(r"^[A-Z0-9]+$")
    complex_terms = sorted((k for k in pairs if not simple.match(k)), key=len, reverse=True)
    branches = []
    if complex_terms:
        alts = "|".join(re.escape(t).replace(r"\ ", r"\s+") for t in complex_terms)
        branches.append(rf"(?<![A-Za-z0-9])(?:{alts})(?![A-Za-z0-9])")
    branches.append(r"[A-Za-z0-9]+")
    pattern = re.compile("|".join(branches), re.IGNORECASE)

    def substitute(m):
        text = m.group(0)
        return pairs.get(re.sub(r"\s+", " ", text).upper(), text)

    def replace(text):
        return pattern.sub(substitute, text) if isinstance(text, str) else text

    return replace


def words_of(values: Iterable[str]) -> set:
    """Uppercase whitespace-separated words from a collection of strings."""
    out = set()
    for v in values:
        out.update(str(v).upper().split())
    return out


# ---------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------

@dataclass
class Cleaner:
    """The whole cleaning recipe, self-contained and picklable.

    Build it with build_cleaner(); the fields hold resolved resources, not
    file paths, so nothing is re-read at prediction time.
    """

    settings: dict
    vocabulary: set = field(default_factory=set)
    pattern_steps: list = field(default_factory=list)
    expansions: dict = field(default_factory=dict)
    normalizations: dict = field(default_factory=dict)
    known_tokens: set = field(default_factory=set)
    markers: dict = field(default_factory=dict)

    def __post_init__(self):
        self._replacers = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_replacers"] = {}          # closures do not pickle; rebuilt lazily
        return state

    def _replacer(self, name):
        if name not in self._replacers:
            self._replacers[name] = build_replacer(getattr(self, name))
        return self._replacers[name]

    # -- the steps ----------------------------------------------------------

    def _step(self, name: str, s: pd.Series) -> pd.Series:
        if name == "strip_leading_code":
            voc = self.vocabulary
            return s.map(lambda t: t if first_word_is_real(t, voc)
                         else (t.split(maxsplit=1)[1] if len(t.split(maxsplit=1)) > 1 else ""))
        if name == "regex":
            return apply_patterns(s, self.pattern_steps).str.strip()
        if name == "expand_abbrevs":
            return s.map(self._replacer("expansions"))
        if name == "normalize_misspellings":
            return s.map(self._replacer("normalizations"))
        if name == "filter_to_known_tokens":
            known, min_len = self.known_tokens, self.settings.get("min_token_length", 2)
            return s.map(lambda t: " ".join(w for w in t.split()
                                            if len(w) >= min_len and w.upper() in known))
        raise KeyError(name)

    def active_steps(self) -> List[str]:
        return [n for n in STEPS if self.settings.get(n)]

    def clean(self, series: pd.Series) -> pd.Series:
        """Clean text, one output per input. Nothing is dropped here."""
        out = series.fillna("").astype(str)
        if self.settings.get("uppercase"):
            out = out.str.upper()
        for name in self.active_steps():
            out = self._step(name, out)
        return out.str.replace(r"\s+", " ", regex=True).str.strip()

    def trace(self, series: pd.Series) -> pd.DataFrame:
        """The text after each active step, for inspecting a sample."""
        out = series.fillna("").astype(str).reset_index(drop=True)
        if self.settings.get("uppercase"):
            out = out.str.upper()
        stages = {"input": out}
        for name in self.active_steps():
            out = self._step(name, out)
            stages[name] = out.str.replace(r"\s+", " ", regex=True).str.strip()
        return pd.DataFrame(stages)

    # -- markers ------------------------------------------------------------

    def attach_markers(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Prepend a token per marker column; a blank value emits nothing.

        markers = {column: {"prefix": "VND_"}}, or with "levels": [2, 4, 6, 8]
        for a hierarchical code, emitting e.g. UNSPSC2_46 UNSPSC4_4618 ...
        Values lose anything that is not a letter or digit, so the vectorizer
        keeps each marker as a single token.
        """
        if not self.markers:
            return df
        missing = [c for c in self.markers if c not in df.columns]
        if missing:
            raise KeyError(f"marker column(s) {missing} not in the frame; this "
                           f"cleaner attaches {list(self.markers)}")
        out = df.copy()
        pieces = []
        for col, spec in self.markers.items():
            value = out[col].fillna("").astype(str).str.strip()
            value = value.str.replace(r"\.0+$", "", regex=True).str.replace(r"\W+", "", regex=True)
            prefix = spec["prefix"]
            if spec.get("levels"):
                for n in spec["levels"]:
                    pieces.append((f"{prefix}{n}_" + value.str.slice(0, n))
                                  .where(value.str.len() >= n, ""))
            else:
                pieces.append((prefix + value).where(value != "", ""))
        joined = pieces[0]
        for p in pieces[1:]:
            joined = joined + " " + p
        out[text_col] = ((joined + " " + out[text_col].fillna("").astype(str))
                         .str.replace(r"\s+", " ", regex=True).str.strip())
        return out

    # -- a whole frame ------------------------------------------------------

    def clean_frame(self, df: pd.DataFrame, text_col: str, orig_col: Optional[str] = None,
                    drop_empty: bool = True, verbose: bool = True) -> pd.DataFrame:
        """Clean `text_col`, optionally drop rows left empty, then attach markers.

        Training calls this with drop_empty=True: a description that does not
        survive cleaning cannot teach anything, and dropping happens *before*
        markers so a row carrying only markers is dropped too. Prediction uses
        drop_empty=False, so every input gets an output row.
        """
        out = df.copy()
        if orig_col:
            out[orig_col] = out[text_col]
        n0 = len(out)
        out[text_col] = self.clean(out[text_col])
        if drop_empty:
            out = out[out[text_col] != ""]
            if verbose:
                print(f"[clean] {n0:,} rows -> {len(out):,} "
                      f"({n0 - len(out):,} dropped: nothing survived cleaning)")
        out = self.attach_markers(out, text_col)
        return out.reset_index(drop=True)

    def describe(self) -> str:
        s = self.settings
        lines = [f"steps: {' -> '.join(self.active_steps()) or '(none)'}",
                 f"pattern file: {s.get('pattern_file')} ({len(self.pattern_steps)} patterns)"]
        if s.get("strip_leading_code"):
            lines.append(f"English vocabulary: {len(self.vocabulary):,} words")
        if s.get("expand_abbrevs"):
            lines.append(f"expansions: {len(self.expansions):,}")
        if s.get("normalize_misspellings"):
            lines.append(f"normalizations: {len(self.normalizations):,}")
        if s.get("filter_to_known_tokens"):
            lines.append(f"known tokens: {len(self.known_tokens):,} "
                         f"from {s.get('known_token_sources')}")
        lines.append(f"markers: {self.markers or 'none'}")
        return "\n".join(lines)


def build_cleaner(
    cleaning_files_dir: str = "../cleaning_files",
    pattern_file: Optional[str] = "original.txt",
    uppercase: bool = False,
    strip_leading_code: bool = True,
    regex: bool = True,
    expand_abbrevs: bool = False,
    normalize_misspellings: bool = True,
    filter_to_known_tokens: bool = True,
    known_token_sources: Sequence[str] = ("good_tokens", "expansions", "normalizations"),
    min_token_length: int = 2,
    markers: Optional[dict] = None,
    expansions_file: str = "abbrevs_to_expand.csv",
    normalizations_file: str = "to_normalize.csv",
    good_tokens_file: str = "good_tokens.csv",
) -> Cleaner:
    """Resolve every file the recipe needs and return a self-contained Cleaner.

    known_token_sources: any of "good_tokens", "english" (the NLTK word
    list), "expansions" (words produced by abbreviation expansion) and
    "normalizations" (words produced by normalization). Without the last
    two, the filter would delete the output of the steps before it.
    """
    root = Path(cleaning_files_dir)
    settings = dict(
        pattern_file=pattern_file if regex else None, uppercase=uppercase,
        strip_leading_code=strip_leading_code, regex=regex,
        expand_abbrevs=expand_abbrevs, normalize_misspellings=normalize_misspellings,
        filter_to_known_tokens=filter_to_known_tokens,
        known_token_sources=list(known_token_sources), min_token_length=min_token_length,
        expansions_file=expansions_file, normalizations_file=normalizations_file,
        good_tokens_file=good_tokens_file,
    )
    sources = set(known_token_sources) if filter_to_known_tokens else set()

    vocabulary = set()
    if strip_leading_code or "english" in sources:
        vocabulary = load_english_vocabulary()

    pattern_steps = []
    if regex:
        lists = load_reference_lists(root / "lists")
        pattern_steps = resolve_patterns(parse_pattern_file(root / "patterns" / pattern_file), lists)

    expansions = load_pairs(root / expansions_file) if (expand_abbrevs or "expansions" in sources) else {}
    normalizations = (load_pairs(root / normalizations_file)
                      if (normalize_misspellings or "normalizations" in sources) else {})

    known = set()
    if "good_tokens" in sources:
        known |= set(load_term_list(root / good_tokens_file))
    if "english" in sources:
        known |= {w.upper() for w in vocabulary}
    if "expansions" in sources and expand_abbrevs:
        known |= words_of(expansions.values())
    if "normalizations" in sources and normalize_misspellings:
        known |= words_of(normalizations.values())
    unknown = sources - {"good_tokens", "english", "expansions", "normalizations"}
    if unknown:
        raise ValueError(f"unknown known_token_sources {sorted(unknown)}")

    return Cleaner(settings=settings, vocabulary=vocabulary, pattern_steps=pattern_steps,
                   expansions=expansions if expand_abbrevs else {},
                   normalizations=normalizations if normalize_misspellings else {},
                   known_tokens=known, markers=dict(markers or {}))


def build_markers(vendor: bool = False, prodcat: bool = False, prodline: bool = False,
                  unspsc: bool = False, unspsc_levels=(2, 4, 6, 8),
                  vendor_col: str = "Global Vendno", prodcat_col: str = "ICSP Prodcat",
                  prodline_col: str = "Prodline", unspsc_col: str = "UNSPSC") -> dict:
    """The markers dict from on/off switches."""
    markers = {}
    if vendor:
        markers[vendor_col] = {"prefix": "VND_"}
    if prodcat:
        markers[prodcat_col] = {"prefix": "CAT_"}
    if prodline:
        markers[prodline_col] = {"prefix": "PLINE_"}
    if unspsc:
        markers[unspsc_col] = {"prefix": "UNSPSC", "levels": [int(n) for n in unspsc_levels]}
    return markers


# ---------------------------------------------------------------------------
# Filters and counts
# ---------------------------------------------------------------------------

def drop_rare_paths(df: pd.DataFrame, level_columns: List[str], min_path_count: int,
                    verbose: bool = True) -> pd.DataFrame:
    """Drop rows whose full hierarchy path occurs fewer than min_path_count times."""
    if min_path_count <= 1:
        return df.reset_index(drop=True)
    size = df.groupby(level_columns, dropna=False)[level_columns[0]].transform("size")
    out = df[size >= min_path_count].reset_index(drop=True)
    if verbose:
        p0 = df[level_columns].drop_duplicates().shape[0]
        p1 = out[level_columns].drop_duplicates().shape[0]
        print(f"[paths] min_path_count={min_path_count}: {len(df):,} -> {len(out):,} rows, "
              f"{p0:,} -> {p1:,} paths")
    return out


TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")      # the vectorizer's default


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower()) if isinstance(text, str) else []


def count_ngrams(texts: Iterable[str], n: int, unordered: bool = False,
                 min_occurrences: int = 1) -> pd.Series:
    """Document frequency of each n-gram, as the vectorizer would tokenize.

    unordered=True folds each bigram to its sorted form, so 'ball valve'
    and 'valve ball' count as one -- the view the unordered_bigrams
    vectorizer setting gives the model.
    """
    from collections import Counter
    counts: Counter = Counter()
    for text in texts:
        toks = tokenize(text)
        grams = set()
        for i in range(len(toks) - n + 1):
            g = toks[i:i + n]
            if unordered and n == 2:
                g = sorted(g)
            grams.add(" ".join(g))
        counts.update(grams)
    s = pd.Series(counts, dtype="int64").sort_values(ascending=False)
    return s[s >= min_occurrences] if min_occurrences > 1 else s


def ngram_summary(texts: pd.Series, min_occurrences: int = 1) -> Tuple[pd.DataFrame, dict]:
    """Unique unigram / bigram (both views) / trigram counts, and the counts themselves."""
    counts = {
        "unigrams": count_ngrams(texts, 1, min_occurrences=min_occurrences),
        "bigrams (ordered)": count_ngrams(texts, 2, min_occurrences=min_occurrences),
        "bigrams (unordered)": count_ngrams(texts, 2, unordered=True, min_occurrences=min_occurrences),
        "trigrams": count_ngrams(texts, 3, min_occurrences=min_occurrences),
    }
    table = pd.DataFrame([{"ngram": k, "unique": len(v), "min_occurrences": min_occurrences}
                          for k, v in counts.items()])
    return table, counts
