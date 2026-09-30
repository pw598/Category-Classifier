"""Reading and writing: tables, JSON, ID lists, datestamps.

Everything is read as strings. Category IDs like '00123' lose their
leading zeros the moment anything guesses a numeric dtype, and the
resulting key mismatch is silent.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# What a NULL looks like once a frame has been stringified.
NULL_STRINGS = frozenset({"", "none", "nan", "nat", "null", "<na>", "n/a", "#n/a"})


def stamp() -> str:
    """One datestamp per run, used in every file name the run writes."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def blank_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Stringified nulls ('None', 'nan', ...) back to empty strings."""
    out = df.copy()
    for col in out.columns:
        s = out[col].astype(str)
        out[col] = s.where(~s.str.strip().str.lower().isin(NULL_STRINGS), "")
    return out


def spark_to_pandas(sdf) -> pd.DataFrame:
    """A Spark DataFrame as all-string pandas, nulls blanked."""
    return blank_nulls(sdf.toPandas().astype(str))


def read_table(path, delimiter: str = "|") -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=delimiter, dtype=str, keep_default_na=False)
    return blank_nulls(df)


def write_table(df: pd.DataFrame, path, delimiter: str = "|") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=delimiter, index=False)
    print(f"[io] wrote {len(df):,} rows to {path}")


def _jsonable(obj):
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if hasattr(obj, "item"):          # numpy scalars
        return obj.item()
    return str(obj)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_jsonable)
    print(f"[io] wrote {path}")


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_id_list(path, id_col: str) -> set:
    """IDs from a file: one per line, or a delimited file with an `id_col` column.

    Delimiter (comma, tab or pipe) is detected from the first line. A
    header is recognised when it contains `id_col`; otherwise the first
    column is used, skipping a first line that reads 'id'.
    """
    with open(path, encoding="utf-8-sig") as fh:
        first = fh.readline()
    delim = max(",\t|", key=first.count)
    if first.count(delim) == 0:
        delim = ","
    raw = pd.read_csv(path, sep=delim, header=None, dtype=str, keep_default_na=False,
                      encoding="utf-8-sig", skip_blank_lines=True)
    header = [str(v).strip().strip('"') for v in raw.iloc[0]]
    if id_col in header:
        values = raw.iloc[1:, header.index(id_col)]
    else:
        values = raw.iloc[:, 0]
        if header[0].lower() in ("id", "ids"):
            values = values.iloc[1:]
    ids = {str(v).strip().strip('"') for v in values}
    ids.discard("")
    print(f"[ids] {len(ids):,} IDs from {path}")
    return ids


def drop_ids(df: pd.DataFrame, ids: Iterable[str], id_col: str, label: str = "rows") -> pd.DataFrame:
    """Remove rows whose id_col is in `ids`, reporting what matched."""
    ids = set(ids)
    hit = df[id_col].astype(str).isin(ids)
    print(f"[ids] removed {int(hit.sum()):,} {label} "
          f"({len(ids):,} IDs listed, {len(ids - set(df.loc[hit, id_col])):,} not found)")
    if ids and not hit.any():
        print("[ids] WARNING: no listed ID matched. Check the ID column and file format.")
    return df[~hit].reset_index(drop=True)


def split_off_ids(df: pd.DataFrame, ids: Iterable[str], id_col: str):
    """(rows not in ids, rows in ids)."""
    hit = df[id_col].astype(str).isin(set(ids))
    return df[~hit].reset_index(drop=True), df[hit].reset_index(drop=True)


def latest(directory, prefix: str) -> Optional[str]:
    """Most recent file or folder in `directory` starting with `prefix` (by name,
    which sorts by datestamp)."""
    names = sorted(n for n in os.listdir(directory) if n.startswith(prefix))
    return os.path.join(directory, names[-1]) if names else None


def load_cleaning(config_dir, cleaning_stamp: str):
    """(cleaning config dict, Cleaner) written by notebook 01 for that stamp."""
    import joblib
    cfg = load_json(os.path.join(config_dir, f"cleaning_{cleaning_stamp}.json"))
    cleaner = joblib.load(os.path.join(config_dir, f"cleaner_{cleaning_stamp}.joblib"))
    return cfg, cleaner


def load_labelled(data_dir, cleaning_config: dict, id_col: str,
                  blacklist_file: Optional[str] = None, exclude_file: Optional[str] = None) -> pd.DataFrame:
    """The cleaned labelled data of a cleaning run, less blacklisted and excluded IDs."""
    df = read_table(os.path.join(data_dir, cleaning_config["cleaned_file"]))
    print(f"[data] {len(df):,} rows from {cleaning_config['cleaned_file']}")
    if blacklist_file:
        df = drop_ids(df, load_id_list(os.path.join(data_dir, blacklist_file), id_col), id_col, "blacklisted rows")
    if exclude_file:
        df = drop_ids(df, load_id_list(os.path.join(data_dir, exclude_file), id_col), id_col, "excluded rows")
    return df
