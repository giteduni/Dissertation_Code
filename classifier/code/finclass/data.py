"""Gold-label loading, Stage-1/Stage-2 view construction, and leakage-safe splits.

The single most important correctness property here is *no channel leakage*: the same
``channel_id`` must never straddle a train/test boundary, otherwise the model can learn
channel identity instead of the educational/misleading construct.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

import config

_REQUIRED = ["title", "description", "view_count", "like_count", "label"]


def load_gold(csv_path=None) -> pd.DataFrame:
    """Load the labelled CSV defensively and add parsed date / fallback grouping columns."""
    path = str(csv_path or config.DATA_CSV)
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Gold CSV missing required columns: {missing}")

    for col in ("view_count", "like_count"):
        df[col] = (
            df[col].astype(str).str.replace(",", "", regex=False).str.strip().replace("", "0")
        )
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["label"] = df["label"].str.strip().str.upper()

    if "upload_date" in df.columns:
        df["upload_dt"] = pd.to_datetime(
            df["upload_date"], format=config.UPLOAD_DATE_FORMAT, errors="coerce"
        )
    else:
        df["upload_dt"] = pd.NaT
    if "channel_id" not in df.columns:
        df["channel_id"] = ""
    df["channel_id"] = df["channel_id"].astype(str).str.strip()
    blank = df["channel_id"].isin(["", "nan", "none", "NaN"])
    df.loc[blank, "channel_id"] = ["__row_%d" % i for i in df.index[blank]]

    if "strategy" not in df.columns:
        df["strategy"] = "UNKNOWN"
    df["strategy"] = df["strategy"].astype(str).str.strip()

    return df



# Stage views

@dataclass
class StageData:
    """A modelling view: feature frame ``X``, binary target ``y``, groups, dates, strata."""

    X: pd.DataFrame
    y: np.ndarray  
    groups: np.ndarray
    dates: pd.Series
    strategy: pd.Series
    positive_name: str
    negative_name: str

    def __len__(self) -> int:
        return len(self.y)


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The columns the feature pipeline is allowed to see (gold label deliberately absent)."""
    return df[["title", "description", "view_count", "like_count"]].reset_index(drop=True)


def stage1_view(df: pd.DataFrame) -> StageData:
    """Stage 1: positive = NON_FINANCIAL, negative = FINANCIAL (the three financial labels)."""
    keep = df["label"].isin(list(config.FINANCIAL_LABELS) + [config.NON_FINANCIAL_LABEL])
    sub = df[keep].reset_index(drop=True)
    y = (sub["label"] == config.NON_FINANCIAL_LABEL).astype(int).to_numpy()
    return StageData(
        X=_feature_frame(sub),
        y=y,
        groups=sub["channel_id"].to_numpy(),
        dates=sub["upload_dt"].reset_index(drop=True),
        strategy=sub["strategy"].reset_index(drop=True),
        positive_name=config.NON_FINANCIAL_LABEL,
        negative_name="FINANCIAL",
    )


def stage2_view(df: pd.DataFrame, borderline_as: str | None = None) -> StageData:
    """Stage 2: financial videos only, positive = MISLEADING, negative = EDUCATIONAL.

    ``borderline_as`` follows ``config.BORDERLINE_AS`` unless overridden.
    """
    mode = (borderline_as or config.BORDERLINE_AS).lower()
    sub = df[df["label"].isin(config.FINANCIAL_LABELS)].copy()

    label = sub["label"].copy()
    if mode == "exclude":
        sub = sub[label != "BORDERLINE"]
    elif mode == "misleading":
        label = label.replace("BORDERLINE", config.POSITIVE_CLASS)
        sub["label"] = label
    elif mode == "educational":
        label = label.replace("BORDERLINE", config.NEGATIVE_CLASS)
        sub["label"] = label
    else: 
        raise ValueError(mode)

    sub = sub.reset_index(drop=True)
    y = (sub["label"] == config.POSITIVE_CLASS).astype(int).to_numpy()
    return StageData(
        X=_feature_frame(sub),
        y=y,
        groups=sub["channel_id"].to_numpy(),
        dates=sub["upload_dt"].reset_index(drop=True),
        strategy=sub["strategy"].reset_index(drop=True),
        positive_name=config.POSITIVE_CLASS,
        negative_name=config.NEGATIVE_CLASS,
    )



# Splits
def grouped_folds(data: StageData, n_splits: int | None = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield channel-grouped (train_idx, test_idx) pairs — no channel crosses the split."""
    n = n_splits or config.N_SPLITS
    n = min(n, np.unique(data.groups).size)
    gkf = GroupKFold(n_splits=n)
    yield from gkf.split(data.X, data.y, groups=data.groups)


def demo_holdout_split(df: pd.DataFrame, frac: float | None = None, seed: int | None = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Channel-disjoint (train_idx, holdout_idx) over the FULL gold frame.

    The holdout is excluded from the deployed model fit so the app's demo CSV is a
    genuinely out-of-sample illustration (Priority 1). Positional indices into ``df``.
    """
    f = config.DEMO_HOLDOUT_FRAC if frac is None else frac
    s = config.SEED if seed is None else seed
    gss = GroupShuffleSplit(n_splits=1, test_size=f, random_state=s)
    train_idx, holdout_idx = next(gss.split(df, groups=df["channel_id"].to_numpy()))
    return train_idx, holdout_idx


def three_way_group_split(
    groups: np.ndarray, test_size: float = 0.2, val_size: float = 0.15, seed: int | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Channel-disjoint (train_idx, val_idx, test_idx) for the expensive transformer (Model C).

    A single split (not full CV) is an accepted cost trade-off for a fine-tuned LM; the
    split is channel-disjoint so no channel leaks across train/val/test.
    """
    s = config.SEED if seed is None else seed
    idx_all = np.arange(len(groups))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=s)
    rest, test = next(gss1.split(idx_all, groups=groups))
    rel_val = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=s + 1)
    tr_rel, val_rel = next(gss2.split(rest, groups=groups[rest]))
    return rest[tr_rel], rest[val_rel], test


def normalise_title(title: str) -> str:
    """Lowercased, punctuation-stripped title used as a fuzzy fingerprint key."""
    return re.sub(r"[^a-z0-9]+", " ", str(title).lower()).strip()


def fingerprints(df: pd.DataFrame) -> dict:
    """Identity fingerprints (display_ids + hashed normalised titles) for overlap detection."""
    ids = []
    if "display_id" in df.columns:
        ids = [str(x).strip() for x in df["display_id"] if str(x).strip()]
    title_hashes = [
        hashlib.sha1(normalise_title(t).encode("utf-8")).hexdigest()
        for t in df.get("title", pd.Series([], dtype=str))
        if normalise_title(t)
    ]
    return {"display_ids": sorted(set(ids)), "title_hashes": sorted(set(title_hashes))}


def overlap_fraction(uploaded: pd.DataFrame, prints: dict) -> float:
    """Fraction of uploaded rows whose display_id or normalised title matches training."""
    if len(uploaded) == 0:
        return 0.0
    id_set = set(prints.get("display_ids", []))
    title_set = set(prints.get("title_hashes", []))
    cols = {c.lower(): c for c in uploaded.columns}
    id_col = cols.get("display_id")
    title_col = cols.get("title") or cols.get("video_title")

    hits = 0
    for _, row in uploaded.iterrows():
        match = False
        if id_col and str(row[id_col]).strip() in id_set:
            match = True
        elif title_col:
            h = hashlib.sha1(normalise_title(row[title_col]).encode("utf-8")).hexdigest()
            if h in title_set:
                match = True
        hits += int(match)
    return hits / len(uploaded)


def temporal_split(data: StageData, cutoff: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """(train_idx, test_idx) with train = uploaded before ``cutoff``, test = on/after.

    Rows with an unparseable date are placed in the training side (conservative: they are
    never used to inflate the forward-in-time test score).
    """
    cut = pd.Timestamp(cutoff or config.TEMPORAL_CUTOFF)
    dates = data.dates
    test_mask = dates.notna() & (dates >= cut)
    test_idx = np.where(test_mask.to_numpy())[0]
    train_idx = np.where(~test_mask.to_numpy())[0]
    return train_idx, test_idx
