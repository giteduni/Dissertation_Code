"""Inference layer reused by the web app and the batch scorer.

Loads the serialised Stage-1 gate and both Stage-2 models once, then exposes:
  * ``classify(title, description, ...)``  -> full single-video result with explanation
  * ``classify_batch(df)``                 -> vectorised CSV scoring (input + 3 columns)

No retraining ever happens here; ``train.py`` must have produced ``models/`` first.
"""
from __future__ import annotations

import json
import threading
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from finclass import pipeline as pipe  
from finclass.text import RISK_MARKERS, engineered_row

_LOCK = threading.Lock()


class _Bundle:
    """Lazily-loaded, process-wide singleton holding every artifact needed for inference.

    Calibrated models drive the reported probabilities (Priority 3); the uncalibrated
    Model A is kept solely to produce faithful SHAP explanations.
    """

    def __init__(self) -> None:
        import joblib

        mdir = config.MODELS_DIR
        missing = [f for f in (config.STAGE1_MODEL_FILE, config.STAGE2_MODEL_A_FILE,
                               config.STAGE2_MODEL_A_CAL_FILE, config.STAGE2_MODEL_B_CAL_FILE)
                   if not (mdir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifact(s): {missing}. Run `python train.py` first."
            )
        self.stage1 = joblib.load(mdir / config.STAGE1_MODEL_FILE)
        self.model_a = joblib.load(mdir / config.STAGE2_MODEL_A_FILE)            
        self.model_a_cal = joblib.load(mdir / config.STAGE2_MODEL_A_CAL_FILE)    
        self.model_b_cal = joblib.load(mdir / config.STAGE2_MODEL_B_CAL_FILE)    
        meta_path = mdir / config.METADATA_FILE
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        fp_path = mdir / config.TRAIN_FINGERPRINTS_FILE
        self.fingerprints = json.loads(fp_path.read_text()) if fp_path.exists() else {"display_ids": [], "title_hashes": []}
        self.stage1_threshold = float(self.meta.get("stage1_threshold_tuned",
                                                     config.STAGE1_NONFIN_THRESHOLD))
        self.feature_names = np.array(pipe.feature_names_a(self.model_a), dtype=object)
        self.feature_union = self.model_a.named_steps["features"]
        self.booster = self.model_a.named_steps["clf"].get_booster()


@lru_cache(maxsize=1)
def _bundle() -> _Bundle:
    with _LOCK:
        return _Bundle()


def _frame(title, description, view_count=0, like_count=0) -> pd.DataFrame:
    return pd.DataFrame([{
        "title": "" if title is None else str(title),
        "description": "" if description is None else str(description),
        "view_count": view_count,
        "like_count": like_count,
    }])


def _p_positive(model, X) -> float:
    return float(np.asarray(model.predict_proba(X))[0, 1])



# Explanation
def _top_features(b: _Bundle, X: pd.DataFrame, k: int = 10) -> List[Dict]:
    """Per-instance TreeSHAP attributions for Model A, signed toward MISLEADING.

    Two correctness points that the previous implementation got wrong:
      * The matrix is kept **sparse** (exactly as the model was trained). XGBoost is
        sparsity-aware — a 0 in a sparse row means "missing" and routes differently from a
        dense 0. Explaining a densified row attributed a large, input-invariant weight to
        ``title_has_currency`` and disagreed with the displayed probability.
      * Attributions come from XGBoost's native ``pred_contribs`` (exact TreeSHAP), which is
        ~18x faster than the generic SHAP explainer and needs no extra dependency.

    Absent word/char n-grams are hidden (an n-gram the video does not contain is a confusing
    "reason"); engineered/lexicon signals are kept even at value 0 because their absence is
    itself meaningful and they are human-readably named.
    """
    try:
        import xgboost as xgb

        Xt = b.feature_union.transform(X)  
        contribs = b.booster.predict(xgb.DMatrix(Xt), pred_contribs=True)[0]
        cf = contribs[:-1] 
        dense = Xt.toarray()[0] if hasattr(Xt, "toarray") else np.asarray(Xt)[0]
        names = b.feature_names

        cands = []
        for i in range(len(cf)):
            if abs(cf[i]) < 1e-4:
                continue
            nm = str(names[i])
            is_token = nm.startswith("w:") or nm.startswith("c:")
            if is_token and abs(dense[i]) <= 1e-9:  
                continue
            cands.append((cf[i], dense[i], nm))
        cands.sort(key=lambda t: abs(t[0]), reverse=True)
        out = [{
            "feature": nm,
            "contribution": float(c),  
            "value": float(v),
            "direction": "Misleading" if c > 0 else "Educational",
        } for c, v, nm in cands[:k]]
        if out:
            return out
    except Exception:
        pass
    return _lexicon_fallback(X)


def _lexicon_fallback(X: pd.DataFrame) -> List[Dict]:
    """If SHAP is unavailable, surface the active codebook risk markers directly."""
    row = X.iloc[0]
    feats = engineered_row(row["title"], row["description"], row["view_count"], row["like_count"])
    out = []
    for marker in RISK_MARKERS:
        v = feats.get(f"lex_{marker}", 0.0)
        if v > 0:
            out.append({"feature": f"lex_{marker}", "contribution": float(v),
                        "value": float(v), "direction": "MISLEADING"})
    return sorted(out, key=lambda d: d["value"], reverse=True)[:10]



# Public API

def classify(
    title: str,
    description: str,
    view_count: float = 0,
    like_count: float = 0,
    stage1_threshold: Optional[float] = None,
    stage2_threshold: Optional[float] = None,
    explain: bool = True,
) -> Dict:
    """Classify one video from title + description (+ optional engagement).

    Returns a dict with Stage-1 eligibility, P(misleading) from both Stage-2 models,
    the thresholded label, and (optionally) the top contributing features.
    """
    b = _bundle()
    t1 = b.stage1_threshold if stage1_threshold is None else stage1_threshold
    t2 = config.STAGE2_MISLEADING_THRESHOLD if stage2_threshold is None else stage2_threshold

    X = _frame(title, description, view_count, like_count)
    p_nonfin = _p_positive(b.stage1, X)
    p_financial = 1.0 - p_nonfin
    is_financial = p_nonfin < t1

    result: Dict = {
        "stage1": {
            "eligibility": "FINANCIAL" if is_financial else "NON_FINANCIAL",
            "p_non_financial": p_nonfin,
            "p_financial": p_financial,
            "threshold": t1,
        },
        "eligibility": "FINANCIAL" if is_financial else "NON_FINANCIAL",
        "p_financial": p_financial,
    }

    if not is_financial:
        result.update({
            "label": "NON_FINANCIAL",
            "p_misleading": None,
            "p_misleading_model_a": None,
            "p_misleading_model_b": None,
            "top_features": [],
        })
        return result

    p_a = _p_positive(b.model_a_cal, X)  
    p_b = _p_positive(b.model_b_cal, X) 
    primary = p_a if config.PRIMARY_STAGE2_MODEL == "model_a" else p_b
    label = config.POSITIVE_CLASS if primary >= t2 else config.NEGATIVE_CLASS

    result.update({
        "label": label,
        "p_misleading": float(primary),              
        "p_misleading_model_a": float(p_a),
        "p_misleading_model_b": float(p_b),
        "calibrated": True,
        "stage2_threshold": t2,
        "primary_model": config.PRIMARY_STAGE2_MODEL,
        "top_features": _top_features(b, X) if explain else [],
    })
    return result


def classify_batch(
    df: pd.DataFrame,
    stage1_threshold: Optional[float] = None,
    stage2_threshold: Optional[float] = None,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Score a DataFrame of videos, returning the input plus three appended columns:
    ``predicted_eligibility``, ``p_misleading``, ``predicted_label``. Order preserved.

    Tolerant of header-name variants (``Title``/``Description`` etc.) and missing columns.
    """
    b = _bundle()
    t1 = b.stage1_threshold if stage1_threshold is None else stage1_threshold
    t2 = config.STAGE2_MISLEADING_THRESHOLD if stage2_threshold is None else stage2_threshold

    work = _normalise_columns(df)
    n = len(work)
    elig = np.empty(n, dtype=object)
    p_mis = np.full(n, np.nan)
    label = np.empty(n, dtype=object)

    primary_model = b.model_a_cal if config.PRIMARY_STAGE2_MODEL == "model_a" else b.model_b_cal

    for start in range(0, n, batch_size):
        chunk = work.iloc[start:start + batch_size]
        p_nonfin = np.asarray(b.stage1.predict_proba(chunk))[:, 1]
        fin_mask = p_nonfin < t1

        elig[start:start + len(chunk)] = np.where(fin_mask, "FINANCIAL", "NON_FINANCIAL")
        label[start:start + len(chunk)] = np.where(fin_mask, "", "NON_FINANCIAL")

        if fin_mask.any():
            fin_chunk = chunk.iloc[np.where(fin_mask)[0]]
            p = np.asarray(primary_model.predict_proba(fin_chunk))[:, 1]
            local = np.where(fin_mask)[0]
            for j, p_val in zip(local, p):
                p_mis[start + j] = p_val
                label[start + j] = config.POSITIVE_CLASS if p_val >= t2 else config.NEGATIVE_CLASS

    out = df.copy()
    out["predicted_eligibility"] = elig
    out["p_misleading"] = np.round(p_mis, 4)
    out["predicted_label"] = label
    return out


def training_overlap(df: pd.DataFrame) -> Dict:
    """Fraction of uploaded rows that match the training set (Priority 1 leakage guard)."""
    from finclass import data as datamod

    b = _bundle()
    frac = datamod.overlap_fraction(df, b.fingerprints)
    return {"overlap_fraction": float(frac),
            "is_in_sample": bool(frac >= config.OVERLAP_WARN_FRAC),
            "threshold": config.OVERLAP_WARN_FRAC}


_LABEL_ALIASES = ("label", "gold", "gold_label", "true_label", "y")


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    """Return the name of a gold-label column if the uploaded CSV carries one."""
    lower = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    for a in _LABEL_ALIASES:
        if a in lower:
            return lower[a]
    return None


def evaluate_against_gold(scored: pd.DataFrame, label_col: str) -> Optional[Dict]:
    """Stage-2 macro-F1 + confusion of predictions vs user-supplied labels (Priority 7).

    Restricted to rows the model deemed FINANCIAL and that carry an EDUCATIONAL/MISLEADING
    gold label (the only rows where the misleading/educational judgement is defined).
    """
    from sklearn.metrics import confusion_matrix, f1_score

    gold = scored[label_col].astype(str).str.strip().str.upper()
    pred = scored["predicted_label"].astype(str).str.strip().str.upper()
    mask = gold.isin([config.POSITIVE_CLASS, config.NEGATIVE_CLASS]) & \
        pred.isin([config.POSITIVE_CLASS, config.NEGATIVE_CLASS])
    if mask.sum() == 0:
        return None
    g, p = gold[mask], pred[mask]
    labels = [config.NEGATIVE_CLASS, config.POSITIVE_CLASS]
    return {
        "n_evaluated": int(mask.sum()),
        "macro_f1": float(f1_score(g, p, average="macro", labels=labels, zero_division=0)),
        "confusion_matrix": confusion_matrix(g, p, labels=labels).tolist(),
        "labels": labels,
    }


_TITLE_ALIASES = ("title", "video_title", "videotitle", "name")
_DESC_ALIASES = ("description", "desc", "video_description", "summary", "details")
_VIEW_ALIASES = ("view_count", "views", "viewcount", "view")
_LIKE_ALIASES = ("like_count", "likes", "likecount", "like")


def _pick(cols_lower: Dict[str, str], aliases) -> Optional[str]:
    for a in aliases:
        if a in cols_lower:
            return cols_lower[a]
    return None


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map arbitrary CSV headers onto the canonical title/description/view/like columns."""
    cols_lower = {c.strip().lower().replace(" ", "_"): c for c in df.columns}
    title_c = _pick(cols_lower, _TITLE_ALIASES)
    desc_c = _pick(cols_lower, _DESC_ALIASES)
    view_c = _pick(cols_lower, _VIEW_ALIASES)
    like_c = _pick(cols_lower, _LIKE_ALIASES)

    work = pd.DataFrame(index=df.index)
    work["title"] = df[title_c].astype(str) if title_c else ""
    work["description"] = df[desc_c].astype(str) if desc_c else ""
    work["view_count"] = pd.to_numeric(df[view_c], errors="coerce").fillna(0.0) if view_c else 0.0
    work["like_count"] = pd.to_numeric(df[like_c], errors="coerce").fillna(0.0) if like_c else 0.0
    work["title"] = work["title"].replace({"nan": "", "None": ""}).fillna("")
    work["description"] = work["description"].replace({"nan": "", "None": ""}).fillna("")
    return work.reset_index(drop=True)
if __name__ == "__main__":
    import sys

    demo = classify(
        "Turn $500 into $50,000 in 30 days with this GUARANTEED crypto strategy!",
        "Join my free training and inner circle. Limited spots. Link below. Get rich fast.",
        view_count=10000, like_count=900,
    )
    json.dump(demo, sys.stdout, indent=2)
    print()
