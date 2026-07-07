"""Leakage-safe evaluation: channel-grouped CV, temporal hold-out, metrics, and plots.

All estimators are produced by a ``build_fn(y_train) -> estimator`` factory so that any
fold can set imbalance handling (e.g. XGBoost ``scale_pos_weight``) from its own training
labels - never from data it is about to be tested on.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from finclass.data import StageData, grouped_folds, temporal_split

BuildFn = Callable[[np.ndarray], object]



# Metric computation

def compute_metrics(
    y_true: np.ndarray, p_pos: np.ndarray, threshold: float, pos_name: str, neg_name: str
) -> Dict:
    """Full metric bundle for a binary task scored by ``P(positive)``."""
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos, dtype=float)
    y_pred = (p_pos >= threshold).astype(int)

    out: Dict = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "threshold": float(threshold),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "per_class": {},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "confusion_labels": [neg_name, pos_name],
    }
    for cls_idx, name in ((1, pos_name), (0, neg_name)):
        yt = (y_true == cls_idx).astype(int)
        yp = (y_pred == cls_idx).astype(int)
        out["per_class"][name] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support": int(yt.sum()),
        }
    
    if 0 < y_true.sum() < len(y_true):
        out["roc_auc"] = float(roc_auc_score(y_true, p_pos))
        out["pr_auc"] = float(average_precision_score(y_true, p_pos))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def per_strategy_metrics(
    y_true: np.ndarray, p_pos: np.ndarray, strategy: pd.Series, threshold: float
) -> Dict[str, Dict]:
    """Macro-F1 (and support) broken down by ``strategy`` stratum."""
    strategy = pd.Series(strategy).reset_index(drop=True)
    res: Dict[str, Dict] = {}
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(p_pos) >= threshold).astype(int)
    for strat in sorted(strategy.dropna().unique()):
        mask = (strategy == strat).to_numpy()
        if mask.sum() == 0:
            continue
        res[str(strat)] = {
            "n": int(mask.sum()),
            "n_positive": int(y_true[mask].sum()),
            "macro_f1": float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)),
        }
    return res

# Protocols
def grouped_cv(data: StageData, build_fn: BuildFn, threshold: float, n_splits: int | None = None) -> Dict:
    """Channel-grouped CV producing out-of-fold predictions + per-fold macro-F1 with CI."""
    oof_p = np.full(len(data), np.nan)
    fold_macro_f1: List[float] = []
    fold_assignment = np.full(len(data), -1)

    for fold, (tr, te) in enumerate(grouped_folds(data, n_splits)):
        est = build_fn(data.y[tr])
        est.fit(data.X.iloc[tr], data.y[tr])
        p = _proba(est, data.X.iloc[te])
        oof_p[te] = p
        fold_assignment[te] = fold
        fold_macro_f1.append(
            f1_score(data.y[te], (p >= threshold).astype(int), average="macro", zero_division=0)
        )

    metrics = compute_metrics(data.y, oof_p, threshold, data.positive_name, data.negative_name)
    metrics["protocol"] = "channel_grouped_cv"
    metrics["n_folds"] = len(fold_macro_f1)
    metrics["fold_macro_f1"] = [float(x) for x in fold_macro_f1]
    metrics["macro_f1_mean"] = float(np.mean(fold_macro_f1))
    metrics["macro_f1_std"] = float(np.std(fold_macro_f1))
    half = 1.96 * np.std(fold_macro_f1, ddof=1) / np.sqrt(len(fold_macro_f1)) if len(fold_macro_f1) > 1 else 0.0
    metrics["macro_f1_ci95"] = [float(np.mean(fold_macro_f1) - half), float(np.mean(fold_macro_f1) + half)]
    metrics["per_strategy"] = per_strategy_metrics(data.y, oof_p, data.strategy, threshold)
    return {"metrics": metrics, "oof_p": oof_p, "y": data.y}


def temporal_holdout(data: StageData, build_fn: BuildFn, threshold: float) -> Dict:
    """Train before the temporal cutoff, test on/after it (RQ1 temporal generalisation)."""
    tr, te = temporal_split(data)
    if len(te) == 0 or len(tr) == 0:
        return {"metrics": {"protocol": "temporal_holdout", "skipped": True,
                            "reason": "empty train or test side"}, "oof_p": None, "y": None}
    est = build_fn(data.y[tr])
    est.fit(data.X.iloc[tr], data.y[tr])
    p = _proba(est, data.X.iloc[te])

    metrics = compute_metrics(data.y[te], p, threshold, data.positive_name, data.negative_name)
    metrics["protocol"] = "temporal_holdout"
    metrics["cutoff"] = config.TEMPORAL_CUTOFF
    metrics["n_train"] = int(len(tr))
    metrics["n_test"] = int(len(te))
    metrics["per_strategy"] = per_strategy_metrics(data.y[te], p, data.strategy.iloc[te], threshold)
    return {"metrics": metrics, "oof_p": p, "y": data.y[te], "test_idx": te}


def _proba(est, X) -> np.ndarray:
    """P(positive class) regardless of estimator API quirks."""
    proba = est.predict_proba(X)
    return np.asarray(proba)[:, 1]



# Calibration metrics 
def brier(y_true, p_pos) -> float:
    """Brier score (mean squared error of probabilistic forecasts); lower is better."""
    return float(brier_score_loss(np.asarray(y_true).astype(int), np.asarray(p_pos, float)))


def expected_calibration_error(y_true, p_pos, n_bins: int = 10) -> float:
    """ECE: average gap between confidence and accuracy over equal-width probability bins."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p_pos, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += (m.mean()) * abs(p[m].mean() - y_true[m].mean())
    return float(ece)


def bootstrap_macro_f1_ci(
    y_true, p_pos, threshold: float, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05
) -> Dict:
    """Percentile bootstrap 95% CI for macro-F1 by resampling test predictions with replacement."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(p_pos, float) >= threshold).astype(int)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.randint(0, n, n)
        stats[i] = f1_score(y_true[s], y_pred[s], average="macro", zero_division=0)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "ci95": [float(lo), float(hi)],
        "n_boot": int(n_boot),
    }


def mcnemar_test(y_true, pred_a, pred_b) -> Dict:
    """McNemar test comparing two classifiers' correctness on the SAME test items."""
    y_true = np.asarray(y_true).astype(int)
    a_correct = np.asarray(pred_a).astype(int) == y_true
    b_correct = np.asarray(pred_b).astype(int) == y_true
    b01 = int(np.sum(a_correct & ~b_correct))  
    b10 = int(np.sum(~a_correct & b_correct))   
    n = b01 + b10
    from scipy.stats import binomtest

    p = binomtest(min(b01, b10), n, 0.5, alternative="two-sided").pvalue if n > 0 else 1.0
    return {
        "a_right_b_wrong": b01,
        "a_wrong_b_right": b10,
        "n_discordant": n,
        "p_value": float(p),
        "acc_a": float(a_correct.mean()),
        "acc_b": float(b_correct.mean()),
    }


def threshold_for_recall(y_true, p_pos, target_recall: float = 0.60) -> Dict:
    """Highest threshold whose positive-class recall >= target (max precision at that recall).

    For the Stage-1 gate (positive = NON_FINANCIAL), this exposes a high-recall operating
    point and its precision cost, making the precision/recall trade-off concrete.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p_pos, float)
    best = None
    for thr in np.round(np.arange(0.05, 0.96, 0.01), 2):
        yp = (p >= thr).astype(int)
        rec = recall_score(y_true, yp, zero_division=0)
        if rec >= target_recall:
            prec = precision_score(y_true, yp, zero_division=0)
            cand = {"threshold": float(thr), "recall": float(rec), "precision": float(prec),
                    "macro_f1": float(f1_score(y_true, yp, average="macro", zero_division=0))}
            if best is None or thr > best["threshold"]:
                best = cand
    if best is None:  
        yp = (p >= 0.05).astype(int)
        best = {"threshold": 0.05, "recall": float(recall_score(y_true, yp, zero_division=0)),
                "precision": float(precision_score(y_true, yp, zero_division=0)),
                "macro_f1": float(f1_score(y_true, yp, average="macro", zero_division=0)),
                "target_unreachable": True}
    best["target_recall"] = float(target_recall)
    return best


def tune_threshold(y_true, p_pos, objective: str = "macro_f1", grid=None) -> Dict:
    """Pick the probability threshold that maximises an objective on out-of-fold scores.

    objective: "macro_f1" (default) or "pos_recall_at_precision" style not needed here.
    Returns the best threshold and the resulting metrics, for honest threshold reporting.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p_pos, float)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    best = {"threshold": 0.5, "macro_f1": -1.0}
    for thr in grid:
        yp = (p >= thr).astype(int)
        mf1 = f1_score(y_true, yp, average="macro", zero_division=0)
        if mf1 > best["macro_f1"]:
            best = {
                "threshold": float(thr),
                "macro_f1": float(mf1),
                "pos_recall": float(recall_score(y_true, yp, zero_division=0)),
                "pos_precision": float(precision_score(y_true, yp, zero_division=0)),
            }
    return best



# Plots

def plot_confusion(metrics: Dict, title: str, path) -> None:
    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["confusion_labels"]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=labels)
    ax.set_yticks([0, 1], labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_calibration(y_true, p_pos, title: str, path) -> None:
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos, dtype=float)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    n_bins = min(10, max(2, len(np.unique(np.round(p_pos, 2))) // 2))
    try:
        frac_pos, mean_pred = calibration_curve(y_true, p_pos, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", label="model")
    except ValueError:
        pass
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="perfect")
    ax.set_xlabel("Mean predicted P(positive)")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_calibration_compare(y_true, p_before, p_after, title: str, path) -> None:
    """Overlay uncalibrated vs calibrated reliability curves (Priority 3)."""
    y_true = np.asarray(y_true).astype(int)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="perfect")
    for p, lab, style in ((p_before, "before (raw)", "o-"), (p_after, "after (calibrated)", "s-")):
        p = np.asarray(p, float)
        try:
            frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, style, label=lab)
        except ValueError:
            pass
    ax.set_xlabel("Mean predicted P(positive)")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_pr_threshold(y_true, p_pos, title: str, path) -> None:
    """Precision/recall vs threshold — the trade-off curve required by the brief."""
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos, dtype=float)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return
    prec, rec, thr = precision_recall_curve(y_true, p_pos)
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ax.plot(thr, prec[:-1], label="precision")
    ax.plot(thr, rec[:-1], label="recall")
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    ax.plot(thr, f1[:-1], label="F1", linestyle=":")
    ax.axvline(config.STAGE2_MISLEADING_THRESHOLD, color="grey", alpha=0.5, linestyle="--")
    ax.set_xlabel("Threshold on P(positive)")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower center", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
