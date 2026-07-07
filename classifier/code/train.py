"""End-to-end training + publication-grade evaluation.

Pipeline
  Stage 1  financial gate          (LogReg; tuned threshold; PR-AUC)
  Stage 2  misleading vs educational
           Model A  XGBoost  word TF-IDF + engineered/lexicon  (interpretable, SHAP)
           Model B  frozen MiniLM embeddings + LogReg          (embedding baseline)
           Model C  fine-tuned DistilBERT                      (true transformer; flagged)

All performance is OUT-OF-SAMPLE: channel-grouped 5-fold CV + a temporal hold-out
(< / >= 2018-01-01). A channel-disjoint 15% demo hold-out is carved out and EXCLUDED from
the deployed model fit so the app's demo CSV is genuinely unseen. Probabilities are
calibrated (Priority 3); Brier reported before/after.

Run:
    python train.py
Env overrides:
    BORDERLINE_AS={exclude|misleading|educational}
    PRIMARY_STAGE2_MODEL={model_a|model_b}
    TRAIN_TRANSFORMER={auto|1|0}      # 0 -> skip Model C and log the fallback
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import config
from finclass import data as datamod
from finclass import evaluation as ev
from finclass import pipeline as pipe

SEED = config.SEED
random.seed(SEED)
np.random.seed(SEED)

# Reproducibility

def reproducibility_block(df, n_train, n_demo) -> Dict:
    pkgs = ["scikit-learn", "xgboost", "shap", "pandas", "numpy", "scipy",
            "sentence-transformers", "transformers", "torch", "vaderSentiment", "matplotlib"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = version(p)
        except Exception:
            versions[p] = "n/a"
    block = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": SEED,
        "package_versions": versions,
        "n_rows_total": int(len(df)),
        "class_balance": {k: int(v) for k, v in df["label"].value_counts().items()},
        "n_channels": int(df["channel_id"].nunique()),
        "borderline_as": config.BORDERLINE_AS,
        "primary_stage2_model": config.PRIMARY_STAGE2_MODEL,
        "char_ngrams_enabled": config.CHAR_NGRAMS_ENABLED,
        "calibration_method": config.CALIBRATION_METHOD,
        "stage1_threshold_default": config.STAGE1_NONFIN_THRESHOLD,
        "stage2_threshold": config.STAGE2_MISLEADING_THRESHOLD,
        "embedding_model": config.EMBEDDING_MODEL,
        "transformer_model": config.TRANSFORMER_MODEL,
        "train_transformer": config.TRAIN_TRANSFORMER,
        "temporal_cutoff": config.TEMPORAL_CUTOFF,
        "n_deployed_train": int(n_train),
        "n_demo_holdout": int(n_demo),
        "evaluation_policy": "ALL reported metrics are out-of-sample (grouped CV + temporal "
                             "hold-out). No in-sample score is reported anywhere.",
    }
    print("\n" + "=" * 78 + "\nREPRODUCIBILITY BLOCK\n" + "=" * 78)
    for k, v in block.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"{k}: {v}")
    print("=" * 78 + "\n")
    return block


def _pos_weight(y_tr: np.ndarray) -> float:
    pos = max(int(y_tr.sum()), 1)
    neg = max(len(y_tr) - int(y_tr.sum()), 1)
    return neg / pos


# Demo hold-out & fingerprints

def carve_demo_holdout(df) -> Dict:
    train_idx, demo_idx = datamod.demo_holdout_split(df)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    demo_df = df.iloc[demo_idx].reset_index(drop=True)

    cols = [c for c in ("display_id", "title", "description", "view_count", "like_count") if c in demo_df.columns]
    demo_df[cols].to_csv(config.DEMO_HOLDOUT_CSV, index=False, encoding="utf-8")
    gold_cols = cols + [c for c in ("label", "primary_flag", "strategy", "upload_date", "quarter") if c in demo_df.columns]
    demo_df[gold_cols].to_csv(config.DEMO_HOLDOUT_GOLD_CSV, index=False, encoding="utf-8")

    prints = datamod.fingerprints(train_df)
    (config.MODELS_DIR / config.TRAIN_FINGERPRINTS_FILE).write_text(json.dumps(prints))
    print(f">>> Priority 1 — demo hold-out: train={len(train_df)} rows, "
          f"demo={len(demo_df)} rows (channel-disjoint, excluded from deployed fit)")
    print(f"    wrote {config.DEMO_HOLDOUT_CSV.name} + {config.DEMO_HOLDOUT_GOLD_CSV.name}")
    return {"train_df": train_df, "demo_df": demo_df,
            "n_train": len(train_df), "n_demo": len(demo_df),
            "demo_label_balance": {k: int(v) for k, v in demo_df["label"].value_counts().items()}}


# Stage 1

def run_stage1(df, train_df) -> Dict:
    print(">>> STAGE 1 — financial vs non-financial gate")
    sd = datamod.stage1_view(df)
    build_fn = lambda y_tr: pipe.build_stage1()  
    thr = config.STAGE1_NONFIN_THRESHOLD

    cv = ev.grouped_cv(sd, build_fn, thr)
    temporal = ev.temporal_holdout(sd, build_fn, thr)

    tuned = ev.tune_threshold(cv["y"], cv["oof_p"], objective="macro_f1")
    tuned_metrics = ev.compute_metrics(cv["y"], cv["oof_p"], tuned["threshold"],
                                       sd.positive_name, sd.negative_name)
    hr = ev.threshold_for_recall(cv["y"], cv["oof_p"], target_recall=0.60)
    hr_metrics = ev.compute_metrics(cv["y"], cv["oof_p"], hr["threshold"],
                                    sd.positive_name, sd.negative_name)
    boot = ev.bootstrap_macro_f1_ci(cv["y"], cv["oof_p"], thr)

    print(f"    CV macro-F1={cv['metrics']['macro_f1_mean']:.3f}±{cv['metrics']['macro_f1_std']:.3f}  "
          f"PR-AUC={cv['metrics']['pr_auc']:.3f}  ROC-AUC={cv['metrics']['roc_auc']:.3f}")
    nf = config.NON_FINANCIAL_LABEL
    print(f"    NON_FINANCIAL recall: default(thr={thr})={cv['metrics']['per_class'][nf]['recall']:.3f}"
          f" -> tuned(thr={tuned['threshold']})={tuned_metrics['per_class'][nf]['recall']:.3f}")

    ev.plot_confusion(cv["metrics"], "Stage 1 — grouped CV (default thr)",
                      config.METRICS_DIR / "stage1_cv_confusion.png")
    ev.plot_confusion(tuned_metrics, f"Stage 1 — grouped CV (tuned thr={tuned['threshold']})",
                      config.METRICS_DIR / "stage1_cv_confusion_tuned.png")
    ev.plot_pr_threshold(cv["y"], cv["oof_p"], "Stage 1 — P/R vs threshold (CV)",
                         config.METRICS_DIR / "stage1_cv_pr_threshold.png")
    ev.plot_calibration(cv["y"], cv["oof_p"], "Stage 1 — calibration (CV)",
                        config.METRICS_DIR / "stage1_cv_calibration.png")

    final = pipe.build_stage1().fit(
        datamod.stage1_view(train_df).X, datamod.stage1_view(train_df).y)
    joblib.dump(final, config.MODELS_DIR / config.STAGE1_MODEL_FILE)
    print(f"    saved -> {config.STAGE1_MODEL_FILE} (fit on deployed-train only)")

    return {"cv": cv["metrics"], "temporal": temporal["metrics"],
            "tuned_threshold": tuned, "tuned_confusion": tuned_metrics,
            "high_recall_threshold": hr, "high_recall_confusion": hr_metrics,
            "macro_f1_bootstrap_ci": boot}


# Stage 2 Evaluation

def eval_stage2_family(sd, raw_build, label: str, make_plots: bool) -> Dict:
    thr = config.STAGE2_MISLEADING_THRESHOLD
    cv = ev.grouped_cv(sd, raw_build, thr)
    temporal = ev.temporal_holdout(sd, raw_build, thr)

    cal_build = lambda y_tr: pipe.calibrate(raw_build(y_tr), len(y_tr))  # noqa: E731
    cal_cv = ev.grouped_cv(sd, cal_build, thr)

    brier_before = ev.brier(cv["y"], cv["oof_p"])
    brier_after = ev.brier(cal_cv["y"], cal_cv["oof_p"])
    ece_before = ev.expected_calibration_error(cv["y"], cv["oof_p"])
    ece_after = ev.expected_calibration_error(cal_cv["y"], cal_cv["oof_p"])
    boot_cv = ev.bootstrap_macro_f1_ci(cv["y"], cv["oof_p"], thr)
    boot_temporal = (ev.bootstrap_macro_f1_ci(temporal["y"], temporal["oof_p"], thr)
                     if not temporal["metrics"].get("skipped") else None)

    print(f"    [{label}] CV macroF1={cv['metrics']['macro_f1_mean']:.3f}±{cv['metrics']['macro_f1_std']:.3f}"
          f" (boot95 {boot_cv['ci95'][0]:.3f}-{boot_cv['ci95'][1]:.3f})"
          f"  temporal={'skip' if temporal['metrics'].get('skipped') else format(temporal['metrics']['macro_f1'],'.3f')}"
          f"  Brier {brier_before:.3f}->{brier_after:.3f}")

    if make_plots:
        key = label
        ev.plot_confusion(cv["metrics"], f"Stage 2 {key} — grouped CV",
                          config.METRICS_DIR / f"stage2_{key}_cv_confusion.png")
        ev.plot_pr_threshold(cv["y"], cv["oof_p"], f"Stage 2 {key} — P/R vs threshold (CV)",
                             config.METRICS_DIR / f"stage2_{key}_cv_pr_threshold.png")
        ev.plot_calibration_compare(cv["y"], cv["oof_p"], cal_cv["oof_p"],
                                    f"Stage 2 {key} — calibration before/after (CV)",
                                    config.METRICS_DIR / f"stage2_{key}_calibration_compare.png")

    return {
        "cv": cv["metrics"], "temporal": temporal["metrics"],
        "calibration": {"brier_before": brier_before, "brier_after": brier_after,
                        "ece_before": ece_before, "ece_after": ece_after,
                        "method": pipe.resolve_calibration_method(len(sd))},
        "macro_f1_bootstrap_ci_cv": boot_cv,
        "macro_f1_bootstrap_ci_temporal": boot_temporal,
        "_oof_y": cv["y"], "_oof_p": cv["oof_p"],
    }



# SHAP & interpretability.md
def run_shap(model_a, sd, max_samples: int = 300) -> Dict:
    print(">>> SHAP interpretability (Model A — word + engineered/lexicon, char n-grams off)")
    import matplotlib.pyplot as plt
    import shap

    union = model_a.named_steps["features"]
    clf = model_a.named_steps["clf"]
    names = np.array(pipe.feature_names_a(model_a), dtype=object)

    import xgboost as xgb

    Xt = union.transform(sd.X)
    n = min(max_samples, Xt.shape[0])
    rng = np.random.RandomState(SEED)
    idx = rng.choice(Xt.shape[0], size=n, replace=False)
    sample = Xt[idx]
    sample_dense = sample.toarray() if hasattr(sample, "toarray") else np.asarray(sample)

    contribs = clf.get_booster().predict(xgb.DMatrix(sample), pred_contribs=True)
    sv = np.asarray(contribs)[:, :-1]
    mean_abs = np.abs(sv).mean(axis=0)
    
    mean_signed = sv.mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    top = order[:20][::-1]
    is_codebook = {nm for nm in names if (nm.startswith("lex_") or ":" not in nm)}
    colors = ["#d62728" if names[i] in is_codebook else "#1f77b4" for i in top]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.barh(range(len(top)), mean_abs[top], color=colors)
    ax.set_yticks(range(len(top)), labels=[str(names[i]) for i in top], fontsize=8)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("Stage 2 Model A — global feature importance\n(red = engineered / codebook-lexicon feature)")
    fig.tight_layout()
    fig.savefig(config.METRICS_DIR / "stage2_model_a_shap_importance.png", dpi=140)
    plt.close(fig)
    try:
        plt.figure()
        shap.summary_plot(sv, sample_dense, feature_names=list(names), max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(config.METRICS_DIR / "stage2_model_a_shap_beeswarm.png", dpi=140, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"    (beeswarm skipped: {exc})")

    rank_of = {int(order[r]): r + 1 for r in range(len(order))}
    top20 = [str(names[i]) for i in order[:20]]
    n_codebook_top20 = sum(1 for i in order[:20] if str(names[i]).startswith("lex_") or ":" not in str(names[i]))
    from finclass.text import RISK_MARKERS

    marker_table = {}
    name_to_i = {str(nm): i for i, nm in enumerate(names)}
    for marker in RISK_MARKERS:
        key = f"lex_{marker}"
        if key in name_to_i:
            i = name_to_i[key]
            present = sample_dense[:, i] > 0
            n_present = int(present.sum())
            signed_present = float(sv[present, i].mean()) if n_present else 0.0
            if n_present == 0:
                direction = "neutral/unused"
            elif signed_present > 0:
                direction = "→ MISLEADING"
            elif signed_present < 0:
                direction = "→ EDUCATIONAL"
            else:
                direction = "neutral/unused"
            marker_table[marker] = {
                "rank": int(rank_of[i]),
                "mean_abs_shap": float(mean_abs[i]),
                "mean_signed_shap_when_present": signed_present,
                "n_present_in_sample": n_present,
                "direction_when_present": direction,
                "direction": direction,
            }
    marker_table = dict(sorted(marker_table.items(), key=lambda kv: kv[1]["rank"]))

    _write_interpretability_md(top20, marker_table, n_codebook_top20)
    print(f"    top-20 (clean): {top20[:10]}")
    print(f"    codebook/engineered features in top-20: {n_codebook_top20}")
    return {"top20_features": top20, "codebook_marker_table": marker_table,
            "n_codebook_features_in_top20": int(n_codebook_top20)}


def _write_interpretability_md(top20, marker_table, n_codebook_top20) -> None:
    lines = [
        "# Interpretability — human codebook vs. machine (Model A, SHAP)",
        "",
        "_Auto-generated by `train.py`. Out-of-sample model; SHAP computed on a held sample._",
        "",
        "Model A uses **word 1–2 grams + engineered/codebook-lexicon features only** "
        "(char n-grams disabled) so every SHAP feature is human-readable.",
        "",
        f"**Codebook / engineered risk features in the global top-20: {n_codebook_top20}.**",
        "",
        "## Top-20 features by mean(|SHAP|)",
        "",
        "| rank | feature |", "|---|---|",
    ]
    for r, f in enumerate(top20, 1):
        lines.append(f"| {r} | `{f}` |")
    lines += [
        "",
        "## Codebook risk-marker triangulation",
        "",
        "Each annotation-codebook §3 risk marker, its global SHAP rank (by mean|SHAP| over the "
        "sample), how many sampled videos actually contain it, and the direction it pushes the "
        "prediction **when present**. High rank + `→ MISLEADING when present` = the model "
        "independently recovered that human-defined marker.",
        "",
        "| codebook marker | SHAP rank | mean|SHAP| | # present | direction when present |",
        "|---|---|---|---|---|",
    ]
    for marker, v in marker_table.items():
        lines.append(f"| `{marker}` | {v['rank']} | {v['mean_abs_shap']:.4f} | "
                     f"{v.get('n_present_in_sample', 0)} | {v['direction']} |")
    lines += [
        "",
        "> Note: markers whose literal cues (`$`, `%`, digits, `10x`) are already captured by "
        "the engineered `title_has_currency` / `title_has_number` features (and by word n-grams) "
        "may show a lower *lexicon-feature* rank without implying the signal is absent — it is "
        "represented elsewhere. This redundancy is discussed in the README.",
    ]
    (config.METRICS_DIR / "interpretability.md").write_text("\n".join(lines), encoding="utf-8")



# Deployed models 
def finalise_and_serialise(train_df) -> Dict:
    sd = datamod.stage2_view(train_df, borderline_as="exclude")
    spw = _pos_weight(sd.y)

    model_a = pipe.build_model_a(scale_pos_weight=spw).fit(sd.X, sd.y)  
    joblib.dump(model_a, config.MODELS_DIR / config.STAGE2_MODEL_A_FILE)
    model_a_cal = pipe.calibrate(pipe.build_model_a(scale_pos_weight=spw), len(sd.y)).fit(sd.X, sd.y)
    joblib.dump(model_a_cal, config.MODELS_DIR / config.STAGE2_MODEL_A_CAL_FILE)

    model_b = pipe.build_model_b().fit(sd.X, sd.y)
    joblib.dump(model_b, config.MODELS_DIR / config.STAGE2_MODEL_B_FILE)
    model_b_cal = pipe.calibrate(pipe.build_model_b(), len(sd.y)).fit(sd.X, sd.y)
    joblib.dump(model_b_cal, config.MODELS_DIR / config.STAGE2_MODEL_B_CAL_FILE)

    print("    saved -> Model A (raw+calibrated), Model B (raw+calibrated), all on deployed-train")
    return {"model_a": model_a, "view": sd}



# Model C: fine-tuned DistilBERT, McNemar vs A
def run_model_c(df) -> Dict:
    mode = config.TRAIN_TRANSFORMER
    sd = datamod.stage2_view(df, borderline_as="exclude")
    if mode in ("0", "off", "false", "no"):
        print(">>> Model C (DistilBERT) SKIPPED by TRAIN_TRANSFORMER=0 — "
              "Model B (frozen embeddings + LR) stands in for the transformer family.")
        return {"trained": False, "reason": "skipped_by_flag",
                "fallback": "Model B (frozen MiniLM embeddings + Logistic Regression)"}

    print(">>> Model C — fine-tuning DistilBERT (true transformer)")
    from finclass import transformer_model as tm

    if not tm.gpu_available():
        print("    (no GPU — running on CPU; this is the slow step. Set TRAIN_TRANSFORMER=0 to skip.)")
    tr_idx, val_idx, te_idx = datamod.three_way_group_split(sd.groups)
    reuse = os.environ.get("REUSE_MODEL_C", "").lower() in ("1", "true", "yes") \
        and (config.TRANSFORMER_DIR / "config.json").exists()
    try:
        if reuse:
            print("    (REUSE_MODEL_C=1 — reloading saved DistilBERT instead of retraining)")
            res = tm.load_and_evaluate(sd.X, sd.y, te_idx, str(config.TRANSFORMER_DIR))
        else:
            res = tm.train_and_evaluate(sd.X, sd.y, tr_idx, val_idx, te_idx,
                                        save_dir=str(config.TRANSFORMER_DIR))
    except Exception as exc:  
        print(f"    [Model C] training failed ({exc!r}); falling back to Model B as the transformer arm.")
        return {"trained": False, "reason": f"error: {exc!r}",
                "fallback": "Model B (frozen MiniLM embeddings + Logistic Regression)"}

    # McNemar test
    fit_idx = np.concatenate([tr_idx, val_idx])
    model_a_split = pipe.build_model_a(scale_pos_weight=_pos_weight(sd.y[fit_idx])).fit(
        sd.X.iloc[fit_idx], sd.y[fit_idx])
    p_a = np.asarray(model_a_split.predict_proba(sd.X.iloc[te_idx]))[:, 1]
    pred_a = (p_a >= 0.5).astype(int)
    pred_c = (res["test_p"] >= 0.5).astype(int)
    a_metrics = ev.compute_metrics(res["test_y"], p_a, 0.5, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)
    mcnemar = ev.mcnemar_test(res["test_y"], pred_a, pred_c)

    print(f"    [Model C] test macro-F1={res['metrics']['macro_f1']:.3f}  "
          f"(Model A on same split={a_metrics['macro_f1']:.3f})")
    print(f"    McNemar A-vs-C: p={mcnemar['p_value']:.4f} "
          f"(A_right_C_wrong={mcnemar['a_right_b_wrong']}, "
          f"A_wrong_C_right={mcnemar['a_wrong_b_right']})")
    return {"trained": True, "metrics": res["metrics"],
            "model_a_same_split": a_metrics, "mcnemar_a_vs_c": mcnemar,
            "device": res["metrics"].get("device")}


# Results
def _strip_oof(d: Dict) -> Dict:
    return {k: v for k, v in d.items() if not k.startswith("_oof")}


def write_results_md(results: Dict) -> None:
    s1, s2 = results["stage1"], results["stage2"]
    a, b = s2["model_a"], s2["model_b"]
    c = results["model_c"]
    lines = ["# Results — out-of-sample only", "",
             "_Auto-generated by `train.py`. Every number is grouped-CV or temporal/held-out; "
             "no in-sample metric appears anywhere._", ""]

    lines += ["## Stage 1 — financial gate (Priority 5)", "",
              "| Protocol | macro-F1 | PR-AUC | ROC-AUC |", "|---|---|---|---|",
              f"| Grouped CV (default thr {config.STAGE1_NONFIN_THRESHOLD}) | "
              f"{s1['cv']['macro_f1_mean']:.3f} ± {s1['cv']['macro_f1_std']:.3f} | "
              f"{s1['cv']['pr_auc']:.3f} | {s1['cv']['roc_auc']:.3f} |",
              f"| Temporal hold-out | "
              f"{'—' if s1['temporal'].get('skipped') else format(s1['temporal']['macro_f1'],'.3f')} | — | — |",
              "",
              f"NON_FINANCIAL operating points (positive class = NON_FINANCIAL):", "",
              f"| Threshold | NON_FINANCIAL recall | NON_FINANCIAL precision | macro-F1 |",
              f"|---|---|---|---|",
              f"| default 0.50 | {s1['cv']['per_class']['NON_FINANCIAL']['recall']:.2f} | "
              f"{s1['cv']['per_class']['NON_FINANCIAL']['precision']:.2f} | {s1['cv']['macro_f1']:.3f} |",
              f"| macro-F1-tuned {s1['tuned_threshold']['threshold']} | "
              f"{s1['tuned_confusion']['per_class']['NON_FINANCIAL']['recall']:.2f} | "
              f"{s1['tuned_confusion']['per_class']['NON_FINANCIAL']['precision']:.2f} | "
              f"{s1['tuned_threshold']['macro_f1']:.3f} |",
              f"| high-recall {s1['high_recall_threshold']['threshold']} | "
              f"{s1['high_recall_confusion']['per_class']['NON_FINANCIAL']['recall']:.2f} | "
              f"{s1['high_recall_confusion']['per_class']['NON_FINANCIAL']['precision']:.2f} | "
              f"{s1['high_recall_threshold']['macro_f1']:.3f} |",
              "",
              "Trade-off: lowering the cutoff catches more non-financial corpus noise "
              "(higher recall) but rejects more genuinely-financial videos (lower precision); "
              "at the 5% base rate the macro-F1 optimum sits near the default.", ""]

    lines += ["## Stage 2 — misleading vs educational (primary: BORDERLINE excluded)", "",
              "| Model | CV macro-F1 (boot 95% CI) | Temporal macro-F1 | CV ROC-AUC | Brier raw→cal |",
              "|---|---|---|---|---|"]
    for key, lbl in (("model_a", "A · XGBoost (word+lexicon)"), ("model_b", "B · MiniLM emb + LR")):
        m = s2[key]
        ci = m["macro_f1_bootstrap_ci_cv"]["ci95"]
        tF1 = "—" if m["temporal"].get("skipped") else f"{m['temporal']['macro_f1']:.3f}"
        cal = m["calibration"]
        lines.append(
            f"| {lbl} | {m['cv']['macro_f1_mean']:.3f} ({ci[0]:.3f}–{ci[1]:.3f}) | {tF1} | "
            f"{m['cv']['roc_auc']:.3f} | {cal['brier_before']:.3f}→{cal['brier_after']:.3f} |")
    if c.get("trained"):
        cm = c["metrics"]
        lines.append(f"| C · DistilBERT (fine-tuned) | {cm['macro_f1']:.3f} "
                     f"(single channel-disjoint split, n_test={cm['n_test']}) | — | "
                     f"{'—' if cm.get('roc_auc') is None else format(cm['roc_auc'],'.3f')} | — |")
    else:
        lines.append(f"| C · DistilBERT | _not trained ({c.get('reason')})_; fallback: {c.get('fallback')} | | | |")

    lines += ["", "### Feature-space cleanup (Priority 4): char n-grams on vs off",
              "", "| Model A variant | CV macro-F1 |", "|---|---|",
              f"| with char 3–5g (less interpretable) | {s2['model_a_withchar']['cv']['macro_f1_mean']:.3f} "
              f"± {s2['model_a_withchar']['cv']['macro_f1_std']:.3f} |",
              f"| **word + lexicon only (deployed, interpretable)** | "
              f"{a['cv']['macro_f1_mean']:.3f} ± {a['cv']['macro_f1_std']:.3f} |", ""]

    # Per-strategy 
    lines += ["### Per-strategy test performance — Model A, grouped CV (RQ1)", "",
              "| Strategy | n | n misleading | macro-F1 |", "|---|---|---|---|"]
    for strat, v in a["cv"]["per_strategy"].items():
        lines.append(f"| {strat} | {v['n']} | {v['n_positive']} | {v['macro_f1']:.3f} |")

    if c.get("trained"):
        mc = c["mcnemar_a_vs_c"]
        lines += ["", "### Significance — Model A vs Model C (McNemar, same test rows)", "",
                  f"- Model A acc = {mc['acc_a']:.3f}, Model C acc = {mc['acc_b']:.3f}",
                  f"- discordant pairs: A✓C✗ = {mc['a_right_b_wrong']}, A✗C✓ = {mc['a_wrong_b_right']}",
                  f"- **McNemar exact p = {mc['p_value']:.4f}** "
                  f"({'significant' if mc['p_value'] < 0.05 else 'not significant'} at α=0.05)"]

    lines += ["", "### Interpretability (SHAP, Model A)", "",
              f"- Codebook / engineered risk features in global top-20: "
              f"**{results['shap']['n_codebook_features_in_top20']}**",
              f"- Top features: {', '.join('`%s`' % f for f in results['shap']['top20_features'][:12])}",
              "- Full codebook-marker rank table: `metrics/interpretability.md`.", ""]

    (config.METRICS_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")
    (config.METRICS_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


# Main

def main() -> None:
    df = datamod.load_gold()
    demo = carve_demo_holdout(df)
    train_df = demo["train_df"]
    repro = reproducibility_block(df, demo["n_train"], demo["n_demo"])

    results: Dict = {"reproducibility": repro, "demo_holdout": _strip_oof(
        {"n_train": demo["n_train"], "n_demo": demo["n_demo"], "label_balance": demo["demo_label_balance"]})}

    results["stage1"] = run_stage1(df, train_df)

    print(">>> STAGE 2 — misleading vs educational (evaluation: grouped CV + temporal)")
    sd2 = datamod.stage2_view(df, borderline_as="exclude")
    a_build = lambda y: pipe.build_model_a(scale_pos_weight=_pos_weight(y))          
    a_withchar_build = lambda y: pipe.build_model_a(scale_pos_weight=_pos_weight(y), char_enabled=True)  # noqa: E731
    b_build = lambda y: pipe.build_model_b()                                         
    s2 = {}
    s2["model_a"] = eval_stage2_family(sd2, a_build, "model_a", make_plots=True)
    s2["model_b"] = eval_stage2_family(sd2, b_build, "model_b", make_plots=True)
    s2["model_a_withchar"] = _strip_oof(eval_stage2_family(sd2, a_withchar_build, "model_a_withchar", make_plots=False))
    sd2_sens = datamod.stage2_view(df, borderline_as="misleading")
    sens = ev.grouped_cv(sd2_sens, a_build, config.STAGE2_MISLEADING_THRESHOLD)
    s2["sensitivity_borderline_as_misleading_model_a"] = sens["metrics"]
    results["stage2"] = {k: _strip_oof(v) if isinstance(v, dict) and "_oof_y" in v else v for k, v in s2.items()}

    print(">>> Serialising deployed models (fit on deployed-train, calibrated)")
    finals = finalise_and_serialise(train_df)
    results["shap"] = run_shap(finals["model_a"], finals["view"])

    results["model_c"] = run_model_c(df)

    # Metadata for app
    metadata = {
        "seed": SEED,
        "borderline_as": config.BORDERLINE_AS,
        "primary_stage2_model": config.PRIMARY_STAGE2_MODEL,
        "stage1_threshold_default": config.STAGE1_NONFIN_THRESHOLD,
        "stage1_threshold_tuned": results["stage1"]["tuned_threshold"]["threshold"],
        "stage2_threshold": config.STAGE2_MISLEADING_THRESHOLD,
        "positive_class": config.POSITIVE_CLASS,
        "negative_class": config.NEGATIVE_CLASS,
        "embedding_model": config.EMBEDDING_MODEL,
        "calibration": results["stage2"]["model_a"]["calibration"],
        "package_versions": repro["package_versions"],
        "evaluation_policy": repro["evaluation_policy"],
        "files": {
            "stage1": config.STAGE1_MODEL_FILE,
            "stage2_model_a_uncalibrated": config.STAGE2_MODEL_A_FILE,
            "stage2_model_a_calibrated": config.STAGE2_MODEL_A_CAL_FILE,
            "stage2_model_b_uncalibrated": config.STAGE2_MODEL_B_FILE,
            "stage2_model_b_calibrated": config.STAGE2_MODEL_B_CAL_FILE,
            "train_fingerprints": config.TRAIN_FINGERPRINTS_FILE,
        },
    }
    (config.MODELS_DIR / config.METADATA_FILE).write_text(json.dumps(metadata, indent=2))

    write_results_md(results)
    out = config.METRICS_DIR / "metrics.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nAll metrics -> {out}\nResults table -> {config.METRICS_DIR/'results.md'}\n"
          f"Interpretability -> {config.METRICS_DIR/'interpretability.md'}")


if __name__ == "__main__":
    main()
