# Misleading-vs-Educational YouTube Finance Content Classifier

A reproducible, **text-only** machine-learning system that scores YouTube "hustlesphere"
(money-making) videos as **EDUCATIONAL** vs **MISLEADING** from their **title and
description alone** — videos are never watched. A first stage gates out **NON_FINANCIAL**
corpus false-positives so that only genuine financial content reaches the
misleading/educational judgement.

Built for an MSc dissertation (University of Edinburgh, Informatics). Labels and the
risk-marker construct come from `../labelling/annotation_codebook.md`, grounded in Kakhbod
et al. (2023), Jukwey (2024) and IOSCO (2024).

> **Evaluation policy:** every performance number reported anywhere is **out-of-sample** —
> channel-grouped 5-fold CV and/or a temporal hold-out. No in-sample score is ever reported.
> A channel-disjoint 15% demo hold-out (`data/demo_holdout.csv`) is carved out and **excluded
> from the deployed model fit**, and the app warns if you score a file that overlaps training.

---

## 1. Two-stage pipeline with three models

```
                 ┌──────────────────────────────┐
 title + desc ──▶│ STAGE 1  financial gate       │── NON_FINANCIAL ──▶ stop
   (+ views)     │ word+char TF-IDF → LogReg     │   (tuned threshold; PR-AUC reported)
                 └───────────────┬──────────────┘
                                 │ FINANCIAL
                                 ▼
                 ┌──────────────────────────────┐
                 │ STAGE 2  calibrated P(misleading)             │
                 │ A · XGBoost  word TF-IDF + codebook lexicon   │──▶ EDUCATIONAL / MISLEADING
                 │ B · MiniLM embeddings + LogReg                │     + top contributing signals
                 │ C · fine-tuned DistilBERT (true transformer)  │
                 └──────────────────────────────┘
```

- **Stage 1** — `NON_FINANCIAL` vs `FINANCIAL` (= EDUCATIONAL ∪ MISLEADING ∪ BORDERLINE).
- **Stage 2** — `Misleading` vs `Educational` on financial videos only. `BORDERLINE` is
  **excluded** by default (primary analysis) with a sensitivity check folding it into
  MISLEADING. Output is a **calibrated** `P(misleading) ∈ [0,1]` plus a thresholded label.

### Three model families
- **Model A — XGBoost** over **word 1–2 gram TF-IDF + engineered/codebook-lexicon features**.
  Char n-grams are **off** so SHAP is human-readable; this is the deployed, interpretable
  model and the scientific deliverable.
- **Model B — frozen MiniLM sentence-embeddings + Logistic Regression** (embedding baseline;
  embeddings disk-cached).
- **Model C — genuinely fine-tuned `distilbert-base-uncased`** on `title [SEP] description`
  (max 256 tokens, class-weighted loss, early-stopping on val macro-F1). GPU-aware: trains on
  GPU if present, else CPU with a warning; `TRAIN_TRANSFORMER=0` skips it and logs the
  fallback. Evaluated on a single channel-disjoint train/val/test split (full grouped CV is
  too expensive for a fine-tuned LM — documented).

### Leakage safety (enforced in code)
Channel-grouped `GroupKFold` (no channel in both train and test) · temporal hold-out
(train `< 2018-01-01`, test `≥`) · channel-disjoint demo hold-out excluded from deployment ·
gold `label` never enters any feature matrix.

---

## 2. Results (all out-of-sample)

Full numbers, per-class/per-strategy breakdowns, calibration and confusion plots are in
`metrics/results.md`, `metrics/metrics.json`, `metrics/interpretability.md`, and the PNGs.

### Stage 1 — financial gate
| Protocol | macro-F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Channel-grouped CV | **0.756 ± 0.029** | 0.556 | 0.910 |
| Temporal hold-out (≥2018) | 0.734 | — | — |

PR-AUC (0.556) is reported because it is more honest than ROC-AUC under the 5% NON_FINANCIAL
base rate. Operating points make the precision/recall trade-off explicit: default thr 0.50 →
recall 0.43 / precision 0.72; a high-recall point (thr 0.31) → recall 0.61 / precision 0.41.
The macro-F1 optimum sits at the default.

### Stage 2 — misleading vs educational (primary: BORDERLINE excluded)
| Model | CV macro-F1 (bootstrap 95% CI) | Temporal | CV ROC-AUC | Brier raw→calibrated |
|---|---|---|---|---|
| **A · XGBoost (word + lexicon)** | **0.811 (0.784–0.838)** | 0.815 | 0.903 | 0.127 → 0.123 |
| B · MiniLM emb + LR | 0.671 (0.642–0.704) | 0.654 | 0.745 | 0.280 → **0.187** |
| C · DistilBERT (fine-tuned) | 0.722 (single split, n_test=174) | — | 0.805 | — |

- **Calibration (Priority 3):** Stage-2 probabilities are calibrated (Platt/isotonic, fit by
  internal CV within the training fold only). Model B's Brier improves markedly (0.280→0.187);
  Model A was already well-calibrated. The app gauge and the batch `p_misleading` use the
  **calibrated** probabilities. See `stage2_model_*_calibration_compare.png`.
- **Feature cleanup (Priority 4):** dropping char n-grams not only made SHAP readable, it
  **slightly improved** Model A (0.797 → 0.811 macro-F1) — interpretability at no accuracy cost.
- **Per-strategy (RQ1):** E-commerce 0.835 · Online-Earning 0.803 · Trading 0.792 ·
  **Entrepreneurship & Real Estate 0.698** (hardest — fewest misleading examples, n_pos=42).

### Model comparison — a reportable finding
On this small (~880-example), lexicon-rich corpus the **interpretable XGBoost (A) beats the
fine-tuned DistilBERT (C)**: 0.783 vs 0.722 macro-F1 on the identical channel-disjoint test
split. A **McNemar test** on the same rows gives **p = 0.065** — A is numerically better but
the difference is *not* significant at α=0.05. This is consistent with the literature: feature-
and lexicon-based models can match or beat fine-tuned LMs in small-data, lexically-distinctive
regimes, where 66M transformer parameters cannot be estimated well from ~700 training rows. We
report it honestly rather than hiding the "wrong-way" result, and recommend **A for deployment
and explanation**, with C expected to overtake given substantially more labelled data.

### Interpretability — does the model rediscover the codebook?
SHAP (exact TreeSHAP on the **sparse** feature matrix — XGBoost is sparsity-aware, so a dense
0 ≠ a missing 0) top-20 for Model A is fully human-readable: `title_has_currency`, `w:course`,
`lex_RISK_OMISSION`, `lex_UNDISCLOSED_PROMO`, `title_has_number`, `w:get`, … **9 codebook/engineered
risk features rank in the global top-20.** Judged *when the marker is present*, `RISK_OMISSION`
(rank 3), `UNDISCLOSED_PROMO` (rank 4) and `GET_RICH_QUICK` (rank 60) all push **→ MISLEADING** —
the model independently recovers the human codebook. `metrics/interpretability.md` tabulates every
§3 marker with its rank, present-count, and present-direction. Markers whose literal cues
(`$`, `%`, digits) are absorbed by the engineered currency/number features (or word n-grams) show
a lower *lexicon-feature* rank without the signal being absent — discussed there.

---

## 3. Install & run

Python **3.11+** (developed/validated on 3.13, CPU-only).
```bash
cd code
python -m venv .venv && . .venv/Scripts/activate    
pip install -r requirements.txt
```

### Train (one command: raw CSV → models + metrics + SHAP + Model C)
```bash
python train.py
```
Runs both stages and all three models under channel-grouped CV + temporal hold-out, calibrates
probabilities, carves the demo hold-out, runs SHAP, fine-tunes DistilBERT, and writes
`metrics/` + `models/`. First run downloads MiniLM (~80 MB) and DistilBERT (~250 MB).
On CPU the DistilBERT fine-tune is the slow step (several minutes/epoch).

Optional overrides:
```bash
TRAIN_TRANSFORMER=0 python train.py      
REUSE_MODEL_C=1     python train.py      
BORDERLINE_AS=misleading python train.py 
```

### Run the web app (loads serialised models; never retrains)
```bash
streamlit run app.py
```
- **Mode 1 — Single video:** Stage-1 eligibility, a prominent **calibrated P(misleading) gauge**,
  the label, and the **top contributing signals** (per-instance SHAP). Thresholds adjustable live.
- **Mode 2 — CSV batch:** input **plus three appended columns** (`predicted_eligibility`,
  `p_misleading`, `predicted_label`), order/columns preserved, with summary, distribution chart,
  and a download button. **Overlap banner** warns if the file overlaps training (Priority 1).
  If the CSV carries a `label` column it also shows a **confusion matrix + macro-F1 vs your
  labels** (Priority 7). Try `data/demo_holdout.csv` (held-out) or `data/demo_holdout_gold.csv`.

### Programmatic inference
```python
import predict
predict.classify("Turn $500 into $50k — GUARANTEED!", "Join my inner circle, link below")


import pandas as pd
out = predict.classify_batch(pd.read_csv("videos.csv"))   
predict.training_overlap(df)                               
```

---

## 4. Project layout
```
code/
├── config.py            # single source of truth (paths, seeds, thresholds, calibration, transformer)
├── train.py             # trains everything; metrics, SHAP, calibration, Model C, results.md
├── predict.py           # classify() + classify_batch() + overlap/gold helpers
├── app.py               # Streamlit two-mode app (+ overlap banner, gold comparison)
├── requirements.txt     # pinned
├── .streamlit/config.toml   # disables the source watcher (silences transformers/torchvision noise)
├── finclass/
│   ├── text.py          # cleaning, codebook risk lexicons, engineered features, VADER
│   ├── data.py          # gold loading, stage views, channel-grouped / temporal / demo splits, fingerprints
│   ├── pipeline.py      # feature union + Model A/B builders + calibration
│   ├── transformer_model.py  # Model C — fine-tuned DistilBERT (train + reload/evaluate)
│   └── evaluation.py    # CV, temporal, metrics, Brier/ECE, bootstrap CI, McNemar, threshold tuning, plots
├── data/                # demo_holdout.csv + demo_holdout_gold.csv (created by train.py)
├── models/              # serialised artifacts incl. calibrated models + DistilBERT (created by train.py)
└── metrics/             # results.md, interpretability.md, metrics.json, PNG plots (created by train.py)
```

---

## 5. Design decisions & limitations
- **Deployed models are fit on the 85% train portion** (demo hold-out excluded) so the app's
  demo CSV is genuinely unseen; CV/temporal estimates use the full data (cross-validated, no
  deployed-model leakage). For a production rebuild on 100% of data, retrain without the hold-out.
- **XGBoost is a first-class result, not a baseline** — its SHAP output is the dissertation's
  evidence that the learned model recovers the hand-coded risk taxonomy; the app headlines it so
  every score is explainable. The fine-tuned transformer is included for a fair comparison and,
  honestly, loses on this data size.
- **Stage-1 NON_FINANCIAL is rare (54/1000)** so recall is modest; corpus precision is ~95%, so
  the gate is a refinement, not a workhorse. The recall/precision knob is exposed and reported.
- **Temporal split is not additionally channel-grouped** (by design — it measures forward-in-time
  generalisation); a channel active before and after 2018 can appear on both sides.
- **Reproducibility:** all seeds fixed (`SEED=42`); the reproducibility block (versions, seed,
  split sizes, class balance, deployed/holdout sizes) is printed by `train.py` and saved in
  `metrics/metrics.json`. UTF-8 stdout is forced so logging cannot crash on report glyphs.

## 6. Delivery checklist (self-verified)
- [x] No in-sample metric reported anywhere; demo CSV held-out; overlap warning verified (0.85 on training CSV, 0.00 on hold-out).
- [x] Model C (fine-tuned DistilBERT) trains, evaluated, in the results table; CPU/skip paths logged.
- [x] Calibration applied; Brier before/after reported; app + batch use calibrated probabilities.
- [x] SHAP (sparse-correct TreeSHAP) top-20 human-readable; codebook-marker ranks/present-direction tabulated in `interpretability.md` (RISK_OMISSION #3, UNDISCLOSED_PROMO #4, both → MISLEADING).
- [x] Stage-1 tuned threshold + PR-AUC + recall/precision operating points reported.
- [x] Bootstrap 95% CIs on macro-F1; per-strategy breakdown; A-vs-C McNemar significance test.
- [x] Batch mode does gold-comparison when a `label` column is present; appends 3 cols, order preserved, downloadable.
- [x] `results.md` regenerated; README states the out-of-sample-only policy; app runs with one command.
