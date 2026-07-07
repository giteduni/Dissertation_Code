# Design & Methodology — A Text-Only Classifier for Misleading vs. Educational Financial YouTube Content

**An explanatory research document.**
Author of record: s2829951 · MSc dissertation, School of Informatics, University of Edinburgh.
Document status: explanatory companion to the software in `code/`. It contains **no code** and
prescribes **no code changes**; it documents *why*, *what*, *when*, and *how* the system was
designed, built, and validated, at a level suitable for a methods chapter, a viva, or a
replication attempt.

> **Where this lives:** `code/docs/DESIGN_AND_METHODOLOGY.md`.
> Companion machine-generated artefacts it refers to: `code/metrics/results.md`,
> `code/metrics/interpretability.md`, `code/metrics/metrics.json`, and the figures in
> `code/metrics/*.png`. The runnable system is documented operationally in `code/README.md`.

---

## Table of contents
1. [Motivation and research context (the *why*)](#1-motivation-and-research-context-the-why)
2. [Problem formulation (the *what*)](#2-problem-formulation-the-what)
3. [Data and ground truth](#3-data-and-ground-truth)
4. [Methodological principles that constrain every later choice](#4-methodological-principles)
5. [Feature design and the codebook-to-feature mapping](#5-feature-design)
6. [The three model families and why each exists](#6-the-three-model-families)
7. [The evaluation harness (leakage-safe by construction)](#7-the-evaluation-harness)
8. [Probability calibration](#8-probability-calibration)
9. [Interpretability: triangulating the human codebook with SHAP](#9-interpretability)
10. [System architecture and the inference contract](#10-system-architecture)
11. [The web application](#11-the-web-application)
12. [Results and how to read them](#12-results)
13. [Threats to validity and limitations](#13-threats-to-validity)
14. [Reproducibility](#14-reproducibility)
15. [Chronological development narrative (the *when*) and decision log](#15-development-narrative)
16. [Glossary and references](#16-glossary-and-references)

---

## 1. Motivation and research context (the *why*)

### 1.1 The phenomenon
Short-form "hustlesphere" finance content — day-trading tips, dropshipping funnels,
"passive-income" pitches, crypto calls — has become a mass-market channel of financial
advice that reaches audiences far larger than regulated advisers. The academic and regulatory
literature converges on a worrying picture:

- **Kakhbod, Kazempour, Livdan & Schürhoff (2023)** find that **56% of finfluencers are
  *antiskilled*** — they deliver systematically negative-alpha advice with confidence — and,
  critically, that **this low-quality, overconfident content attracts *larger* audiences**.
- **Jukwey (2024, MoneySuperMarket)**, reviewing 350+ short finance videos, reports **74%
  contained poor, misleading or dangerous tips**, **76% presented unrealistic gains while
  downplaying risk**, and **41% promoted get-rich-quick "money mindsets."**
- **IOSCO (2024, CR/08/2024)** codifies the regulatory hazards: undisclosed promotion, fake
  credentials/impersonation, pump-and-dump manipulation, and absence of risk disclosure.

The construct of interest is therefore **not** whether the advice is *financially correct*
(that would require following the trades). It is **communicative integrity**: does the
public-facing text of a video instruct honestly, or does it use unsubstantiated claims,
omitted risk, hidden incentives, or manipulation likely to induce poor financial decisions?

### 1.2 Why a text-only classifier
The dissertation studies a corpus of YouTube videos at scale. Watching every video is
infeasible and, more importantly, *unnecessary for the construct*: the markers of misleading
communication (guarantees, urgency, "turn $X into $Y", undisclosed course funnels) are
overwhelmingly present in the **title and description** — the same evidence a scrolling user
sees before clicking. Restricting the model to title + description (a) matches the human
annotation protocol exactly, (b) keeps the system cheap and reproducible, and (c) makes the
classifier deployable as a pre-watch triage signal. This is a deliberate scoping decision, not
a limitation of convenience, and it is defended again in §13.

### 1.3 The research questions this system serves
- **RQ1 (prevalence & temporal evolution):** how common is misleading content, and does it
  change across the ecosystem's history? → motivates the **temporal hold-out** and the
  **per-strategy** breakdown (§7).
- **RQ2 (operationalisation):** can the human "educational vs misleading" construct, defined in
  a pre-registered codebook, be reproduced by a machine from text alone — and does the machine
  *rediscover the same risk markers* the codebook defines? → motivates **Model A + SHAP** as a
  first-class scientific deliverable, not a baseline (§9).

---

## 2. Problem formulation (the *what*)

### 2.1 A two-stage decision
A naïve single classifier would be asked to do two incompatible jobs at once: decide whether a
video is *about money-making at all*, and, if so, *how honest it is*. These are different
questions with different base rates and different error costs, so the system splits them:

```
                 ┌──────────────────────────────┐
 title + desc ──▶│ STAGE 1  eligibility gate     │── NON_FINANCIAL ──▶ stop (ineligible)
   (+ engagement)│ FINANCIAL vs NON_FINANCIAL    │
                 └───────────────┬──────────────┘
                                 │ FINANCIAL
                                 ▼
                 ┌──────────────────────────────┐
                 │ STAGE 2  the core judgement   │
                 │ EDUCATIONAL vs MISLEADING     │──▶ calibrated  P(misleading) ∈ [0,1]
                 │                               │     + thresholded label + explanation
                 └──────────────────────────────┘
```

- **Stage 1 — eligibility.** Binary `NON_FINANCIAL` vs `FINANCIAL`, where
  `FINANCIAL = EDUCATIONAL ∪ MISLEADING ∪ BORDERLINE`. Its purpose is to convert residual
  corpus noise (a gaming video that matched a money keyword) into a *measured* precision
  statistic, and to ensure the misleading/educational judgement is only ever applied to videos
  for which it is defined. Only financial videos proceed.
- **Stage 2 — the core task.** Binary `MISLEADING` vs `EDUCATIONAL` on financial videos only,
  producing a probability `P(misleading)` and a thresholded label.

### 2.2 The four labels and how they map onto two binary tasks
The codebook uses four mutually-exclusive labels (§3). They are folded into the two binary
tasks as follows:

| Codebook label | Stage 1 target | Stage 2 target |
|---|---|---|
| `EDUCATIONAL` | FINANCIAL (negative) | EDUCATIONAL (negative) |
| `MISLEADING` | FINANCIAL (negative) | MISLEADING (positive) |
| `BORDERLINE` | FINANCIAL (negative) | *configurable* (see below) |
| `NON_FINANCIAL` | NON_FINANCIAL (positive) | — (gated out) |

**BORDERLINE handling** is a documented, configurable decision (`config.BORDERLINE_AS`):
- `exclude` (default, **primary analysis**) — BORDERLINE is dropped from Stage-2 training for a
  cleaner signal;
- `misleading` — folded into MISLEADING as a **sensitivity check**;
- `educational` — the symmetric alternative.

Results are reported for `exclude` as primary and `misleading` as sensitivity (§12). This makes
the treatment of genuine annotator uncertainty an explicit, auditable lever rather than a silent
default.

### 2.3 The deliverable prediction
The headline output is a **calibrated probability** `P(misleading)`, thresholded (default 0.5,
configurable) into a label, accompanied by **per-instance feature attributions** so a human can
see *why*. The probability — not just the label — is the product, because downstream prevalence
analysis and triage want a graded score, and because a calibrated score is interpretable as a
frequency ("≈80% of videos that look like this are misleading").

---

## 3. Data and ground truth

### 3.1 The labelled corpus
A single gold CSV of **1,000 videos** (`annotation_sample_verified_labelled_revised.csv`),
stratified across four money-making strategies (250 each): *Trading & Day-Trading*,
*E-commerce (Amazon/Shopify)*, *Entrepreneurship & Real Estate*, *Online Earning / Freelance /
WFH*. Each row carries, among other fields: `display_id`, `title`, `description`, `view_count`,
`like_count`, `upload_date`, `channel_id`, `strategy`, `quarter`, and the gold `label` plus a
`primary_flag` risk code.

**Observed class balance** (computed at runtime, not assumed):

| Label | Count | Share |
|---|---|---|
| EDUCATIONAL | 509 | 50.9% |
| MISLEADING | 371 | 37.1% |
| BORDERLINE | 66 | 6.6% |
| NON_FINANCIAL | 54 | 5.4% |

There are **669 distinct channels** across the 1,000 videos, and uploads span **2008–2019**
(≈460 before 2018-01-01, ≈540 on/after). These two facts drive the two evaluation protocols
in §7.

### 3.2 The annotation codebook (the source of the construct)
Labels come from a **pre-registered codebook** (`labelling/annotation_codebook.md`, v1.0) fixed
*before* annotation began. Its core is a set of **risk markers** observed from title +
description only:

| Code | Marker | Tier |
|---|---|---|
| `RETURN_GUARANTEE` | guaranteed / specific unrealistic returns ("turn £100 into £10k", "50% guaranteed") | Primary |
| `GET_RICH_QUICK` | effortless-fast wealth ("quit your job this month", "passive income while you sleep") | Primary |
| `RISK_OMISSION` | upside-only framing of an inherently risky activity, no risk acknowledgement | Primary |
| `UNDISCLOSED_PROMO` | course/signal/referral funnel masked as neutral advice | Primary |
| `PUMP_HYPE` | FOMO/hype to buy a specific asset ("last chance before it explodes") | Primary |
| `FAKE_CREDENTIAL` | fabricated expertise / impersonation | Primary |
| `URGENCY_SCARCITY` | manufactured time pressure ("today only", "spots closing") | Secondary |
| `UNVERIFIABLE_PROOF` | screenshots / "my student made £X" as the core claim | Secondary |

Decision rule: **≥1 primary marker present ⇒ MISLEADING**; only secondary/weak ⇒ BORDERLINE;
no marker and plausible ⇒ EDUCATIONAL; not money-making ⇒ NON_FINANCIAL. This taxonomy is the
*ground truth the classifier learns* and the *hypothesis SHAP tests* (does the model
independently re-derive these markers?). Reliability is protected by a double-coded subset with
Cohen's κ (reported in the dissertation's methods chapter; the human remains the primary
annotator of record).

### 3.3 Defensive ingestion
Real CSVs are messy, so loading (`finclass/data.py::load_gold`) is deliberately defensive:
UTF-8 with `keep_default_na=False`; numeric coercion of `view_count`/`like_count` that tolerates
blanks, commas and stray strings; `upload_date` parsed with an explicit `DD/MM/YYYY HH:MM`
format and coerced to `NaT` on failure; and a **grouping-key safeguard** — a blank `channel_id`
is replaced with a per-row unique token so that missing channels can never silently collapse
many rows into one leakage-causing group (§7).

---

## 4. Methodological principles

Four principles are fixed before any model is chosen; every later decision is downstream of
them. They are what separate a defensible scientific instrument from a leaderboard number.

1. **No leakage, enforced in code.** The same `channel_id` must never appear in both train and
   test; the gold `label` must never enter a feature matrix; and the deployed artefact must not
   be evaluated on data it trained on. Each of these is implemented and *checked*, not merely
   intended (§7, §10).
2. **Out-of-sample only.** Every reported number is from channel-grouped cross-validation or a
   forward-in-time hold-out. No in-sample score appears anywhere in the outputs, and the app
   actively warns if a user scores a file that overlaps the training set (§10.4).
3. **Regularise and estimate with uncertainty, because the data is small.** ~880 financial
   videos is small; the system prefers regularisation, class weighting, and cross-validated
   estimates with **confidence intervals** over any single split.
4. **Interpretability is a result, not a courtesy.** The interpretable model exists to *test the
   codebook*, so its explanation pipeline is held to the same correctness standard as its
   accuracy (a subtle violation of this is documented in §9.3 and §15).

---

## 5. Feature design

All features are computed identically at train and inference time by one module
(`finclass/text.py`), so there is exactly one definition of every feature. Construction is
deterministic and dependency-light.

### 5.1 Text cleaning
Lowercase; strip URLs, `@`/`#` handles, and emoji; collapse whitespace — **but deliberately
keep `$`, `%`, digits, and `k`/`m` suffixes**, because `"$10k"` and `"50%"` carry the signal.
A naïve cleaner that strips punctuation would destroy the very tokens the codebook cares about.

### 5.2 Two text representations
- **Sparse TF-IDF** for the interpretable model: word 1–2 grams (`min_df=2`) and, in the
  *comparison* variant, character 3–5 grams. Char n-grams add robustness to spelling/spacing
  but are uninterpretable in SHAP, so they are **off in the deployed Model A** and only switched
  on to *measure their F1 cost* (§12).
- **Dense sentence embeddings** (`all-MiniLM-L6-v2`, 384-d, L2-normalised) for the embedding
  model, **cached to disk** keyed by a SHA-1 of the cleaned text so they are computed once.

### 5.3 Engineered linguistic features (the interpretable signal)
A fixed-order vector of human-named features per video:
- **Engagement:** `view_count`, `like_count`, `like_to_view = like/(view+1)`, `log_views`,
  `log_likes`.
- **Title surface signals:** length (chars/words), ALLCAPS ratio, `!`/`?` counts, presence of a
  currency symbol, presence of a digit, presence of a `digits+k/m` token, emoji count.
- **VADER** sentiment compound of the title.
- **Description signals:** length, word count, emptiness flag, URL count.

### 5.4 The codebook lexicons — the bridge between theory and features
For each codebook risk marker, a small, auditable set of regular expressions emits a
**count/binary feature** (`lex_RETURN_GUARANTEE`, `lex_GET_RICH_QUICK`, …). For example
`RETURN_GUARANTEE` matches `guaranteed`, `\d+\s*x` ("10x"), `turn $X into`, `\d+%\s*(returns|
profit|monthly)`, `double your money`, `risk-free`; `RISK_OMISSION` is a *heuristic* — a risky
asset class (day-trading, forex, crypto, options, leverage) is mentioned **and** no risk
vocabulary (`risk`, `lose`, `volatile`, `not financial advice`, `DYOR`) appears. These features
are the **operationalisation of the codebook inside the model**, and they are what allow SHAP to
answer RQ2: *does the model, free to use thousands of n-grams, nonetheless rank the
codebook-derived features highly?*

### 5.5 Leakage-safe transformers
The features are packaged as picklable scikit-learn transformers (`CleanTextExtractor`,
`EngineeredFeatures`) inside a `FeatureUnion`, so the exact fitted pipeline serialises and
reloads. The transformers accept only `title`, `description`, `view_count`, `like_count` — the
gold `label` is structurally absent from the feature path, which is principle #1 enforced by
construction rather than vigilance.

---

## 6. The three model families

The brief asks for an interpretable model and a performance model; the system implements **three**
so the comparison is honest and complete.

### 6.1 Model A — XGBoost over TF-IDF + engineered/lexicon features (the scientific instrument)
A gradient-boosted tree ensemble (400 trees, depth 5, learning-rate 0.05, subsampling and
`colsample_bytree=0.5`, L1/L2 regularisation, `scale_pos_weight` set per fold from the training
labels) over the sparse union of word TF-IDF and the engineered/lexicon block. **Why XGBoost
here:** it handles sparse high-dimensional text natively, captures non-linear interactions among
lexicon markers and engagement, and — decisively — admits **exact TreeSHAP** attributions. This
is the deployed, explainable model and the vehicle for RQ2.

### 6.2 Model B — frozen MiniLM embeddings + Logistic Regression (the embedding baseline)
Frozen sentence embeddings → standardisation → class-weighted logistic regression. This is the
**sanctioned CPU fallback** for the "transformer family" when no GPU fine-tuning is run: it
captures distributional semantics a bag-of-features misses, while remaining cheap and
deterministic. Embeddings are disk-cached.

### 6.3 Model C — fine-tuned DistilBERT (the genuine transformer)
`distilbert-base-uncased` fine-tuned on `title [SEP] description` (max 256 tokens) with a
**class-weighted** cross-entropy loss, AdamW (lr 2e-5, weight-decay 0.01, gradient clipping),
**early-stopping on validation macro-F1**, everything seeded. The training loop is a
self-contained PyTorch loop (no `accelerate`/`datasets` dependency) so it runs anywhere; it is
**GPU-aware** (trains on GPU if present, else CPU with an explicit time warning) and **flagged**
(`TRAIN_TRANSFORMER=0` skips it and logs that Model B stands in; `REUSE_MODEL_C=1` reloads a
saved fine-tune). Because full channel-grouped CV of a 66M-parameter model is prohibitively
expensive, Model C is evaluated on a **single channel-disjoint train/val/test split**, and that
trade-off is stated wherever its number appears.

### 6.4 Stage-1 gate
A class-weighted logistic regression over the **word + char** TF-IDF union (char n-grams are
*kept* here: the gate is not a SHAP deliverable, and sub-word cues help separate gaming/vlog
false-positives). LR is chosen over a tree because financial-vs-not is essentially a *topical*
decision and LR is stable at the 5% positive base rate.

---

## 7. The evaluation harness

Evaluation is the heart of the scientific contribution; the harness (`finclass/evaluation.py`)
is built so that the *only* way to produce a number is an honest one.

### 7.1 Two protocols, both always reported
1. **Channel-grouped 5-fold CV** (`GroupKFold` on `channel_id`) — the primary estimate. No
   channel ever straddles the train/test boundary, which prevents the model from learning
   *channel identity* as a shortcut (a leakage mode that silently inflates naïve splits). Each
   model is rebuilt per fold via a `build_fn(y_train)` factory so that imbalance handling (e.g.
   XGBoost `scale_pos_weight`) is derived **only** from the fold's own training labels.
2. **Temporal hold-out** — train on `upload_date < 2018-01-01`, test on `≥`. This directly
   measures **RQ1 forward-in-time generalisation**: can a model trained on the early ecosystem
   classify its later, evolved forms? Some degradation is expected and discussed.

### 7.2 Metrics
For every protocol/model: **macro-F1** (primary), per-class precision/recall/F1, ROC-AUC,
**PR-AUC** (reported for the imbalanced Stage 1 because it is more honest than ROC-AUC at a 5%
base rate), confusion matrix, a **calibration curve**, and a **per-strategy** macro-F1 breakdown
that ties performance back to RQ1. Stage-2 macro-F1 is also given **bootstrap 95% confidence
intervals** (2,000 resamples of the test predictions), and the per-fold macro-F1 spread yields a
fold-level CI — two complementary uncertainty estimates on a small dataset.

### 7.3 Threshold reporting
The default decision threshold is 0.5, but the system reports the **precision/recall-vs-threshold
curve** and, for Stage 1, multiple **operating points**: the default, the macro-F1-optimal
threshold, and a deliberately **high-recall** point (the lowest threshold reaching
NON_FINANCIAL recall ≥ 0.60) with its precision cost. This makes the recall/precision trade-off
an explicit, reported quantity rather than a hidden hyper-parameter.

### 7.4 Significance testing
Model A and Model C are compared with a **McNemar exact test** on the *same* test rows (Model A
is refit on Model C's exact train+val split, then both predict the identical test set), so the
claim "A vs C" carries a p-value rather than two point estimates whose error bars are unknown.

### 7.5 The demo hold-out (deployment honesty)
Separately from evaluation, a **channel-disjoint 15% slice** is carved out and **excluded from
the deployed model fit**, then written to `data/demo_holdout.csv` (+ a `_gold.csv` with labels).
This guarantees the app can demonstrate genuinely *unseen* predictions, and underpins the
in-sample overlap warning (§10.4). The CV/temporal estimates still use the full data (they are
cross-validated and therefore not contaminated by the deployed artefact's training set); the
deployed model trains on the remaining 85% (853 rows).

---

## 8. Probability calibration

A tree ensemble or an embedding-LR can rank well yet be **over-confident** — its 0.9 may not
mean "90% of such cases are misleading." Because the *score* is the product, Stage-2 models are
wrapped in `CalibratedClassifierCV`, fit by **internal cross-validation within the training fold
only** (so the test fold is never touched and the reported Brier/curves stay out-of-sample). The
method is chosen by data size — **isotonic** when the calibration set is large enough, otherwise
**Platt/sigmoid** (the robust choice on small data). **Brier score is reported before and after**
calibration. Calibration improves Model B substantially (Brier 0.280 → 0.187) and leaves the
already-well-calibrated Model A essentially unchanged (0.127 → 0.123). The app gauge and the
batch `p_misleading` column use the **calibrated** probabilities.

---

## 9. Interpretability

### 9.1 The scientific claim
RQ2 asks whether the human codebook is *recoverable from text by a machine*. The test: give
Model A thousands of free n-gram features plus the codebook-derived lexicon features, train it
only on labels, then ask **SHAP** whether the lexicon/engineered codebook features rank highly
and push in the codebook-implied direction. A "yes" is independent, mechanistic evidence that the
human markers are real, learnable signals — not annotator artefacts.

### 9.2 What SHAP shows (corrected results)
Using **exact TreeSHAP** on a held sample, **9 of the global top-20 features are
engineered/codebook features.** Judged **when the marker is actually present** in a video:

| Codebook marker | Global SHAP rank | # present in sample | Direction when present |
|---|---|---|---|
| `RISK_OMISSION` | 3 | 55 | → MISLEADING |
| `UNDISCLOSED_PROMO` | 4 | 131 | → MISLEADING |
| `GET_RICH_QUICK` | 60 | 28 | → MISLEADING |
| `FAKE_CREDENTIAL`, `PUMP_HYPE`, `UNVERIFIABLE_PROOF`, `URGENCY_SCARCITY`, `RETURN_GUARANTEE` | low | few | not used as a lexicon feature |

The top of the list is dominated by `title_has_currency`, `w:course`, `title_has_number`,
`w:get` — i.e. exactly the surface cues the codebook describes (currency claims, course funnels,
quantified promises). Markers that rank low *as lexicon features* are not absent from the model:
their literal cues (`$`, `%`, digits, "10x") are **redundantly captured** by the engineered
`title_has_currency`/`title_has_number` features and by word n-grams, so the tree splits on those
instead. This redundancy is reported honestly rather than presented as the marker being unused.

### 9.3 The correctness subtlety that makes this trustworthy (a cautionary methods note)
TreeSHAP must be computed on the **same sparse representation the model was trained on**.
XGBoost is *sparsity-aware*: in a sparse row a `0` means "missing" and routes down a different
branch than an explicit dense `0`. An early version of both the app explanation and the global
SHAP plot densified the matrix first; the dense `0`s routed incorrectly, producing a large,
near-constant, *backwards* attribution to `title_has_currency` (it appeared as the top "reason"
even for videos with no currency) and an explanation that *disagreed with the displayed
probability*. The fix — compute attributions on the **sparse** matrix via XGBoost's native
`pred_contribs` (exact TreeSHAP, also ~18× faster than the generic explainer) — makes the
explanation correct, consistent with the gauge, input-responsive, and fast. The lesson is
general and worth a sentence in any methods chapter: **an interpretability method inherits the
input contract of the model it explains.**

---

## 10. System architecture

### 10.1 Module map
```
code/
├── config.py                 single source of truth: paths, seeds, thresholds, calibration, transformer
├── finclass/
│   ├── text.py               cleaning, codebook lexicons, engineered features, VADER, sklearn transformers
│   ├── data.py               gold loading, Stage-1/Stage-2 views, channel-grouped / temporal / demo splits, fingerprints
│   ├── pipeline.py           feature union + Model A / B / Stage-1 builders + calibration
│   ├── transformer_model.py  Model C — fine-tune + reload/evaluate (self-contained PyTorch)
│   └── evaluation.py         CV, temporal, metrics, Brier/ECE, bootstrap CI, McNemar, threshold tuning, plots
├── train.py                  orchestrates everything; writes models/, metrics/, results.md, interpretability.md
├── predict.py                classify() + classify_batch() + overlap/gold helpers (the inference contract)
└── app.py                    Streamlit two-mode UI
```

### 10.2 Separation of concerns
`config.py` holds every tunable so `train.py`, `predict.py` and the app share one source of
truth. Feature logic lives only in `text.py`; split logic only in `data.py`; metric logic only
in `evaluation.py`. `train.py` is the *only* writer of artefacts; `predict.py` is a pure reader
that never retrains.

### 10.3 The inference contract (`predict.py`)
A lazily-loaded, process-wide singleton bundle loads the serialised Stage-1 gate, the
calibrated Stage-2 models (for probabilities), and the uncalibrated Model A (for fast TreeSHAP
explanations). `classify(title, description, view_count, like_count)` returns a structured dict:
Stage-1 eligibility and `P(financial)`; if financial, a **calibrated** `P(misleading)` from both
Model A and Model B, the thresholded label, and the **top contributing signals** (sparse
TreeSHAP, signed toward MISLEADING, absent n-grams hidden). `classify_batch(df)` vectorises this
and is tolerant of header-name variants (`Title`/`Description`) and missing columns; it appends
exactly three columns (`predicted_eligibility`, `p_misleading`, `predicted_label`) and preserves
all original columns and row order.

### 10.4 The leakage guard
At training time, identity **fingerprints** of the deployed-train rows (display-ids + hashed
normalised titles) are saved. At inference, `training_overlap(df)` reports the fraction of
uploaded rows that match training; above 50% the app raises a visible banner that any metrics on
that file are **in-sample and invalid**. This operationalises principle #2 at the point of use,
not just in the report.

---

## 11. The web application

A local Streamlit app (`app.py`) loads the serialised models once (no retraining at startup) and
offers two modes:

- **Mode 1 — Single video.** Title + description (+ optional engagement) → the Stage-1 eligibility
  result; if financial, a prominent **calibrated P(misleading) gauge**, the EDUCATIONAL/MISLEADING
  label, and a **"Why" table** of the top contributing signals (per-instance sparse TreeSHAP),
  with live-adjustable thresholds.
- **Mode 2 — CSV batch.** Upload a CSV → every row through Stage 1 then Stage 2 → a downloadable
  output that is the **input plus three appended columns**, with a summary, a label-distribution
  chart, the **in-sample overlap banner**, and — if the CSV carries a `label` column — a
  **confusion matrix + macro-F1 vs the supplied labels** (clearly noted as in/out-of-sample).

An operational footnote: the app disables Streamlit's source-file watcher
(`.streamlit/config.toml`). The watcher walks every imported module including the entire
`transformers` model zoo, triggering benign-but-noisy `torchvision` import errors for vision
models the system never uses; disabling it silences the noise and speeds startup without
affecting behaviour.

---

## 12. Results

All figures below are **out-of-sample** and are regenerated by `train.py` into
`metrics/results.md` and `metrics/metrics.json`; the SHAP table is in `metrics/interpretability.md`.

### 12.1 Stage 1 — eligibility gate
| Protocol | macro-F1 | PR-AUC | ROC-AUC |
|---|---|---|---|
| Channel-grouped CV | 0.756 ± 0.029 | 0.556 | 0.910 |
| Temporal hold-out (≥2018) | 0.734 | — | — |

FINANCIAL is recovered almost perfectly; the macro average is pulled down by the rare
NON_FINANCIAL class (54/1000). Operating points: default thr 0.50 → recall 0.43 / precision 0.72;
high-recall thr 0.31 → recall 0.61 / precision 0.41. The gate is deliberately a high-precision
*refinement* (corpus precision ≈95%), not a workhorse classifier.

### 12.2 Stage 2 — misleading vs educational (primary: BORDERLINE excluded)
| Model | CV macro-F1 (bootstrap 95% CI) | Temporal | CV ROC-AUC | Brier raw→calibrated |
|---|---|---|---|---|
| **A · XGBoost (word + lexicon)** | **0.811 (0.784–0.838)** | 0.815 | 0.903 | 0.127 → 0.123 |
| B · MiniLM emb + LR | 0.671 (0.642–0.704) | 0.654 | 0.745 | 0.280 → 0.187 |
| C · DistilBERT (fine-tuned) | 0.722 (single channel-disjoint split, n_test=174) | — | 0.805 | — |

Supporting findings:
- **Feature cleanup is free.** Removing char n-grams from Model A *raised* CV macro-F1 from
  0.797 to **0.811** while making SHAP human-readable — interpretability at no accuracy cost.
- **Sensitivity.** Folding BORDERLINE into MISLEADING gives Model A 0.784 — a small, expected dip
  that does not change conclusions.
- **Per-strategy (RQ1).** E-commerce 0.835 · Online-Earning 0.803 · Trading 0.792 ·
  **Entrepreneurship & Real Estate 0.698** (the hardest stratum — fewest misleading examples,
  n_pos=42, hence widest uncertainty).

### 12.3 The model comparison — a reportable "wrong-way" result
On this ~880-example, lexicon-rich corpus the **interpretable XGBoost beats the fine-tuned
DistilBERT** on the identical channel-disjoint test split: **0.783 vs 0.722** macro-F1. The
McNemar exact test gives **p = 0.065** — A is numerically better but the difference is *not*
significant at α = 0.05. This is consistent with the literature on small-data, lexically-distinct
problems, where a 66M-parameter LM cannot be estimated well from ~700 rows and a sparse
feature+lexicon model is competitive or superior. It is reported plainly rather than hidden, and
the recommendation follows the evidence: **deploy and explain with Model A**, expecting Model C
to overtake with substantially more labelled data.

### 12.4 Interpretability
Reported in §9.2: 9/20 top features are codebook/engineered; `RISK_OMISSION` (#3),
`UNDISCLOSED_PROMO` (#4) and `GET_RICH_QUICK` (#60) all push **→ MISLEADING when present** — the
model independently recovers the human codebook.

---

## 13. Threats to validity

- **Construct boundary (text-only).** The model judges *communicative integrity from public text*,
  not advice quality. A video could be honestly framed yet financially harmful, or salesy yet
  benign. This is the intended construct (§1.2), but it bounds the claims: outputs are a proxy for
  *likely-misleading presentation*, not adjudicated harm.
- **Promotional confound.** Top signals include `course`, currency and funnel cues; the model may
  partly detect "has a sales funnel" rather than "misleading," and the per-strategy spread
  (E-commerce 0.835 vs Entrepreneurship 0.698) is consistent with that. The codebook treats
  *undisclosed* promotion masked as advice as misleading, which mitigates but does not eliminate
  the confound; a disclosed affiliate link in genuine instruction stays EDUCATIONAL by design.
- **Label ceiling.** The classifier cannot exceed the reliability of its labels; the dissertation
  reports Cohen's κ so the model's macro-F1 can be read against human–human agreement. A single
  primary annotator (with an LLM/second-annotator κ subset) is the practical constraint.
- **Small-data uncertainty.** ~880 financial videos and ~8 misleading examples per fold in the
  smallest stratum mean per-strategy numbers are noisy; this is why CIs are reported and why
  per-strategy claims are stated cautiously.
- **Temporal split is not additionally channel-grouped.** By design it measures forward-in-time
  generalisation, so a channel active before and after 2018 can appear on both sides; the temporal
  number is therefore an optimistic bound on true temporal novelty, and this is stated.
- **Transformer evaluated on a single split.** Model C's 0.722 has wider unreported variance than
  Model A's CV estimate; the comparison is honest about this asymmetry.
- **Lexicon brittleness.** Regex lexicons can miss paraphrase and novel slang; they are an
  auditable *operationalisation* of the codebook, not a complete semantics, which is precisely why
  a learned model (and the embedding/transformer arms) is layered on top.

---

## 14. Reproducibility

- **Determinism.** All seeds fixed (`SEED = 42`) across NumPy, Python, scikit-learn, XGBoost and
  PyTorch; splits are deterministic (`GroupKFold`, seeded `GroupShuffleSplit`).
- **Environment.** Pinned `requirements.txt`; developed/validated on Python 3.13, CPU-only.
- **Reproducibility block.** `train.py` prints and saves (in `metrics/metrics.json`) the Python
  and library versions, seed, total rows and class balance, channel count, deployed-train and
  demo-hold-out sizes, calibration method, and the chosen configuration flags.
- **One-command path.** Raw CSV → `python train.py` → `models/` + `metrics/` → `streamlit run
  app.py`. Flags (`BORDERLINE_AS`, `TRAIN_TRANSFORMER`, `REUSE_MODEL_C`, `PRIMARY_STAGE2_MODEL`)
  reproduce every reported variant.
- **Robust logging.** stdout is forced to UTF-8 so report glyphs cannot crash a run (a real bug
  encountered and fixed — see §15).

---

## 15. Development narrative (the *when*) and decision log

The system was built in **four passes**, each leaving the previous behaviour working. This
history is itself methodologically informative: it shows where naïve choices were caught and
corrected.

**Pass 1 — Baseline two-stage system.** Establish the data contract, the codebook-to-feature
mapping, channel-grouped CV + temporal hold-out, Model A (XGBoost) and the embedding Model B, the
SHAP deliverable, `predict.py`, and the two-mode Streamlit app. Outcome: Model A macro-F1 ≈ 0.81,
codebook markers ranking highly — the core scientific result in place.

**Pass 2 — Critical appraisal.** A self-review against examiner expectations identified four
weaknesses: the "transformer" was only frozen embeddings; no human-agreement ceiling; potential
promotional confound; and probabilities were uncalibrated. These became the Pass-3 agenda.

**Pass 3 — Publication-grade hardening (seven priorities).**
1. *Leakage in the demo* — a batch demo had been run on the training data, yielding a meaningless
   100%-agreement output. Fixed permanently with a channel-disjoint demo hold-out excluded from
   deployment, training fingerprints, an in-app overlap warning, and an out-of-sample-only policy.
2. *A genuine transformer* — DistilBERT fine-tuning added as Model C with class weights, early
   stopping, CPU-awareness and a skip flag.
3. *Calibration* — `CalibratedClassifierCV` with Brier reported before/after; the app uses
   calibrated probabilities.
4. *Interpretable feature space* — char n-grams dropped from Model A (SHAP became readable and F1
   *improved* 0.797 → 0.811); `interpretability.md` tabulating each codebook marker's rank and
   direction added as a first-class output.
5. *Stronger Stage 1* — PR-AUC, threshold tuning, and an explicit high-recall operating point.
6. *Reporting rigour* — bootstrap CIs, per-strategy breakdown, and a McNemar A-vs-C significance
   test.
7. *Batch gold-comparison* — confusion matrix + macro-F1 when an uploaded CSV carries labels.

   *Bugs caught and fixed during Pass 3:* a Windows `cp1252` `UnicodeEncodeError` on a `✓` glyph
   crashed the run after Model C trained (fixed with ASCII logging + a UTF-8 stdout guard, and a
   `REUSE_MODEL_C` reload path so the expensive fine-tune was not wasted); and re-enabling char
   n-grams for the Stage-1 gate only (a global toggle had inadvertently weakened it).

**Pass 4 — Explanation correctness and latency.** A user observed the app's "Why" table appeared
frozen and the app had slowed. Investigation (documented in §9.3) traced both symptoms to a single
root cause: the per-instance explanation densified the feature matrix before SHAP, but XGBoost is
sparsity-aware (dense `0` ≠ missing `0`). The dense path explained a *different, wrong* prediction
than the gauge and produced a near-constant dominant feature, while the generic `shap` library was
≈18× slower than necessary. Replacing it with native sparse `pred_contribs` (exact TreeSHAP) made
the explanation correct, gauge-consistent, input-responsive, and fast; the same fix was applied to
the training-time global SHAP, which *strengthened* the codebook result (4 → 9 codebook/engineered
features in the top-20, with present-conditional directions).

**Design forks and the reasoning taken at each (summary):**
- *Two stages vs one* → two, to separate incompatible base rates and error costs (§2.1).
- *BORDERLINE* → exclude as primary, sensitivity-checked, configurable (§2.2).
- *XGBoost as the interpretable model* → for native exact TreeSHAP and sparse handling (§6.1).
- *Which model the app headlines* → Model A, because every score must be explainable, even though
  one might "deploy the best raw model"; here A is also the best (§6, §12.3).
- *Calibration within-fold* → to keep test folds untouched and Brier out-of-sample (§8).
- *Demo hold-out excluded from deployment* → so the app's demonstration is genuinely unseen (§7.5).

---

## 16. Glossary and references

### Glossary
- **Macro-F1** — unweighted mean of per-class F1; robust to class imbalance, hence the primary
  metric.
- **Channel-grouped CV** — cross-validation where all videos from a channel are kept together in a
  single fold, preventing the model from memorising channel identity.
- **Temporal hold-out** — train on the past, test on the future; measures forward-in-time
  generalisation.
- **Calibration / Brier score** — how well predicted probabilities match observed frequencies;
  Brier is the mean squared error of probabilistic forecasts (lower is better).
- **TreeSHAP / `pred_contribs`** — exact Shapley-value feature attributions for tree ensembles;
  XGBoost's native implementation, computed on the *sparse* matrix to match training.
- **McNemar test** — paired significance test comparing two classifiers' correctness on the same
  items.
- **Bootstrap CI** — confidence interval from resampling the test predictions with replacement.
- **PR-AUC** — area under the precision–recall curve; more informative than ROC-AUC under heavy
  class imbalance.

### References
- Kakhbod, A., Kazempour, S., Livdan, D., & Schürhoff, N. (2023). *Finfluencers.* Swiss Finance
  Institute / SSRN 4428232.
- Jukwey, E. (2024). *Finfluencer study: Risky business.* MoneySuperMarket.
- IOSCO (2024). *Finfluencers* (CR/08/2024). International Organization of Securities Commissions.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions
  (SHAP).* NeurIPS.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT.*
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT.* EMNLP. (`all-MiniLM-L6-v2`.)
- Hutto, C. J., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment
  Analysis of Social Media Text.* ICWSM.
- Platt, J. (1999) / Zadrozny & Elkan (2002). Probability calibration (Platt scaling / isotonic).

---

*End of document. This file is documentation only; it changes no application behaviour. For
operational instructions see `code/README.md`; for machine-generated results see
`code/metrics/`.*
