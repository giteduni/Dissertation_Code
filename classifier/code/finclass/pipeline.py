"""Feature transformers and the Model A / Model B / Stage-1 estimator builders.

Model A  - TF-IDF (word 1-2g + char 3-5g) + engineered/lexicon features -> XGBoost.
           Fully explainable (SHAP over named features); the scientific deliverable.
Model B  - frozen sentence-embeddings (all-MiniLM-L6-v2, disk-cached) -> Logistic
           Regression. This is the *sanctioned CPU fallback* for the transformer family
           (no GPU available); it stands in for DistilBERT fine-tuning.
Stage 1  - same TF-IDF text union -> Logistic Regression (financial vs non-financial is
           essentially a topical decision; LR is stable under the 5% positive rate).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from xgboost import XGBClassifier

import config
from finclass.text import CleanTextExtractor, EngineeredFeatures, combine


# Model A: interpretable TF-IDF and engineered union

def build_text_union(char_enabled: bool | None = None) -> FeatureUnion:
    """Word TF-IDF + engineered/lexicon features, optionally with char n-grams.

    ``char_enabled`` defaults to ``config.CHAR_NGRAMS_ENABLED`` (False for the deployed,
    interpretable Model A). Char n-grams help raw F1 slightly but produce uninterpretable
    SHAP fragments, so they are off by default and only switched on for the F1-cost report.
    """
    use_char = config.CHAR_NGRAMS_ENABLED if char_enabled is None else char_enabled
    word = Pipeline([
        ("clean", CleanTextExtractor()),
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=config.WORD_NGRAM_RANGE,
            min_df=config.WORD_MIN_DF,
            max_features=config.WORD_MAX_FEATURES,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
    ])
    eng = Pipeline([
        ("eng", EngineeredFeatures()),
        ("scale", MaxAbsScaler()), 
    ])
    branches = [("word", word)]
    if use_char:
        char = Pipeline([
            ("clean", CleanTextExtractor()),
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=config.CHAR_NGRAM_RANGE,
                min_df=config.CHAR_MIN_DF,
                max_features=config.CHAR_MAX_FEATURES,
                sublinear_tf=True,
            )),
        ])
        branches.append(("char", char))
    branches.append(("eng", eng))
    return FeatureUnion(branches)


def build_model_a(scale_pos_weight: float = 1.0, char_enabled: bool | None = None) -> Pipeline:
    """Model A: feature union -> regularised XGBoost (handles sparse input natively)."""
    clf = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.5,
        reg_lambda=2.0,
        reg_alpha=0.5,
        min_child_weight=2.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=config.SEED,
    )
    return Pipeline([("features", build_text_union(char_enabled)), ("clf", clf)])


def feature_names_a(model_a: Pipeline) -> List[str]:
    """Flat, human-readable feature names aligned to Model A's fitted feature matrix.

    The branch order must match ``FeatureUnion`` concatenation order (word, [char], eng).
    For the engineered branch we read names from the ``EngineeredFeatures`` step (the
    trailing ``MaxAbsScaler`` does not carry meaningful names), so codebook-lexicon
    features keep their ``lex_*`` identity for SHAP.
    """
    from finclass.text import ENGINEERED_FEATURE_NAMES

    union: FeatureUnion = model_a.named_steps["features"]
    names: List[str] = []
    for name, trans in union.transformer_list:
        if name in ("word", "char"):
            sub = list(trans.named_steps["tfidf"].get_feature_names_out())
            prefix = "w:" if name == "word" else "c:"
            names.extend(f"{prefix}{s}" for s in sub)
        elif name == "eng":
            names.extend(ENGINEERED_FEATURE_NAMES)
        else:  
            last = trans.steps[-1][1]
            names.extend(f"{name}:{s}" for s in last.get_feature_names_out())
    return names



# Probability calibration

def resolve_calibration_method(n_calib: int) -> str:
    """'isotonic' when the calibration set is large enough, else 'sigmoid' (Platt)."""
    if config.CALIBRATION_METHOD in ("isotonic", "sigmoid"):
        return config.CALIBRATION_METHOD
    return "isotonic" if n_calib >= config.CALIBRATION_ISOTONIC_MIN_N else "sigmoid"


def calibrate(base_estimator, n_train: int):
    """Wrap a (fresh, unfitted) estimator in CalibratedClassifierCV.

    Calibration is fit by internal CV *within the training fold only*, so the outer test
    fold is never touched — the reported Brier/curves remain out-of-sample.
    """
    from sklearn.calibration import CalibratedClassifierCV

    method = resolve_calibration_method(n_train)
    return CalibratedClassifierCV(base_estimator, method=method, cv=config.CALIBRATION_INTERNAL_CV)

# Model B: Logistic Regression
_ST_MODELS: Dict[str, object] = {} 


def _get_st_model(name: str):
    if name not in _ST_MODELS:
        from sentence_transformers import SentenceTransformer 

        _ST_MODELS[name] = SentenceTransformer(name, device="cpu")
    return _ST_MODELS[name]


class EmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """DataFrame -> (n, dim) sentence embeddings, with a persistent disk cache.

    The live SentenceTransformer is never pickled (only ``model_name`` is), so the fitted
    Model B serialises to a few KB and reloads instantly.
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL, cache_dir: str | None = None):
        self.model_name = model_name
        self.cache_dir = cache_dir or str(config.CACHE_DIR)

    def _cache_path(self) -> Path:
        safe = self.model_name.replace("/", "_")
        return Path(self.cache_dir) / f"emb_cache_{safe}.npz"

    def _load_cache(self) -> Dict[str, np.ndarray]:
        path = self._cache_path()
        if path.exists():
            with np.load(path, allow_pickle=True) as data:
                return {k: data[k] for k in data.files}
        return {}

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def fit(self, X, y=None):  
        return self

    def transform(self, X):  
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            texts = [combine(r.get("title"), r.get("description")) for _, r in X.iterrows()]
        elif isinstance(X, dict):
            texts = [combine(X.get("title"), X.get("description"))]
        else:
            texts = [combine(d.get("title"), d.get("description")) for d in X]

        cache = self._load_cache()
        keys = [self._key(t) for t in texts]
        missing = sorted({k: t for k, t in zip(keys, texts) if k not in cache}.items())
        if missing:
            model = _get_st_model(self.model_name)
            vecs = model.encode(
                [t for _, t in missing], batch_size=64, show_progress_bar=False, normalize_embeddings=True
            )
            for (k, _), v in zip(missing, np.asarray(vecs, dtype=np.float32)):
                cache[k] = v
            np.savez_compressed(self._cache_path(), **cache)

        return np.vstack([cache[k] for k in keys]).astype(np.float64)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state  


def build_model_b() -> Pipeline:
    """Model B: cached embeddings -> standardise -> balanced Logistic Regression."""
    return Pipeline([
        ("emb", EmbeddingVectorizer()),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=3000, class_weight="balanced", random_state=config.SEED
        )),
    ])

# Stage 1: financial gate

def build_stage1() -> Pipeline:
    """Stage-1 gate: TF-IDF union -> balanced Logistic Regression (robust at 5% positives).

    Char n-grams are kept ON here: the gate is not a SHAP deliverable, and sub-word cues
    (e.g. game/vlog vocabulary) help separate non-financial corpus false-positives.
    """
    return Pipeline([
        ("features", build_text_union(char_enabled=True)),
        ("clf", LogisticRegression(
            C=2.0, max_iter=3000, class_weight="balanced", random_state=config.SEED
        )),
    ])
