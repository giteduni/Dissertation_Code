"""Text cleaning, codebook risk lexicons, and engineered linguistic features.

This module is deliberately dependency-light and deterministic so the *exact* same
feature construction runs at training time and at inference time. The lexicons are a
direct, auditable operationalisation of the human annotation codebook (§3 risk markers),
which is what lets SHAP later test whether the model rediscovers the human codebook.
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_VADER = SentimentIntensityAnalyzer()

# Cleaning
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HANDLE_RE = re.compile(r"[@#]\w+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  
    "\U00002600-\U000027BF"  
    "\U0001F1E6-\U0001F1FF"  
    "\U00002190-\U000021FF"  
    "\U00002B00-\U00002BFF"  
    "\U0000FE00-\U0000FE0F"  
    "\U00002000-\U0000206F"  
    "]+",
    flags=re.UNICODE,
)
_WS_RE = re.compile(r"\s+")
_KEEP_RE = re.compile(r"[^a-z0-9$%.,!?\s]")


def clean_text(text: str | float | None) -> str:
    """Lowercase; strip URLs, @/# handles, emojis; keep ``$ % digits``; collapse spaces.

    Designed for TF-IDF: ``$10k`` and ``50%`` survive because those tokens carry signal.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).lower()
    s = _URL_RE.sub(" ", s)
    s = _HANDLE_RE.sub(" ", s)
    s = _EMOJI_RE.sub(" ", s)
    s = _KEEP_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def combine(title: str | float | None, description: str | float | None) -> str:
    """Cleaned ``title`` + cleaned ``description`` joined for the bag-of-features models."""
    return (clean_text(title) + " " + clean_text(description)).strip()

LEXICONS: Dict[str, List[str]] = {
    "RETURN_GUARANTEE": [
        r"\bguarantee(d|s)?\b",
        r"\b\d+\s*x\b",                      
        r"turn\s+[£$€]?\s*\d",               
        r"\binto\s+[£$€]\s*\d",
        r"\bwill\s+(10|5|2|3|4)\s*x\b",
        r"\b\d{1,3}%\s*(returns?|profit|gains?|monthly|weekly|daily)\b",
        r"\b(double|triple)\s+your\s+(money|investment|account)\b",
        r"\brisk[\s-]?free\b",
    ],
    "GET_RICH_QUICK": [
        r"\bget\s+rich\b",
        r"\bquit\s+(your\s+)?(job|9\s*-?\s*5|nine\s+to\s+five)\b",
        r"\bpassive\s+income\b",
        r"\bovernight\b",
        r"\bwhile\s+you\s+sleep\b",
        r"\bfinancial\s+freedom\b",
        r"\beasy\s+money\b",
        r"\b(in|within)\s+(a|one|1)\s+(day|week|month)\b",
        r"\bno\s+(experience|skill|work)\s+(needed|required)\b",
        r"\bautopilot\b",
    ],
    "URGENCY_SCARCITY": [
        r"\btoday\s+only\b",
        r"\blast\s+chance\b",
        r"\bspots?\s+(left|closing|filling|remaining)\b",
        r"\b(before|until)\s+(it'?s\s+)?(too\s+late|gone)\b",
        r"\bhurry\b",
        r"\blimited\s+(time|spots?|seats?)\b",
        r"\bact\s+now\b",
        r"\bclosing\s+soon\b",
        r"\bdon'?t\s+miss\b",
    ],
    "UNVERIFIABLE_PROOF": [
        r"\bproof\b",
        r"\bscreenshots?\b",
        r"\bmy\s+student(s)?\s+(made|earned|got)\b",
        r"\bhow\s+i\s+made\b",
        r"\bi\s+made\s+[£$€]?\s*\d",
        r"\bresults?\s+(don'?t\s+lie|speak)\b",
        r"\blive\s+(proof|results)\b",
        r"\bmy\s+(income|earnings)\s+report\b",
    ],
    "UNDISCLOSED_PROMO": [
        r"\b(free\s+)?(training|webinar|masterclass|workshop|bootcamp)\b",
        r"\b(join|enroll|enrol)\s+(my|the|our)\b",
        r"\bcourse\b",
        r"\bmentorship|mentoring\b",
        r"\bsignal(s)?\s+(group|service)\b",
        r"\bdiscord\b",
        r"\b(link|links?)\s+(in|below|bio)\b",
        r"\bsign\s+up\b",
        r"\bget\s+in\s+my\s+team\b",
        r"\b(dm|message)\s+me\b",
        r"\baffiliate\b",
        r"\bpromo\s*code\b",
        r"\binner\s+circle\b",
    ],
    "PUMP_HYPE": [
        r"\bto\s+the\s+moon\b",
        r"\bmoon(ing|shot)?\b",
        r"\bexplode|exploding\b",
        r"\b(everyone|everybody)\s+is\s+buying\b",
        r"\bnext\s+(bitcoin|big\s+thing|100x|1000x)\b",
        r"\bbefore\s+it\s+(explodes|moons|takes\s+off|skyrockets)\b",
        r"\bgem\b",
        r"\bfomo\b",
        r"\bdon'?t\s+miss\s+out\b",
    ],
    "FAKE_CREDENTIAL": [
        r"\binsider\b",
        r"\bsecret(s)?\b",
        r"\bthey\s+don'?t\s+want\s+you\s+to\s+know\b",
        r"\bmillionaire\b",
        r"\bself[\s-]?made\b",
        r"\bexpert\b",
        r"\bwall\s+street\b",
        r"\bhedge\s+fund\b",
        r"\b(banned|hidden|forbidden)\s+(trick|secret|method)\b",
    ],
}


_RISKY_ASSET_RE = re.compile(
    r"\b(day[\s-]?trad\w*|forex|crypto\w*|bitcoin|altcoin|options?|leverage|margin|"
    r"futures?|stocks?\s+to\s+buy|penny\s+stocks?|cfd|short\s+squeeze)\b"
)
_RISK_TERMS_RE = re.compile(
    r"\b(risk\w*|lose|loss(es)?|losing|volatil\w*|caution|careful|not\s+financial\s+advice|"
    r"nfa|do\s+your\s+own\s+research|dyor|disclaimer|can\s+go\s+down)\b"
)

_COMPILED: Dict[str, List[re.Pattern]] = {
    marker: [re.compile(p) for p in patterns] for marker, patterns in LEXICONS.items()
}
RISK_MARKERS: List[str] = list(LEXICONS.keys()) + ["RISK_OMISSION"]


def lexicon_hits(raw_text: str) -> Dict[str, int]:
    """Count codebook-marker pattern hits in raw lowercased text (one int per marker)."""
    t = str(raw_text).lower()
    hits: Dict[str, int] = {}
    for marker, patterns in _COMPILED.items():
        hits[marker] = sum(len(p.findall(t)) for p in patterns)
    risky = bool(_RISKY_ASSET_RE.search(t))
    has_risk_terms = bool(_RISK_TERMS_RE.search(t))
    hits["RISK_OMISSION"] = int(risky and not has_risk_terms)
    return hits



_NUM_K_M_RE = re.compile(r"\d+\s*[km]\b", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"[£$€]")
_NUMBER_RE = re.compile(r"\d")
_EXCLAIM_RE = re.compile(r"!")
_QUESTION_RE = re.compile(r"\?")


def _allcaps_ratio(title: str) -> float:
    words = re.findall(r"[A-Za-z]{2,}", str(title))
    if not words:
        return 0.0
    caps = sum(1 for w in words if w.isupper())
    return caps / len(words)


def _emoji_count(text: str) -> int:
    return len(_EMOJI_RE.findall(str(text)))


def _safe_num(x) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0
# feature names
ENGINEERED_FEATURE_NAMES: List[str] = [
    "view_count",
    "like_count",
    "like_to_view",
    "log_views",
    "log_likes",
    "title_char_len",
    "title_word_len",
    "title_allcaps_ratio",
    "title_exclaim_count",
    "title_question_count",
    "title_has_currency",
    "title_has_number",
    "title_has_digit_km",
    "title_emoji_count",
    "title_vader_compound",
    "desc_char_len",
    "desc_word_len",
    "desc_is_empty",
    "desc_url_count",
] + [f"lex_{m}" for m in RISK_MARKERS]


def engineered_row(
    title: str | float | None,
    description: str | float | None,
    view_count=0,
    like_count=0,
) -> Dict[str, float]:
    """Compute the full engineered-feature dict for a single video."""
    title_s = "" if title is None or (isinstance(title, float) and np.isnan(title)) else str(title)
    desc_s = "" if description is None or (isinstance(description, float) and np.isnan(description)) else str(description)
    views = _safe_num(view_count)
    likes = _safe_num(like_count)
    combined_raw = f"{title_s} {desc_s}"

    feats: Dict[str, float] = {
        "view_count": views,
        "like_count": likes,
        "like_to_view": likes / (views + 1.0),
        "log_views": float(np.log1p(max(views, 0.0))),
        "log_likes": float(np.log1p(max(likes, 0.0))),
        "title_char_len": float(len(title_s)),
        "title_word_len": float(len(title_s.split())),
        "title_allcaps_ratio": _allcaps_ratio(title_s),
        "title_exclaim_count": float(len(_EXCLAIM_RE.findall(title_s))),
        "title_question_count": float(len(_QUESTION_RE.findall(title_s))),
        "title_has_currency": float(bool(_CURRENCY_RE.search(title_s))),
        "title_has_number": float(bool(_NUMBER_RE.search(title_s))),
        "title_has_digit_km": float(bool(_NUM_K_M_RE.search(title_s))),
        "title_emoji_count": float(_emoji_count(title_s)),
        "title_vader_compound": float(_VADER.polarity_scores(title_s)["compound"]),
        "desc_char_len": float(len(desc_s)),
        "desc_word_len": float(len(desc_s.split())),
        "desc_is_empty": float(len(desc_s.strip()) == 0),
        "desc_url_count": float(len(_URL_RE.findall(desc_s))),
    }
    for marker, count in lexicon_hits(combined_raw).items():
        feats[f"lex_{marker}"] = float(count)
    return feats



# sklearn transformers 

RAW_COLUMNS = ("title", "description", "view_count", "like_count")


class CleanTextExtractor(BaseEstimator, TransformerMixin):
    """DataFrame[title, description] -> 1-D array of cleaned combined strings (for TF-IDF)."""

    def fit(self, X, y=None): 
        return self

    def transform(self, X): 
        df = _as_frame(X)
        return df.apply(lambda r: combine(r.get("title"), r.get("description")), axis=1).to_numpy()


class EngineeredFeatures(BaseEstimator, TransformerMixin):
    """DataFrame -> dense numeric matrix of engineered + lexicon + VADER features."""

    feature_names_: List[str] = ENGINEERED_FEATURE_NAMES

    def fit(self, X, y=None): 
        return self

    def transform(self, X):
        df = _as_frame(X)
        rows = [
            engineered_row(
                r.get("title"), r.get("description"), r.get("view_count", 0), r.get("like_count", 0)
            )
            for _, r in df.iterrows()
        ]
        mat = np.array([[row[name] for name in ENGINEERED_FEATURE_NAMES] for row in rows], dtype=np.float64)
        return mat

    def get_feature_names_out(self, input_features=None):  
        return np.array(ENGINEERED_FEATURE_NAMES, dtype=object)


def _as_frame(X) -> pd.DataFrame:
    """Coerce arbitrary input into a DataFrame carrying the expected raw columns."""
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    elif isinstance(X, dict):
        df = pd.DataFrame([X])
    elif isinstance(X, Sequence) and len(X) and isinstance(X[0], dict):
        df = pd.DataFrame(list(X))
    else:
        raise TypeError(f"Unsupported feature input type: {type(X)!r}")
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = 0 if "count" in col else ""
    return df
