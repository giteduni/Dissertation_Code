"""Central configuration for the finance-content classifier.

Every tunable lives here so that ``train.py``, ``predict.py`` and the web app share
exactly one source of truth. Nothing in this module imports heavy ML libraries, so it
is cheap to import from anywhere (including the Streamlit app at startup).
"""
from __future__ import annotations

import os
from pathlib import Path


# Paths
CODE_DIR: Path = Path(__file__).resolve().parent
PROJECT_DIR: Path = CODE_DIR.parent 

DATA_CSV: Path = PROJECT_DIR / "labelling" / "annotation_sample_verified_labelled_revised.csv"

MODELS_DIR: Path = CODE_DIR / "models"
METRICS_DIR: Path = CODE_DIR / "metrics"
CACHE_DIR: Path = CODE_DIR / ".cache" 

for _d in (MODELS_DIR, METRICS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Reproducibility

SEED: int = 42
FINANCIAL_LABELS = ("Educational", "Misleading", "BORDERLINE")
NON_FINANCIAL_LABEL = "NON_FINANCIAL"
BORDERLINE_AS: str = os.environ.get("BORDERLINE_AS", "exclude").lower()
assert BORDERLINE_AS in {"exclude", "misleading", "educational"}, BORDERLINE_AS
POSITIVE_CLASS = "Misleading"
NEGATIVE_CLASS = "Educational"


# Decision thresholds 
STAGE1_NONFIN_THRESHOLD: float = 0.50 
STAGE2_MISLEADING_THRESHOLD: float = 0.50 


# Evaluation
N_SPLITS: int = 5  
TEMPORAL_CUTOFF: str = "2018-01-01"  
UPLOAD_DATE_FORMAT: str = "%d/%m/%Y %H:%M"


# Feature pipeline
WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)
WORD_MIN_DF = 2
CHAR_MIN_DF = 5
WORD_MAX_FEATURES = 20000
CHAR_MAX_FEATURES = 20000
CHAR_NGRAMS_ENABLED = False


# Probability calibration
CALIBRATION_METHOD = "auto"
CALIBRATION_ISOTONIC_MIN_N = 1000
CALIBRATION_INTERNAL_CV = 3

DATA_OUT_DIR: Path = CODE_DIR / "data"
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)
DEMO_HOLDOUT_FRAC: float = 0.15  
DEMO_HOLDOUT_CSV = DATA_OUT_DIR / "demo_holdout.csv"
DEMO_HOLDOUT_GOLD_CSV = DATA_OUT_DIR / "demo_holdout_gold.csv"
TRAIN_FINGERPRINTS_FILE = "train_fingerprints.json"
OVERLAP_WARN_FRAC: float = 0.50  

# Transformer Model C 
TRANSFORMER_MODEL: str = "distilbert-base-uncased"
TRANSFORMER_MAX_LEN: int = 256
TRANSFORMER_EPOCHS: int = 3
TRANSFORMER_LR: float = 2e-5
TRANSFORMER_BATCH: int = 16
TRANSFORMER_WEIGHT_DECAY: float = 0.01
TRANSFORMER_DIR = MODELS_DIR / "model_c_distilbert"
TRAIN_TRANSFORMER: str = os.environ.get("TRAIN_TRANSFORMER", "auto").lower()
# Model B
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
PRIMARY_STAGE2_MODEL: str = os.environ.get("PRIMARY_STAGE2_MODEL", "model_a").lower()
# Serialised artifact filenames
STAGE1_MODEL_FILE = "stage1_financial_gate.joblib"
STAGE2_MODEL_A_FILE = "stage2_model_a_xgb.joblib"            
STAGE2_MODEL_A_CAL_FILE = "stage2_model_a_calibrated.joblib"  
STAGE2_MODEL_B_FILE = "stage2_model_b_embed.joblib"
STAGE2_MODEL_B_CAL_FILE = "stage2_model_b_calibrated.joblib"
METADATA_FILE = "model_metadata.json"
