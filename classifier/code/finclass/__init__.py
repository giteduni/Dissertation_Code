"""finclass - Misleading-vs-Educational YouTube finance-content classifier.

A two-stage text-only pipeline:
  * Stage 1 gates NON_FINANCIAL videos out of the corpus.
  * Stage 2 scores financial videos as Educational vs Misleading -> P(misleading).

Modules:
  text        cleaning, codebook risk lexicons, engineered linguistic features
  data        gold-label loading, Stage-1/Stage-2 view construction, splits
  pipeline    sklearn feature transformers + Model A / Model B builders
  evaluation  channel-grouped CV, temporal hold-out, metrics, plots
"""
__all__ = ["text", "data", "pipeline", "evaluation"]
