"""Model C - a genuinely fine-tuned DistilBERT classifier for Stage 2.

Honours the original proposal (DistilBERT/RoBERTa) rather than the frozen-embedding
fallback. Self-contained PyTorch training loop (no `accelerate`/`datasets` needed):

  * input text = ``title + " [SEP] " + description``, truncated to 256 tokens
  * class-weighted cross-entropy for imbalance
  * early stopping on validation macro-F1
  * everything seeded; GPU used if present, otherwise CPU with a loud time warning

Evaluation uses a single channel-disjoint train/val/test split (a fine-tuned LM is too
expensive for full grouped CV; this trade-off is documented in the results).
"""
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

import config


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(False)  
    except Exception:
        pass


def _build_texts(titles: List[str], descriptions: List[str], sep: str) -> List[str]:
    out = []
    for t, d in zip(titles, descriptions):
        t = "" if t is None else str(t)
        d = "" if d is None else str(d)
        out.append(f"{t} {sep} {d}".strip())
    return out


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class _TextDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        import torch

        item = {k: torch.tensor(v[i]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item


def _evaluate(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs.append(p)
            ys.append(labels.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def load_and_evaluate(X, y: np.ndarray, test_idx, model_dir: str) -> Dict:
    """Reload a previously fine-tuned Model C and evaluate it on the same test split.

    Lets us recover Model C metrics without re-running the (slow) CPU fine-tune, as long
    as the channel-disjoint split is reproduced identically (it is — seeded).
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from finclass.evaluation import compute_metrics

    _seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    sep = tok.sep_token or "[SEP]"
    texts = _build_texts(list(X["title"].astype(str)), list(X["description"].astype(str)), sep)
    enc = tok([texts[i] for i in test_idx], truncation=True, padding="max_length",
              max_length=config.TRANSFORMER_MAX_LEN)
    ds = _TextDataset(enc, [int(y[i]) for i in test_idx])
    loader = DataLoader(ds, batch_size=config.TRANSFORMER_BATCH)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    test_p, test_y = _evaluate(model, loader, device)
    metrics = compute_metrics(test_y, test_p, 0.5, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)
    metrics["protocol"] = "channel_disjoint_split"
    metrics["device"] = device.type
    metrics["n_test"] = len(test_idx)
    metrics["reloaded"] = True
    return {"metrics": metrics, "test_p": test_p, "test_y": test_y, "test_idx": np.asarray(test_idx)}


def train_and_evaluate(
    X, y: np.ndarray, train_idx, val_idx, test_idx, dates=None, save_dir: Optional[str] = None
) -> Dict:
    """Fine-tune DistilBERT and evaluate on the channel-disjoint test split.

    Returns a metrics dict including test predictions/probabilities so the caller can run
    a McNemar test against Model A on the identical test rows. If ``dates`` is given, a
    second (temporal) evaluation reuses the fine-tuned model's classifier over a date split.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from finclass.evaluation import compute_metrics

    _seed_everything(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("    [Model C] No GPU detected — fine-tuning DistilBERT on CPU. "
              "Expect several minutes per epoch.")

    titles = list(X["title"].astype(str))
    descs = list(X["description"].astype(str))
    tok = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL)
    sep = tok.sep_token or "[SEP]"
    texts = _build_texts(titles, descs, sep)

    def encode(idx):
        enc = tok([texts[i] for i in idx], truncation=True, padding="max_length",
                  max_length=config.TRANSFORMER_MAX_LEN)
        return _TextDataset(enc, [int(y[i]) for i in idx])

    train_ds, val_ds, test_ds = encode(train_idx), encode(val_idx), encode(test_idx)
    train_loader = DataLoader(train_ds, batch_size=config.TRANSFORMER_BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.TRANSFORMER_BATCH)
    test_loader = DataLoader(test_ds, batch_size=config.TRANSFORMER_BATCH)

    model = AutoModelForSequenceClassification.from_pretrained(config.TRANSFORMER_MODEL, num_labels=2)
    model.to(device)

    # Class weights for imbalance
    y_tr = y[train_idx]
    n_pos, n_neg = max(int(y_tr.sum()), 1), max(int((1 - y_tr).sum()), 1)
    w = torch.tensor([len(y_tr) / (2 * n_neg), len(y_tr) / (2 * n_pos)], dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    optim = torch.optim.AdamW(model.parameters(), lr=config.TRANSFORMER_LR,
                              weight_decay=config.TRANSFORMER_WEIGHT_DECAY)

    from sklearn.metrics import f1_score

    best_f1, best_state, best_epoch = -1.0, None, -1
    for epoch in range(config.TRANSFORMER_EPOCHS):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            logits = model(**batch).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        vp, vy = _evaluate(model, val_loader, device)
        vf1 = f1_score(vy, (vp >= 0.5).astype(int), average="macro", zero_division=0)
        print(f"    [Model C] epoch {epoch + 1}/{config.TRANSFORMER_EPOCHS}  val_macro_f1={vf1:.3f}")
        if vf1 > best_f1:
            best_f1, best_epoch = vf1, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Channel-disjoint test evaluation.
    test_p, test_y = _evaluate(model, test_loader, device)
    metrics = compute_metrics(test_y, test_p, 0.5, config.POSITIVE_CLASS, config.NEGATIVE_CLASS)
    metrics["protocol"] = "channel_disjoint_split"
    metrics["best_epoch"] = best_epoch
    metrics["device"] = device.type
    metrics["n_train"], metrics["n_val"], metrics["n_test"] = len(train_idx), len(val_idx), len(test_idx)

    result = {
        "metrics": metrics,
        "test_p": test_p,
        "test_y": test_y,
        "test_idx": np.asarray(test_idx),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)

    return result
