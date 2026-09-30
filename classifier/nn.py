"""The multi-head network over sparse count/TF-IDF input.

The first layer is an EmbeddingBag in sum mode with per_sample_weights,
which is exactly x @ W + b evaluated over the non-zeros of x. Batches
are never densified: a batch carries (indices, offsets, values) and is
kilobytes rather than batch x n_features floats. One output head per
hierarchy level; the loss is the (optionally weighted) sum across heads.

Training keeps the best epoch on disk (checkpointing) and restores it at
the end, so a model whose validation score declined before early
stopping halted it is scored with its best weights, not its last.

torch is imported only here, so the NB model runs without it.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

IGNORE = -100          # target for a label unseen in training


@dataclass
class NNSettings:
    # architecture
    hidden_dims: Tuple[int, ...] = (512, 256)
    dropout: float = 0.2
    batch_norm: bool = True
    feature_dropout: float = 0.0          # drops input features during training
    # optimisation
    batch_size: int = 512
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    label_smoothing: float = 0.0
    level_loss_weights: Optional[Tuple[float, ...]] = None
    sparse_grad: bool = True              # SparseAdam for the first layer; much faster
    lr_scheduler: bool = True             # ReduceLROnPlateau on validation loss
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    # early stopping and checkpointing (need a validation set)
    early_stopping: bool = True
    patience: int = 6
    monitor: str = "accuracy"             # "accuracy" (of monitor_level) | "val_loss"
    monitor_level: int = -1               # -1 = deepest level
    checkpoint: bool = True               # keep the best epoch on disk and restore it
    # hardware
    prefer_gpu: bool = True               # use CUDA when available
    num_workers: int = 0
    eval_batch_size: int = 4096
    seed: int = 42

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NNSettings":
        d = dict(d)
        d["hidden_dims"] = tuple(d["hidden_dims"])
        if d.get("level_loss_weights") is not None:
            d["level_loss_weights"] = tuple(d["level_loss_weights"])
        return cls(**d)


def get_device(prefer_gpu: bool = True) -> str:
    return "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SparseLinear(nn.Module):
    """x @ W + b over the non-zeros of a sparse x, via EmbeddingBag."""

    def __init__(self, in_features: int, out_features: int, sparse_grad: bool = False,
                 feature_dropout: float = 0.0):
        super().__init__()
        self.bag = nn.EmbeddingBag(in_features, out_features, mode="sum", sparse=sparse_grad)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.sparse_grad, self.feature_dropout = sparse_grad, feature_dropout

    def forward(self, indices, offsets, values):
        if self.training and self.feature_dropout > 0:
            keep = (torch.rand_like(values) >= self.feature_dropout).to(values.dtype)
            values = values * keep / (1.0 - self.feature_dropout)
        return self.bag(indices, offsets, per_sample_weights=values) + self.bias


class SparseInputMLP(nn.Module):
    """Sparse first layer, dense trunk, one linear head per level."""

    def __init__(self, input_dim: int, n_classes_per_level: Sequence[int],
                 hidden_dims: Sequence[int] = (512, 256), dropout: float = 0.2,
                 batch_norm: bool = True, sparse_grad: bool = False, feature_dropout: float = 0.0):
        super().__init__()
        if not hidden_dims:
            raise ValueError("at least one hidden layer is needed; the sparse layer is the first")
        self.init_kwargs = dict(input_dim=int(input_dim),
                                n_classes_per_level=[int(n) for n in n_classes_per_level],
                                hidden_dims=[int(h) for h in hidden_dims], dropout=float(dropout),
                                batch_norm=bool(batch_norm), sparse_grad=bool(sparse_grad),
                                feature_dropout=float(feature_dropout))

        def block(dim):
            return [nn.BatchNorm1d(dim)] if batch_norm else []

        self.first = SparseLinear(input_dim, hidden_dims[0], sparse_grad, feature_dropout)
        layers = block(hidden_dims[0]) + [nn.ReLU(), nn.Dropout(dropout)]
        prev = hidden_dims[0]
        for dim in hidden_dims[1:]:
            layers += [nn.Linear(prev, dim)] + block(dim) + [nn.ReLU(), nn.Dropout(dropout)]
            prev = dim
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(prev, n) for n in n_classes_per_level])

    def forward(self, batch) -> List[torch.Tensor]:
        z = self.trunk(self.first(*batch))
        return [head(z) for head in self.heads]

    def sparse_parameters(self) -> List[nn.Parameter]:
        return [self.first.bag.weight] if self.first.sparse_grad else []


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class _RowIndex(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


class _CSRCollate:
    """Slice rows of a CSR matrix into EmbeddingBag's (indices, offsets, values)."""

    def __init__(self, X: sparse.csr_matrix, y: Optional[np.ndarray]):
        self.X, self.y = X, y

    def __call__(self, rows):
        rows = np.asarray(rows)
        block = self.X[rows]
        x = (torch.from_numpy(block.indices.astype(np.int64)),
             torch.from_numpy(block.indptr[:-1].astype(np.int64)),
             torch.from_numpy(block.data.astype(np.float32)))
        if self.y is None:
            return x, None
        return x, torch.from_numpy(self.y[rows])


def make_loader(X, y, batch_size: int, shuffle: bool, num_workers: int = 0,
                drop_last: bool = False, pin_memory: bool = False) -> DataLoader:
    X = sparse.csr_matrix(X, dtype=np.float32)
    if y is not None:
        y = np.where(np.asarray(y) < 0, IGNORE, np.asarray(y)).astype(np.int64)
    return DataLoader(_RowIndex(X.shape[0]), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=_CSRCollate(X, y), num_workers=num_workers,
                      persistent_workers=num_workers > 0, drop_last=drop_last,
                      pin_memory=pin_memory)


def _to(x, device):
    return tuple(t.to(device, non_blocking=True) for t in x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class _Optimizers:
    """AdamW for dense parameters plus SparseAdam for the sparse first layer."""

    def __init__(self, model: SparseInputMLP, s: NNSettings):
        sparse_params = model.sparse_parameters()
        ids = {id(p) for p in sparse_params}
        dense_params = [p for p in model.parameters() if id(p) not in ids]
        self.dense = torch.optim.AdamW(dense_params, lr=s.lr, weight_decay=s.weight_decay)
        self.sparse = torch.optim.SparseAdam(sparse_params, lr=s.lr) if sparse_params else None

    def zero_grad(self):
        self.dense.zero_grad(set_to_none=True)
        if self.sparse:
            self.sparse.zero_grad(set_to_none=True)

    def step(self):
        self.dense.step()
        if self.sparse:
            self.sparse.step()

    def sync_lr(self):
        if self.sparse:
            for g in self.sparse.param_groups:
                g["lr"] = self.dense.param_groups[0]["lr"]

    @property
    def lr(self):
        return self.dense.param_groups[0]["lr"]


def _run_epoch(model, loader, criterion, device, weights, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_rows, correct, counted = 0.0, 0, None, None
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = _to(x, device), y.to(device, non_blocking=True)
            logits = model(x)
            loss = sum(criterion(lg, y[:, d]) * (weights[d] if weights else 1.0)
                       for d, lg in enumerate(logits))
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total_rows += len(y)
            valid = y != IGNORE
            hits = np.array([int(((lg.argmax(1) == y[:, d]) & valid[:, d]).sum()) for d, lg in enumerate(logits)])
            n = np.array([int(valid[:, d].sum()) for d in range(len(logits))])
            correct = hits if correct is None else correct + hits
            counted = n if counted is None else counted + n
    return total_loss / max(total_rows, 1), (correct / np.maximum(counted, 1)).tolist()


def fit(model: SparseInputMLP, X_train, y_train, s: NNSettings, X_val=None, y_val=None,
        checkpoint_dir: Optional[str] = None, level_names: Optional[Sequence[str]] = None,
        verbose: bool = True) -> pd.DataFrame:
    """Train; returns the per-epoch history.

    With validation data: early stopping (if on), LR scheduling on
    validation loss (if on), and the best epoch by `s.monitor` restored
    at the end. With s.checkpoint and a checkpoint_dir, the best weights
    are written to <checkpoint_dir>/best.pt whenever they improve, along
    with history.csv, so a crash mid-run still leaves the best model on
    disk. Without validation data it runs exactly s.epochs epochs.
    """
    device = get_device(s.prefer_gpu)
    set_seed(s.seed)
    model.to(device)
    has_val = X_val is not None
    pin = device == "cuda"
    # BatchNorm cannot train on a batch of one, so a trailing short batch is dropped
    drop_last = s.batch_norm and X_train.shape[0] > s.batch_size
    train_loader = make_loader(X_train, y_train, s.batch_size, True, s.num_workers,
                               drop_last=drop_last, pin_memory=pin)
    val_loader = (make_loader(X_val, y_val, s.eval_batch_size, False, s.num_workers, pin_memory=pin)
                  if has_val else None)

    criterion = nn.CrossEntropyLoss(label_smoothing=s.label_smoothing, ignore_index=IGNORE)
    opt = _Optimizers(model, s)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt.dense, mode="min", factor=s.scheduler_factor, patience=s.scheduler_patience)
        if (s.lr_scheduler and has_val) else None)
    weights = list(s.level_loss_weights) if s.level_loss_weights else None
    names = list(level_names) if level_names else [f"L{d + 1}" for d in range(len(model.heads))]

    ckpt_path = None
    if s.checkpoint and checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, "best.pt")
    best_state, best_score, best_epoch, bad = None, None, None, 0
    rows = []
    if verbose:
        print(f"[nn] device={device}, {X_train.shape[0]:,} training rows, "
              f"{X_val.shape[0] if has_val else 0:,} validation rows")

    for epoch in range(1, s.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device, weights, opt)
        row = {"epoch": epoch, "train_loss": tr_loss, "lr": opt.lr}
        row.update({f"train_acc {n}": a for n, a in zip(names, tr_acc)})
        improved = False
        if has_val:
            va_loss, va_acc = _run_epoch(model, val_loader, criterion, device, weights)
            row["val_loss"] = va_loss
            row.update({f"val_acc {n}": a for n, a in zip(names, va_acc)})
            score = -va_loss if s.monitor == "val_loss" else va_acc[s.monitor_level]
            improved = best_score is None or score > best_score + 1e-6
            if improved:
                best_score, best_epoch, bad = score, epoch, 0
                if ckpt_path:
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            else:
                bad += 1
            if scheduler is not None:
                scheduler.step(va_loss)
                opt.sync_lr()
        row["seconds"] = time.time() - t0
        row["best"] = improved
        rows.append(row)
        if ckpt_path:
            pd.DataFrame(rows).to_csv(os.path.join(checkpoint_dir, "history.csv"), index=False)
        if verbose:
            acc = "  ".join(f"{n}={row[f'val_acc {n}']:.3f}" for n in names) if has_val else \
                  "  ".join(f"{n}={row[f'train_acc {n}']:.3f}(train)" for n in names)
            print(f"[nn] epoch {epoch:>3}  train_loss={tr_loss:.4f}"
                  + (f"  val_loss={row['val_loss']:.4f}" if has_val else "")
                  + f"  {acc}  lr={opt.lr:.1e}  {row['seconds']:.0f}s" + ("  *" if improved else ""))
        if has_val and s.early_stopping and bad >= s.patience:
            if verbose:
                print(f"[nn] early stop at epoch {epoch}; best epoch {best_epoch}")
            break

    if has_val and best_epoch is not None:
        state = torch.load(ckpt_path, map_location=device) if ckpt_path else best_state
        model.load_state_dict(state)
        if verbose:
            print(f"[nn] restored epoch {best_epoch} (best {s.monitor}: "
                  f"{abs(best_score):.4f})")
    history = pd.DataFrame(rows)
    history.attrs["best_epoch"] = best_epoch if has_val else len(rows)
    return history


@torch.no_grad()
def predict_proba(model: SparseInputMLP, X, batch_size: int = 4096,
                  prefer_gpu: bool = True) -> List[np.ndarray]:
    """Per-level softmax probabilities, float32."""
    device = get_device(prefer_gpu)
    model.to(device).eval()
    loader = make_loader(X, None, batch_size, False)
    parts = [[] for _ in model.heads]
    for x, _ in loader:
        for d, lg in enumerate(model(_to(x, device))):
            parts[d].append(torch.softmax(lg.float(), dim=1).cpu().numpy())
    return [np.concatenate(p) if p else np.zeros((0, h.out_features), np.float32)
            for p, h in zip(parts, model.heads)]
