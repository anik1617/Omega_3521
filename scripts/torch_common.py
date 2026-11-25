"""Shared PyTorch helpers for the GPU models (feedforward with dropout, CNN, DistilBERT).

Every model here follows the same rules as the scikit-learn models:
train on the training split, pick the checkpoint by validation macro-F1, and
touch the test split once at the end. Results files use the format that
scripts/compare_models.py and app.py read.
"""

import copy
import os
import platform
import random
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score

LABELS = ("negative", "neutral", "positive")
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
SEED = 46


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_labels(series):
    return torch.tensor(series.map(LABEL_TO_ID).to_numpy(), dtype=torch.long)


def class_weights(y, mode):
    """Loss weights per class. "none" is plain cross-entropy.

    "sqrt" weights each class by sqrt(n / (k * count)), which raises the cost of
    missing a neutral review (7% of training data) without overcorrecting the way
    full inverse-frequency weights tend to, since those flood neutral with
    false positives from the positive class.
    """
    if mode == "none":
        return None
    counts = torch.bincount(y, minlength=len(LABELS)).float()
    weights = (counts.sum() / (len(LABELS) * counts)) ** 0.5
    return weights


def macro_f1(y_true, logits):
    pred = logits.argmax(dim=1).numpy()
    return f1_score(y_true.numpy(), pred, labels=range(len(LABELS)), average="macro", zero_division=0)


def fit_with_early_stopping(model, optimizer, train_batches, predict, y_train, y_val,
                            weights, max_epochs, patience, scheduler=None, log_prefix=""):
    """Train until validation macro-F1 stops improving for `patience` epochs.

    train_batches(epoch) yields (inputs, labels) on the device; predict(split)
    returns CPU logits for "train" or "val". Returns the best model (weights
    restored), a history DataFrame, the best epoch, and the seconds spent in the
    training loop (evaluation passes are not counted).
    """
    device = next(model.parameters()).device
    loss_weights = weights.to(device) if weights is not None else None
    best_state, best_score, best_epoch, bad_epochs = None, -1.0, 0, 0
    history, train_seconds = [], 0.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        total_loss, seen = 0.0, 0
        for inputs, labels in train_batches(epoch):
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels, weight=loss_weights)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item() * len(labels)
            seen += len(labels)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_seconds += perf_counter() - start

        model.eval()
        with torch.no_grad():
            val_logits = predict("val")
        row = {
            "epoch": epoch,
            "training_loss": total_loss / seen,
            "validation_loss": F.cross_entropy(val_logits, y_val).item(),
            "validation_macro_f1": macro_f1(y_val, val_logits),
        }
        history.append(row)
        print(f"{log_prefix}epoch {epoch:2}: train loss {row['training_loss']:.4f}, "
              f"val loss {row['validation_loss']:.4f}, val macro-F1 {row['validation_macro_f1']:.4f}",
              flush=True)

        if row["validation_macro_f1"] > best_score:
            best_score, best_epoch, bad_epochs = row["validation_macro_f1"], epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"{log_prefix}early stop: no gain for {patience} epochs", flush=True)
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch, train_seconds


def test_report(y_test, logits):
    pred = [LABELS[i] for i in logits.argmax(dim=1).tolist()]
    actual = [LABELS[i] for i in y_test.tolist()]
    report = classification_report(actual, pred, labels=list(LABELS), output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, pred, labels=list(LABELS))
    return report, matrix.tolist()


def machine_info():
    info = {
        "platform": platform.platform(),
        "processor": platform.machine(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
