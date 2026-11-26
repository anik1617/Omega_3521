"""Train the cheaper PyTorch models from lecture: a feedforward network with
dropout and early stopping, a 1D CNN over word embeddings, and an averaged
word-embedding model.

    python scripts/train_torch_models.py                 # all three
    python scripts/train_torch_models.py --models cnn    # just one

The feedforward network uses the shared Bag of Words and TF-IDF vectorizers
(models/vectorizer_*_seed46.joblib), so it is directly comparable to the
scikit-learn MLP. The CNN and embedding models learn 128-dim word embeddings
from scratch on the training split (no pretrained vectors).

For each model, plain and class-weighted cross-entropy are both tried and the
one with the higher validation macro-F1 is kept. Training stops when validation
macro-F1 has not improved for PATIENCE epochs, and the best epoch's weights are
restored before the single test-set evaluation.

Weights are saved under models/torch/ (gitignored: the TF-IDF network alone is
about 50 MB). Results go to report/model_results/ and training histories to
report/milestone3/aniket/.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from data_utils import clean_text, load_splits  # noqa: E402
from torch_common import (LABELS, SEED, class_weights, encode_labels, fit_with_early_stopping,  # noqa: E402
                          get_device, machine_info, set_seed, test_report)

MAX_EPOCHS = 30
PATIENCE = 3
WEIGHTINGS = ("none", "sqrt")
OUT_DIR = ROOT / "report/milestone3/aniket"
WEIGHTS_DIR = ROOT / "models/torch"
RESULTS_DIR = ROOT / "report/model_results"

# Feedforward network: same 64-32 hidden layers the charter planned.
MLP_HIDDEN = (64, 32)
MLP_DROPOUT = 0.5
MLP_BATCH = 256
MLP_LR = 1e-3
VECTORIZERS = {
    "tfidf": ROOT / "models/vectorizer_tfidf_seed46.joblib",
    "bag_of_words": ROOT / "models/vectorizer_bow_seed46.joblib",
}

# Word-embedding models.
EMBED_DIM = 128
MIN_COUNT = 2
MAX_TOKENS = 300          # 95% of training reviews have at most 227 words
CNN_FILTERS = 100
CNN_KERNELS = (3, 4, 5)
SEQ_DROPOUT = 0.5
SEQ_BATCH = 64
SEQ_LR = 1e-3

TOKEN = re.compile(r"[a-z0-9']+")


# ---------------------------------------------------------------- models

class SparseMLP(nn.Module):
    """Feedforward network on a sparse BoW/TF-IDF matrix.

    The first layer is a sparse-dense matrix product, so a 190k-term vocabulary
    never has to be expanded into a dense batch.
    """

    def __init__(self, n_features, hidden=MLP_HIDDEN, dropout=MLP_DROPOUT):
        super().__init__()
        self.first = nn.Parameter(torch.empty(n_features, hidden[0]))
        nn.init.kaiming_uniform_(self.first.T, a=5 ** 0.5)
        self.first_bias = nn.Parameter(torch.zeros(hidden[0]))
        self.rest = nn.Sequential(
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], len(LABELS)),
        )

    def forward(self, x):
        return self.rest(torch.sparse.mm(x, self.first) + self.first_bias)


class TextCNN(nn.Module):
    """Kim-style CNN: embeddings, parallel 1D convolutions, max-over-time pooling."""

    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(EMBED_DIM, CNN_FILTERS, k) for k in CNN_KERNELS)
        self.dropout = nn.Dropout(SEQ_DROPOUT)
        self.out = nn.Linear(CNN_FILTERS * len(CNN_KERNELS), len(LABELS))

    def forward(self, ids):
        x = self.embed(ids).transpose(1, 2)              # (batch, dim, tokens)
        pooled = [torch.relu(conv(x)).amax(dim=2) for conv in self.convs]
        return self.out(self.dropout(torch.cat(pooled, dim=1)))


class MeanEmbedding(nn.Module):
    """Average the word embeddings of a review, then one hidden layer."""

    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.head = nn.Sequential(nn.Dropout(SEQ_DROPOUT), nn.Linear(EMBED_DIM, 64), nn.ReLU(),
                                  nn.Dropout(SEQ_DROPOUT), nn.Linear(64, len(LABELS)))

    def forward(self, ids):
        mask = (ids != 0).unsqueeze(2).float()
        summed = (self.embed(ids) * mask).sum(dim=1)
        return self.head(summed / mask.sum(dim=1).clamp(min=1))


# ---------------------------------------------------------------- data

def csr_batch(matrix, rows, device):
    coo = matrix[rows].tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()


def build_vocab(texts):
    counts = Counter(tok for text in texts for tok in TOKEN.findall(clean_text(text)))
    words = sorted(w for w, c in counts.items() if c >= MIN_COUNT)
    return {w: i + 2 for i, w in enumerate(words)}           # 0 = padding, 1 = unknown


def encode_texts(texts, vocab):
    ids = torch.zeros(len(texts), MAX_TOKENS, dtype=torch.long)
    for row, text in enumerate(texts):
        tokens = [vocab.get(t, 1) for t in TOKEN.findall(clean_text(text))[:MAX_TOKENS]]
        if tokens:
            ids[row, :len(tokens)] = torch.tensor(tokens)
    return ids


def batch_order(n, epoch, batch_size):
    gen = torch.Generator().manual_seed(SEED * 1000 + epoch)
    order = torch.randperm(n, generator=gen)
    return [order[i:i + batch_size] for i in range(0, n, batch_size)]


# ---------------------------------------------------------------- training

def run_mlp(splits, labels, device):
    out = {}
    for features, vec_path in VECTORIZERS.items():
        vectorizer = joblib.load(vec_path)
        mats = {name: vectorizer.transform(df["text"]).tocsr().astype(np.float32)
                for name, df in splits.items()}

        def make_model(mats=mats):
            return SparseMLP(mats["train"].shape[1])

        def train_batches(epoch, mats=mats):
            for rows in batch_order(mats["train"].shape[0], epoch, MLP_BATCH):
                rows = rows.numpy()
                yield csr_batch(mats["train"], rows, device), labels["train"][rows].to(device)

        def predict_fn(model, mats=mats):
            def predict(split):
                m = mats[split]
                chunks = [model(csr_batch(m, np.arange(i, min(i + 2048, m.shape[0])), device)).cpu()
                          for i in range(0, m.shape[0], 2048)]
                return torch.cat(chunks)
            return predict

        out[features] = search_and_test(f"mlp-{features}", make_model, train_batches, predict_fn,
                                        labels, device, MLP_LR, {"vocabulary_size": mats["train"].shape[1]})
    return out


def run_sequence(kind, splits, labels, device, vocab, ids):
    cls = {"cnn": TextCNN, "mean_embedding": MeanEmbedding}[kind]

    def make_model():
        return cls(len(vocab) + 2)

    def train_batches(epoch):
        for rows in batch_order(len(ids["train"]), epoch, SEQ_BATCH):
            yield ids["train"][rows].to(device), labels["train"][rows].to(device)

    def predict_fn(model):
        def predict(split):
            x = ids[split]
            return torch.cat([model(x[i:i + 512].to(device)).cpu() for i in range(0, len(x), 512)])
        return predict

    return {"word_embeddings": search_and_test(kind, make_model, train_batches, predict_fn, labels,
                                               device, SEQ_LR, {"vocabulary_size": len(vocab) + 2})}


def search_and_test(tag, make_model, train_batches, predict_fn, labels, device, lr, extra):
    """Try each loss weighting, keep the best on validation, then score on test once."""
    runs = []
    for weighting in WEIGHTINGS:
        set_seed()
        model = make_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model, history, best_epoch, seconds = fit_with_early_stopping(
            model, optimizer, train_batches, predict_fn(model), labels["train"], labels["val"],
            class_weights(labels["train"], weighting), MAX_EPOCHS, PATIENCE,
            log_prefix=f"[{tag} / weights={weighting}] ")
        history.insert(0, "loss_weighting", weighting)
        score = history.loc[history["epoch"] == best_epoch, "validation_macro_f1"].iloc[0]
        runs.append(dict(weighting=weighting, model=model, history=history, epoch=best_epoch,
                         seconds=seconds, val_f1=float(score)))

    best = max(runs, key=lambda r: r["val_f1"])
    model = best["model"].eval()
    with torch.no_grad():
        report, matrix = test_report(labels["test"], predict_fn(model)("test"))

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = WEIGHTS_DIR / f"{tag}_seed46.pt"
    torch.save(model.state_dict(), weights_path)
    pd.concat([r["history"] for r in runs]).to_csv(OUT_DIR / f"history_{tag}.csv", index=False)
    print(f"[{tag}] kept weights={best['weighting']} epoch {best['epoch']} "
          f"(val macro-F1 {best['val_f1']:.4f}); test acc {report['accuracy']:.4f}, "
          f"macro-F1 {report['macro avg']['f1-score']:.4f}, neutral F1 {report['neutral']['f1-score']:.4f}",
          flush=True)
    return {
        "model_path": weights_path.relative_to(ROOT).as_posix(),
        "validation_macro_f1": best["val_f1"],
        "training_time_seconds": round(best["seconds"], 3),
        "selected_epoch": best["epoch"],
        "epochs_run": int(best["history"]["epoch"].max()),
        "loss_weighting": best["weighting"],
        "validation_macro_f1_by_weighting": {r["weighting"]: r["val_f1"] for r in runs},
        **extra,
        "classification_report": report,
        "confusion_matrix": matrix,
        "_history": best["history"],
    }


def write_results(path, model_name, settings, test_models, rows):
    histories = {f: m.pop("_history") for f, m in test_models.items()}
    results = {
        "model": model_name,
        "owner": "Aniket Chaudhari",
        "split": "data/splits/split_seed46.csv",
        "rows": rows,
        "labels": list(LABELS),
        "confusion_matrix_axes": "rows=actual, columns=predicted",
        "classifier_settings": settings,
        "test_models": test_models,
        "training_time_scope": ("GPU time in the training loop for the kept loss weighting, over every "
                                "epoch run including the ones after the best epoch; excludes "
                                "validation passes and the weighting search"),
        "machine": machine_info(),
    }
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return histories


def plot_histories(histories, path):
    fig, axes = plt.subplots(1, len(histories), figsize=(4.6 * len(histories), 3.6),
                             squeeze=False, layout="constrained")
    for ax, (name, history) in zip(axes.flat, histories.items()):
        ax.plot(history["epoch"], history["training_loss"], "o-", ms=3, color="#2468a2", label="Training loss")
        ax.plot(history["epoch"], history["validation_loss"], "s-", ms=3, color="#b45309",
                label="Validation loss")
        best = history.loc[history["validation_macro_f1"].idxmax(), "epoch"]
        ax.axvline(best, color="#666666", ls="--", lw=1, label=f"Kept epoch ({best})")
        ax.set(title=name, xlabel="Epoch", ylabel="Cross-entropy")
        ax.grid(axis="y", alpha=0.25)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=["mlp", "cnn", "mean_embedding"],
                        choices=["mlp", "cnn", "mean_embedding"])
    chosen = parser.parse_args().models

    device = get_device()
    print(f"Device: {device}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = load_splits()
    splits = {"train": train, "val": val, "test": test}
    labels = {name: encode_labels(df["Sentiment"]) for name, df in splits.items()}
    rows = {"train": len(train), "validation": len(val), "test": len(test)}
    common = {"optimizer": "Adam", "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
              "early_stopping_metric": "validation macro-F1", "loss_weightings_tried": list(WEIGHTINGS),
              "random_seed": SEED, "framework": f"PyTorch {torch.__version__}"}

    if "mlp" in chosen:
        test_models = run_mlp(splits, labels, device)
        hist = write_results(
            RESULTS_DIR / "torch_mlp_dropout_seed46.json", "Feedforward neural network, dropout",
            {**common, "hidden_layers": list(MLP_HIDDEN), "activation": "relu", "dropout": MLP_DROPOUT,
             "batch_size": MLP_BATCH, "learning_rate": MLP_LR,
             "inputs": "shared vectorizers in models/vectorizer_*_seed46.joblib"},
            test_models, rows)
        plot_histories({f"Dropout MLP ({'TF-IDF' if f == 'tfidf' else 'BoW'})": h for f, h in hist.items()},
                       OUT_DIR / "loss_mlp_dropout.png")

    seq = [m for m in chosen if m != "mlp"]
    if seq:
        vocab = build_vocab(train["text"])
        ids = {name: encode_texts(df["text"].tolist(), vocab) for name, df in splits.items()}
        print(f"Word vocabulary: {len(vocab):,} words (min count {MIN_COUNT})", flush=True)
        seq_settings = {**common, "embedding_dim": EMBED_DIM, "embeddings": "learned from scratch",
                        "min_word_count": MIN_COUNT, "max_tokens": MAX_TOKENS, "dropout": SEQ_DROPOUT,
                        "batch_size": SEQ_BATCH, "learning_rate": SEQ_LR}
        names = {"cnn": ("1D CNN", "torch_cnn_seed46.json",
                         {"conv_kernel_sizes": list(CNN_KERNELS), "filters_per_kernel": CNN_FILTERS,
                          "pooling": "max over time"}),
                 "mean_embedding": ("Averaged word embeddings", "torch_mean_embedding_seed46.json",
                                    {"hidden_layer": 64, "pooling": "mean over words"})}
        for kind in seq:
            name, filename, extra = names[kind]
            test_models = run_sequence(kind, splits, labels, device, vocab, ids)
            hist = write_results(RESULTS_DIR / filename, name, {**seq_settings, **extra}, test_models, rows)
            plot_histories({name: h for h in hist.values()}, OUT_DIR / f"loss_{kind}.png")


if __name__ == "__main__":
    main()
