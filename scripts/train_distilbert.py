"""Fine-tune pre-trained DistilBERT (distilbert-base-uncased) with a classification head.

    python scripts/train_distilbert.py
    python scripts/train_distilbert.py --loss-weighting sqrt

Reviews are Summary + Text (the same `text` column every model uses), cut to
the first MAX_LENGTH WordPiece tokens. The whole network is fine-tuned with
AdamW, linear warmup and decay, and mixed precision on the GPU. After each
epoch the model is scored on validation; the epoch with the best validation
macro-F1 is kept and evaluated on the test split once.

Needs a CUDA GPU in practice (about 5 minutes per epoch on an RTX 3060 Laptop).
The first run downloads the pre-trained weights (about 260 MB) from Hugging Face.
The kept epoch is saved in float16 to models/distilbert_seed46/ for the app
(model.safetensors is stored with Git LFS); full-precision checkpoints go to
models/distilbert_seed46_checkpoint/ (gitignored). Results go to report/model_results/distilbert_seed46.json (or
distilbert_sqrt_weights_seed46.json with --loss-weighting sqrt).
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from data_utils import load_splits  # noqa: E402
from torch_common import (LABELS, SEED, class_weights, encode_labels, get_device, machine_info,  # noqa: E402
                          macro_f1, set_seed, test_report)

CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 256          # covers the whole review for about 90% of training reviews
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_SHARE = 0.1
OUT_DIR = ROOT / "report/milestone3/aniket"


def tokenize(tokenizer, texts):
    return tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH)["input_ids"]


def batches(token_ids, labels, order, batch_size, pad_id, device):
    """Pad each batch only to its longest review, which saves most of the compute."""
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        seqs = [token_ids[i] for i in rows]
        width = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
        for row, seq in enumerate(seqs):
            ids[row, :len(seq)] = torch.tensor(seq)
        mask = (ids != pad_id).long()
        yield ids.to(device), mask.to(device), labels[rows].to(device)


@torch.no_grad()
def predict(model, token_ids, labels, pad_id, device):
    model.eval()
    # Sorting by length keeps padding small; logits are put back in the original order.
    order = sorted(range(len(token_ids)), key=lambda i: len(token_ids[i]))
    logits = torch.empty(len(token_ids), len(LABELS))
    for start, (ids, mask, _) in zip(range(0, len(order), EVAL_BATCH_SIZE),
                                     batches(token_ids, labels, order, EVAL_BATCH_SIZE, pad_id, device)):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(input_ids=ids, attention_mask=mask).logits
        logits[order[start:start + EVAL_BATCH_SIZE]] = out.float().cpu()
    return logits


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.8), layout="constrained")
    ax.plot(history["epoch"], history["training_loss"], "o-", color="#2468a2", label="Training loss")
    ax.plot(history["epoch"], history["validation_loss"], "s-", color="#b45309", label="Validation loss")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy", xticks=history["epoch"],
           title="DistilBERT fine-tuning loss")
    f1 = ax.twinx()
    f1.plot(history["epoch"], history["validation_macro_f1"], "^--", color="#0f766e",
            label="Validation macro-F1")
    f1.set_ylabel("Validation macro-F1")
    lines = ax.get_lines() + f1.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc="center right")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--loss-weighting", choices=["none", "sqrt"], default="none")
    weighting = parser.parse_args().loss_weighting
    suffix = "" if weighting == "none" else "_sqrt_weights"

    set_seed()
    device = get_device()
    print(f"Device: {device}", flush=True)
    train, val, test = load_splits()
    labels = {name: encode_labels(df["Sentiment"]) for name, df in
              (("train", train), ("val", val), ("test", test))}
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    tokens = {name: tokenize(tokenizer, df["text"]) for name, df in
              (("train", train), ("val", val), ("test", test))}
    pad_id = tokenizer.pad_token_id
    truncated = sum(len(t) == MAX_LENGTH for t in tokens["train"]) / len(tokens["train"])
    print(f"Training reviews cut at {MAX_LENGTH} tokens: {truncated:.1%}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
    ).to(device)
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=LEARNING_RATE)
    steps_per_epoch = -(-len(train) // BATCH_SIZE)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_SHARE * EPOCHS * steps_per_epoch),
                                                EPOCHS * steps_per_epoch)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    weights = class_weights(labels["train"], weighting)
    weights = weights.to(device) if weights is not None else None

    # Full-precision checkpoints stay local; the app copy is saved in float16 at the end.
    checkpoint_dir = ROOT / f"models/distilbert{suffix}_seed46_checkpoint"
    save_dir = ROOT / f"models/distilbert{suffix}_seed46"
    history, train_seconds, best_score, best_epoch = [], 0.0, -1.0, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train), generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
        torch.cuda.synchronize() if device.type == "cuda" else None
        start = perf_counter()
        total, seen = 0.0, 0
        for step, (ids, mask, y) in enumerate(batches(tokens["train"], labels["train"], order,
                                                      BATCH_SIZE, pad_id, device), 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = F.cross_entropy(logits.float(), y, weight=weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(y)
            seen += len(y)
            if step % 300 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch}: loss {total / seen:.4f}", flush=True)
        torch.cuda.synchronize() if device.type == "cuda" else None
        epoch_seconds = perf_counter() - start
        train_seconds += epoch_seconds

        val_logits = predict(model, tokens["val"], labels["val"], pad_id, device)
        row = {"epoch": epoch, "training_loss": total / seen,
               "validation_loss": F.cross_entropy(val_logits, labels["val"]).item(),
               "validation_macro_f1": macro_f1(labels["val"], val_logits),
               "epoch_seconds": round(epoch_seconds, 1)}
        history.append(row)
        print(f"Epoch {epoch}/{EPOCHS}: train loss {row['training_loss']:.4f}, val loss "
              f"{row['validation_loss']:.4f}, val macro-F1 {row['validation_macro_f1']:.4f} "
              f"({epoch_seconds:.0f} s)", flush=True)
        if row["validation_macro_f1"] > best_score:
            best_score, best_epoch = row["validation_macro_f1"], epoch
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

    # Save the kept epoch in float16 (about 134 MB, stored with Git LFS) for the app,
    # then reload that copy and touch the test split once, so the reported scores are
    # the ones the app gets.
    AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).half().save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir, dtype=torch.float32).to(device)
    report, matrix = test_report(labels["test"], predict(model, tokens["test"], labels["test"], pad_id, device))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(OUT_DIR / f"history_distilbert{suffix}.csv", index=False)
    plot_history(pd.DataFrame(history), OUT_DIR / f"loss_distilbert{suffix}.png")
    results = {
        "model": "DistilBERT fine-tuned" + (", class-weighted loss" if suffix else ""),
        "owner": "Aniket Chaudhari",
        "split": "data/splits/split_seed46.csv",
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "labels": list(LABELS),
        "confusion_matrix_axes": "rows=actual, columns=predicted",
        "classifier_settings": {
            "checkpoint": CHECKPOINT, "head": "linear classification head on the [CLS] token",
            "fine_tuned_layers": "all", "max_length_tokens": MAX_LENGTH,
            "share_of_training_reviews_truncated": round(truncated, 4),
            "batch_size": BATCH_SIZE, "epochs": EPOCHS, "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "schedule": f"linear warmup over {WARMUP_SHARE:.0%} of steps, then linear decay",
            "mixed_precision": "float16 autocast", "loss_weighting": weighting, "random_seed": SEED,
        },
        "selected": {"epoch": best_epoch, "validation_macro_f1": best_score},
        "test_models": {
            "distilbert_tokens": {
                "model_path": save_dir.relative_to(ROOT).as_posix(),
                "validation_macro_f1": best_score,
                "training_time_seconds": round(train_seconds, 1),
                "selected_epoch": best_epoch,
                "classification_report": report,
                "confusion_matrix": matrix,
            }
        },
        "training_time_scope": (f"GPU time in the fine-tuning loop over all {EPOCHS} epochs; excludes "
                                "tokenization and validation passes"),
        "machine": machine_info(),
    }
    (ROOT / f"report/model_results/distilbert{suffix}_seed46.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Kept epoch {best_epoch} (val macro-F1 {best_score:.4f}). Training time {train_seconds:.0f} s")
    print(f"Test accuracy {report['accuracy']:.4f}, macro-F1 {report['macro avg']['f1-score']:.4f}, "
          f"neutral F1 {report['neutral']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
