"""Train the Milestone 3 feedforward network on the shared TF-IDF features."""

import copy
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import issparse
from sklearn.metrics import classification_report, confusion_matrix, f1_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import load_splits  # noqa: E402

LABELS = ("negative", "neutral", "positive")
SEED = 46
EPOCHS = 18
HIDDEN_LAYERS = (32, 16)
BATCH_SIZE = 1024
VECTORIZER_PATH = ROOT / "models/vectorizer_tfidf_seed46.joblib"
MODEL_PATH = ROOT / "models/neural_network_tfidf_seed46.joblib"
RESULTS_PATH = ROOT / "report/model_results/neural_network_seed46.json"
OUT_DIR = ROOT / "report/milestone3/ethan"


def train_network(x_train, y_train, x_val, y_val):
    # A seeded RNG object lets the shuffle advance across partial_fit calls.
    network = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        batch_size=BATCH_SIZE,
        random_state=np.random.RandomState(SEED),
    )
    history = []
    best_network = None
    best_score = -1.0
    fit_seconds = 0.0

    for epoch in range(1, EPOCHS + 1):
        start = perf_counter()
        network.partial_fit(x_train, y_train, classes=LABELS)
        fit_seconds += perf_counter() - start

        train_probs = network.predict_proba(x_train)
        val_probs = network.predict_proba(x_val)
        val_pred = network.classes_[val_probs.argmax(axis=1)]
        row = {
            "epoch": epoch,
            "training_loss": log_loss(y_train, train_probs, labels=LABELS),
            "validation_loss": log_loss(y_val, val_probs, labels=LABELS),
            "validation_macro_f1": f1_score(
                y_val, val_pred, labels=LABELS, average="macro", zero_division=0
            ),
        }
        history.append(row)
        # Keep the earlier epoch if macro-F1 is tied.
        if row["validation_macro_f1"] > best_score:
            best_score = row["validation_macro_f1"]
            best_epoch = epoch
            best_network = copy.deepcopy(network)

        print(
            f"Epoch {epoch:2}/{EPOCHS}: train loss {row['training_loss']:.4f}, "
            f"val loss {row['validation_loss']:.4f}, "
            f"val macro-F1 {row['validation_macro_f1']:.4f}",
            flush=True,
        )

    return best_network, pd.DataFrame(history), best_epoch, fit_seconds


def save_loss_plot(history, best_epoch):
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot(history["epoch"], history["training_loss"], "o-", ms=3,
            color="#2468a2", label="Training")
    ax.plot(history["epoch"], history["validation_loss"], "s-", ms=3,
            color="#b45309", label="Validation")
    ax.axvline(best_epoch, color="#666666", linestyle="--", linewidth=1,
               label=f"Selected epoch ({best_epoch})")
    ax.set(title="Feedforward network loss by epoch", xlabel="Epoch",
           ylabel="Multiclass log loss", xticks=history["epoch"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(OUT_DIR / "neural_network_loss.png", dpi=300)
    plt.close(fig)


def main():
    train, val, test = load_splits()
    rows = {"train": len(train), "validation": len(val), "test": len(test)}
    vectorizer = joblib.load(VECTORIZER_PATH)
    x_train = vectorizer.transform(train["text"])
    x_val = vectorizer.transform(val["text"])
    if not (issparse(x_train) and issparse(x_val)):
        raise ValueError("Expected sparse TF-IDF matrices")
    # Float32 halves the memory used by the network weights and Adam state.
    x_train = x_train.astype(np.float32)
    x_val = x_val.astype(np.float32)
    print(f"Split sizes: {rows}. TF-IDF features: {x_train.shape[1]:,}", flush=True)

    network, history, best_epoch, fit_seconds = train_network(
        x_train, train["Sentiment"], x_val, val["Sentiment"]
    )
    best_score = float(history.loc[history["epoch"] == best_epoch, "validation_macro_f1"].iloc[0])
    pipeline = Pipeline([("vectorizer", vectorizer), ("classifier", network)])

    # Evaluate only after selecting the checkpoint on validation data.
    test_pred = pipeline.predict(test["text"])
    report = classification_report(
        test["Sentiment"], test_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(test["Sentiment"], test_pred, labels=LABELS)
    results = {
        "model": "Feedforward neural network",
        "owner": "Ethan Brown",
        "split": "data/splits/split_seed46.csv",
        "rows": rows,
        "labels": list(LABELS),
        "confusion_matrix_axes": "rows=actual, columns=predicted",
        "vocabulary_sizes": {"tfidf": x_train.shape[1]},
        "classifier_settings": {
            "class": "sklearn.neural_network.MLPClassifier",
            "hidden_layer_sizes": list(HIDDEN_LAYERS),
            "activation": "relu",
            "solver": "adam",
            "random_state": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "training_dtype": "float32",
        },
        "selected": {
            "features": "tfidf",
            "epoch": best_epoch,
            "validation_macro_f1": best_score,
        },
        "test_models": {
            "tfidf": {
                "model_path": MODEL_PATH.relative_to(ROOT).as_posix(),
                "vectorizer_path": VECTORIZER_PATH.relative_to(ROOT).as_posix(),
                "validation_macro_f1": best_score,
                "training_time_seconds": round(fit_seconds, 3),
                "classification_report": report,
                "confusion_matrix": matrix.tolist(),
            },
        },
        "training_time_scope": "Total time in partial_fit over all epochs, in seconds",
        "machine": {
            "platform": platform.platform(),
            "processor": platform.machine(),
            "cpus": os.cpu_count(),
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(OUT_DIR / "neural_network_training_history.csv", index=False)
    save_loss_plot(history, best_epoch)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Selected epoch {best_epoch} by validation macro-F1 ({best_score:.4f})")
    print(f"Training time: {fit_seconds:.3f} s")
    print(f"Test accuracy: {report['accuracy']:.4f}, macro-F1: {report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    main()
