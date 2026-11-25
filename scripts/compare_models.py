"""Build the Milestone 3 model comparison table and confusion matrix figure.

Run from the repository root after training any model:

    python scripts/compare_models.py

It reads every results file in report/model_results/*.json, so a new model
shows up as soon as its training script writes one. Each file needs:

    "model": display name, e.g. "Logistic Regression"
    "labels": ["negative", "neutral", "positive"]
    "test_models": {
        "<features>": {                      # e.g. "bag_of_words", "tfidf"
            "validation_macro_f1": float,
            "training_time_seconds": float,  # fit time on the training split
            "classification_report": sklearn output_dict on the test split,
            "confusion_matrix": [[...]],     # rows actual, columns predicted
        }
    }
    "machine": {...}                         # optional, where it was timed

This is the same format scripts/train_naive_bayes.py writes and app.py reads.
Models the team plans but has not trained yet are listed as pending.

Outputs go to report/milestone3/: model_comparison.csv, model_comparison.md,
and confusion_matrices.png. The best model is picked by validation macro-F1,
the same rule app.py uses, so the test set is never used to choose it.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "report/model_results"
OUT_DIR = ROOT / "report/milestone3"
LABELS = ["negative", "neutral", "positive"]
FEATURE_NAMES = {"bag_of_words": "BoW", "tfidf": "TF-IDF"}
# Models in the M3 plan. A row stays "pending" until its results file exists.
PLANNED = [
    ("Logistic Regression", "BoW"),
    ("Logistic Regression", "TF-IDF"),
    ("Feedforward neural network", "TF-IDF"),
]


def collect_rows():
    """Return one dict per trained model plus its confusion matrix."""
    rows, matrices = [], {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        results = json.loads(path.read_text(encoding="utf-8"))
        model = results.get("model", path.stem)
        machine = results.get("machine", {})
        for features, saved in results.get("test_models", {}).items():
            report = saved["classification_report"]
            name = f"{model} ({FEATURE_NAMES.get(features, features)})"
            seconds = saved.get("training_time_seconds")
            rows.append({
                "Model": name,
                "Accuracy": report["accuracy"],
                "Macro precision": report["macro avg"]["precision"],
                "Macro recall": report["macro avg"]["recall"],
                "Macro F1": report["macro avg"]["f1-score"],
                "Neutral F1": report["neutral"]["f1-score"],
                "Training time (s)": seconds,
                "Validation macro F1": saved.get("validation_macro_f1"),
                "Machine": ", ".join(str(machine[k]) for k in ("platform", "cpus") if k in machine)
                           + (" CPUs" if "cpus" in machine else ""),
                "Results file": str(path.relative_to(ROOT)),
            })
            matrices[name] = np.array(saved["confusion_matrix"])
        baseline = results.get("majority_baseline")
        if baseline and not any(r["Model"] == "Majority-class baseline" for r in rows):
            report = baseline["classification_report"]
            rows.append({
                "Model": "Majority-class baseline",
                "Accuracy": report["accuracy"],
                "Macro precision": report["macro avg"]["precision"],
                "Macro recall": report["macro avg"]["recall"],
                "Macro F1": report["macro avg"]["f1-score"],
                "Neutral F1": report["neutral"]["f1-score"],
                "Training time (s)": None,
                "Validation macro F1": None,
                "Machine": "",
                "Results file": str(path.relative_to(ROOT)),
            })

    trained = {r["Model"] for r in rows}
    for model, features in PLANNED:
        name = f"{model} ({features})"
        if not any(t.startswith(model) and features in t for t in trained):
            rows.append({"Model": name, "Results file": "pending"})
    return pd.DataFrame(rows), matrices


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "pending" if digits else ""
    return f"{value:.{digits}f}"


def write_markdown(table, best):
    metric_cols = ["Accuracy", "Macro precision", "Macro recall", "Macro F1", "Neutral F1"]
    lines = [
        "| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Neutral F1 | Training time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in table.iterrows():
        pending = row["Results file"] == "pending"
        name = f"**{row['Model']}** (best)" if row["Model"] == best else row["Model"]
        if pending:
            cells = ["pending"] * (len(metric_cols) + 1)
        else:
            cells = [fmt(row[c]) for c in metric_cols]
            seconds = row["Training time (s)"]
            cells.append("n/a" if row["Model"] == "Majority-class baseline"
                         else fmt(seconds, 2) if pd.notna(seconds) else "not recorded")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    machines = sorted({m for m in table.get("Machine", pd.Series(dtype=str)).dropna() if m})
    notes = [
        "All numbers are on the same 20% test split (9,269 reviews, seed 46). "
        "Precision, recall, and F1 are macro averages over negative, neutral, and positive.",
        "Training time is the fit time on the training split (vectorizer plus the selected model), "
        "not counting hyperparameter search.",
    ]
    if best:
        notes.append(f"Best model by validation macro F1: {best}.")
    if machines:
        notes.append("Timed on: " + "; ".join(machines) + ".")
    text = "# Model comparison (Milestone 3)\n\n" + "\n".join(lines) + "\n" + "\n\n".join(notes) + "\n"
    (OUT_DIR / "model_comparison.md").write_text(text, encoding="utf-8")


def plot_confusion_matrices(matrices, best):
    names = list(matrices)
    cols = min(3, len(names))
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.9 * rows), squeeze=False)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    for ax, name in zip(axes.flat, names):
        matrix = matrices[name]
        # Color by row share so the small neutral class is readable next to positive.
        shares = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, f"{matrix[i, j]}\n{shares[i, j]:.0%}", ha="center", va="center",
                        fontsize=9, color="white" if shares[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(LABELS)), LABELS)
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name + ("  (best)" if name == best else ""), fontsize=10)
    fig.suptitle("Test-set confusion matrices (count and share of each actual class)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrices.png", dpi=200)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table, matrices = collect_rows()
    scored = table.dropna(subset=["Validation macro F1"]) if "Validation macro F1" in table else table.iloc[0:0]
    best = scored.loc[scored["Validation macro F1"].idxmax(), "Model"] if len(scored) else None

    table.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    write_markdown(table, best)
    if matrices:
        plot_confusion_matrices(matrices, best)

    shown = table[["Model", "Accuracy", "Macro F1", "Neutral F1", "Training time (s)"]]
    print(shown.to_string(index=False, na_rep="pending", float_format="{:.3f}".format))
    print(f"\nBest by validation macro F1: {best}")
    print(f"Wrote {OUT_DIR.relative_to(ROOT)}/model_comparison.csv, model_comparison.md, "
          "confusion_matrices.png")


if __name__ == "__main__":
    main()
