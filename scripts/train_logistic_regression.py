"""Milestone 3 task 1 (Sophiya): Logistic Regression on Bag of Words and TF-IDF.

Run from the repository root with ``python scripts/train_logistic_regression.py``.

Steps:
1. Fit the shared Bag of Words and TF-IDF vectorizers (data_utils.make_vectorizers)
   on the training split and save them to models/ so the app and the neural
   network can reuse the exact same features.
2. For each representation, tune C and class_weight on the validation split by
   macro-F1. The test split is not used for any choice.
3. Refit the chosen setting on train, time the fit, evaluate once on test, and
   save each vectorizer + classifier as one scikit-learn Pipeline.
4. Write metrics in the format app.py reads (report/model_results/*.json), plot
   the two confusion matrices, and list the top 10 positive and negative terms
   by Logistic Regression coefficient for each representation.
"""

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils import MIN_DF, NGRAM_RANGE, load_splits, make_vectorizers  # noqa: E402

LABELS = ("negative", "neutral", "positive")
FEATURE_NAMES = {"bag_of_words": "Bag of Words", "tfidf": "TF-IDF"}
# C is the inverse regularization strength: small C = simpler model.
# Raw counts are larger than TF-IDF values, so BoW tends to want a smaller C.
C_VALUES = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
# "balanced" reweights classes by 1 / frequency, so the rare neutral class
# (about 7% of training reviews) counts as much as positive in the loss.
CLASS_WEIGHTS = (None, "balanced")
MAX_ITER = 3000
RANDOM_STATE = 46

VECTORIZER_PATHS = {
    "bag_of_words": ROOT / "models/vectorizer_bow_seed46.joblib",
    "tfidf": ROOT / "models/vectorizer_tfidf_seed46.joblib",
}
MODEL_PATHS = {
    "bag_of_words": ROOT / "models/logistic_regression_bow_seed46.joblib",
    "tfidf": ROOT / "models/logistic_regression_tfidf_seed46.joblib",
}
OUT_DIR = ROOT / "report/milestone3/sophiya"
REPORT_PATH = ROOT / "report/model_results/logistic_regression_sophiya_seed46.json"


def make_classifier(C, class_weight):
    return LogisticRegression(C=C, class_weight=class_weight, max_iter=MAX_ITER,
                              random_state=RANDOM_STATE)


def evaluate(y_true, y_pred):
    """Class metrics and a confusion matrix (rows actual, columns predicted)."""
    return {
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def macro_f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def top_terms(pipeline, feature_name, k=10):
    """Top k terms pushing toward positive and toward negative.

    The score for a term is coef[positive] - coef[negative]: how much one unit
    of that feature moves the positive-vs-negative log odds.
    """
    vectorizer = pipeline.named_steps["vectorizer"]
    classifier = pipeline.named_steps["classifier"]
    terms = vectorizer.get_feature_names_out()
    classes = list(classifier.classes_)
    coef = classifier.coef_
    score = coef[classes.index("positive")] - coef[classes.index("negative")]
    neutral = coef[classes.index("neutral")]
    rows = []
    for direction, order in (("positive", np.argsort(-score)), ("negative", np.argsort(score))):
        for rank, idx in enumerate(order[:k], start=1):
            rows.append({
                "features": feature_name, "direction": direction, "rank": rank,
                "term": terms[idx], "positive_minus_negative_coef": round(float(score[idx]), 4),
            })
    # Neutral has its own one-vs-rest style weights in the multinomial model.
    for rank, idx in enumerate(np.argsort(-neutral)[:k], start=1):
        rows.append({
            "features": feature_name, "direction": "neutral", "rank": rank,
            "term": terms[idx], "positive_minus_negative_coef": round(float(score[idx]), 4),
        })
    return rows


def plot_confusion(test_models, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (feature_name, result) in zip(axes, test_models.items()):
        matrix = np.array(result["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, matrix[i, j], ha="center", va="center",
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        macro = result["classification_report"]["macro avg"]["f1-score"]
        ax.set_title(f"LR + {FEATURE_NAMES[feature_name]} (macro-F1 {macro:.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    train, val, test = load_splits()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    bow, tfidf = make_vectorizers()
    vectorizers = {"bag_of_words": bow, "tfidf": tfidf}
    candidates, best_by_feature, vocabulary_sizes = [], {}, {}

    for feature_name, vectorizer in vectorizers.items():
        start = time.perf_counter()
        X_train = vectorizer.fit_transform(train["text"])
        vectorizer_seconds = time.perf_counter() - start
        X_val = vectorizer.transform(val["text"])
        vocabulary_sizes[feature_name] = len(vectorizer.vocabulary_)
        joblib.dump(vectorizer, VECTORIZER_PATHS[feature_name], compress=3)
        print(f"Saved fitted {feature_name} vectorizer: {VECTORIZER_PATHS[feature_name]}")

        for class_weight in CLASS_WEIGHTS:
            for C in C_VALUES:
                start = time.perf_counter()
                classifier = make_classifier(C, class_weight).fit(X_train, train["Sentiment"])
                seconds = time.perf_counter() - start
                score = macro_f1(val["Sentiment"], classifier.predict(X_val))
                row = {"features": feature_name, "C": C,
                       "class_weight": class_weight or "none",
                       "validation_macro_f1": round(score, 4),
                       "fit_seconds": round(seconds, 2)}
                candidates.append(row)
                print(row)
                best = best_by_feature.get(feature_name)
                if best is None or score > best["validation_macro_f1"]:
                    best_by_feature[feature_name] = {
                        "C": C, "class_weight": class_weight,
                        "validation_macro_f1": score,
                        "vectorizer_fit_seconds": vectorizer_seconds,
                    }

    test_models, pipelines, term_rows = {}, {}, []
    for feature_name, chosen in best_by_feature.items():
        # Refit from raw text so the reported time covers vectorizer + classifier.
        vectorizer = make_vectorizers()[0 if feature_name == "bag_of_words" else 1]
        pipeline = Pipeline([
            ("vectorizer", vectorizer),
            ("classifier", make_classifier(chosen["C"], chosen["class_weight"])),
        ])
        start = time.perf_counter()
        pipeline.fit(train["text"], train["Sentiment"])
        train_seconds = time.perf_counter() - start
        joblib.dump(pipeline, MODEL_PATHS[feature_name], compress=3)
        pipelines[feature_name] = pipeline
        test_models[feature_name] = {
            "model_path": str(MODEL_PATHS[feature_name].relative_to(ROOT)).replace("\\", "/"),
            "vectorizer_path": str(VECTORIZER_PATHS[feature_name].relative_to(ROOT)).replace("\\", "/"),
            "C": chosen["C"],
            "class_weight": chosen["class_weight"] or "none",
            "validation_macro_f1": chosen["validation_macro_f1"],
            "training_time_seconds": round(train_seconds, 2),
            **evaluate(test["Sentiment"], pipeline.predict(test["text"])),
        }
        term_rows += top_terms(pipeline, feature_name)

    best_feature = max(best_by_feature, key=lambda f: best_by_feature[f]["validation_macro_f1"])
    results = {
        "model": "Logistic Regression",
        "owner": "Sophiya Khurana",
        "split": "data/splits/split_seed46.csv",
        "rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "labels": list(LABELS),
        "confusion_matrix_axes": "rows=actual, columns=predicted",
        "vectorizer_settings": {
            "source": "data_utils.make_vectorizers()",
            "preprocessor": "data_utils.clean_text: lowercase, HTML tags removed, n't expanded to not",
            "ngram_range": list(NGRAM_RANGE),
            "min_df": MIN_DF,
            "count_vectorizer": "raw counts",
            "tfidf_vectorizer": "sublinear_tf=True, smoothed IDF, L2 normalization",
        },
        "classifier_settings": {
            "class": "sklearn.linear_model.LogisticRegression",
            "solver": "lbfgs (multinomial softmax)", "penalty": "l2",
            "max_iter": MAX_ITER, "random_state": RANDOM_STATE,
            "tuned_on_validation": {"C": list(C_VALUES),
                                    "class_weight": [w or "none" for w in CLASS_WEIGHTS]},
        },
        "vocabulary_sizes": vocabulary_sizes,
        "validation_candidates": candidates,
        "selected": {"features": best_feature,
                     "C": best_by_feature[best_feature]["C"],
                     "class_weight": best_by_feature[best_feature]["class_weight"] or "none",
                     "validation_macro_f1": best_by_feature[best_feature]["validation_macro_f1"]},
        "test_models": test_models,
        "machine": {"platform": platform.platform(), "processor": platform.machine(),
                    "python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    REPORT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(term_rows).to_csv(OUT_DIR / "top_terms_logistic_regression.csv", index=False)
    plot_confusion(test_models, OUT_DIR / "confusion_matrices_logistic_regression.png")
    pd.DataFrame(candidates).to_csv(OUT_DIR / "validation_tuning.csv", index=False)
    print(f"\nSaved metrics: {REPORT_PATH}")

    for feature_name, result in test_models.items():
        report = result["classification_report"]
        print(f"\nLR + {FEATURE_NAMES[feature_name]}: C={result['C']}, "
              f"class_weight={result['class_weight']}, train {result['training_time_seconds']} s, "
              f"accuracy={report['accuracy']:.4f}")
        for label in (*LABELS, "macro avg"):
            row = report[label]
            print(f"  {label:<10} P={row['precision']:.4f} R={row['recall']:.4f} "
                  f"F1={row['f1-score']:.4f} n={int(row['support'])}")
        print(pd.DataFrame(result["confusion_matrix"], index=LABELS, columns=LABELS).to_string())


if __name__ == "__main__":
    main()
