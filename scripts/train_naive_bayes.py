"""Train Naive Bayes with the features from Sophiya's seed-46 notebook.

Run from the repository root with ``python scripts/train_naive_bayes.py``.
The notebook does not save fitted vectorizers, so this script rebuilds the same
Bag of Words and TF-IDF features with data_utils.make_vectorizers() on the
training split. Each representation's best vectorizer and classifier are saved
together as one pipeline, so the app and the error analysis can load either.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils import MIN_DF, NGRAM_RANGE, load_splits, make_vectorizers  # noqa: E402

LABELS = ("negative", "neutral", "positive")
# TF-IDF values are small fractions, so it needs smaller smoothing than counts.
ALPHAS = (0.01, 0.03, 0.1, 0.3, 1.0)
# fit_prior=False uses equal class priors instead of the 78% positive training
# share, which otherwise pushes borderline (often neutral) reviews to positive.
FIT_PRIORS = (True, False)
MODEL_PATHS = {
    "bag_of_words": ROOT / "models/naive_bayes_bow_seed46.joblib",
    "tfidf": ROOT / "models/naive_bayes_tfidf_seed46.joblib",
}
REPORT_PATH = ROOT / "report/model_results/naive_bayes_sophiya_features_seed46.json"


def evaluate(y_true, y_pred):
    """Return class metrics and a confusion matrix with a fixed label order."""
    return {
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=LABELS
        ).tolist(),
    }


def train_naive_bayes():
    """Tune each representation on validation, then evaluate both on test.

    Returns the fitted text-to-sentiment pipeline and a JSON-ready results dict.
    The vectorizers and classifiers are fitted only on the training partition.
    """
    train, val, test = load_splits()
    if any(frame.empty for frame in (train, val, test)):
        raise ValueError("Train, validation, and test partitions must all be nonempty")

    # Match notebooks/02_features_sophiya.ipynb exactly: the shared vectorizers
    # (cleaned text, unigrams plus bigrams, min_df=2) fitted on train['text'].
    bow, tfidf = make_vectorizers()
    features = {"bag_of_words": bow, "tfidf": tfidf}
    candidates = []
    vocabulary_sizes = {}
    best_by_feature = {}

    for feature_name, vectorizer in features.items():
        X_train = vectorizer.fit_transform(train["text"])
        X_val = vectorizer.transform(val["text"])
        vocabulary_sizes[feature_name] = len(vectorizer.vocabulary_)

        for alpha in ALPHAS:
            for fit_prior in FIT_PRIORS:
                classifier = MultinomialNB(alpha=alpha, fit_prior=fit_prior).fit(
                    X_train, train["Sentiment"]
                )
                score = float(f1_score(
                    val["Sentiment"], classifier.predict(X_val),
                    labels=LABELS, average="macro", zero_division=0,
                ))
                candidates.append({
                    "features": feature_name,
                    "alpha": alpha,
                    "fit_prior": fit_prior,
                    "validation_macro_f1": score,
                })
                if (feature_name not in best_by_feature or
                        score > best_by_feature[feature_name]["validation_macro_f1"]):
                    best_by_feature[feature_name] = {
                        "features": feature_name,
                        "alpha": alpha,
                        "fit_prior": fit_prior,
                        "validation_macro_f1": score,
                        "vectorizer": vectorizer,
                        "classifier": classifier,
                    }

    pipelines = {}
    test_models = {}
    for feature_name, chosen in best_by_feature.items():
        pipeline = Pipeline([
            ("vectorizer", chosen["vectorizer"]),
            ("classifier", chosen["classifier"]),
        ])
        pipelines[feature_name] = pipeline
        test_models[feature_name] = {
            "model_path": str(MODEL_PATHS[feature_name].relative_to(ROOT)),
            "alpha": chosen["alpha"],
            "fit_prior": chosen["fit_prior"],
            "validation_macro_f1": chosen["validation_macro_f1"],
            **evaluate(test["Sentiment"], pipeline.predict(test["text"])),
        }

    best = max(best_by_feature.values(), key=lambda item: item["validation_macro_f1"])
    majority_class = train["Sentiment"].value_counts().idxmax()
    baseline_predictions = pd.Series(majority_class, index=test.index)

    results = {
        "split": "data/splits/split_seed46.csv",
        "feature_source": "notebooks/02_features_sophiya.ipynb",
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
        "vocabulary_sizes": vocabulary_sizes,
        "validation_candidates": candidates,
        "selected": {
            "features": best["features"],
            "alpha": best["alpha"],
            "fit_prior": best["fit_prior"],
            "validation_macro_f1": best["validation_macro_f1"],
        },
        "test_models": test_models,
        "majority_baseline": {
            "predicted_class": majority_class,
            **evaluate(test["Sentiment"], baseline_predictions),
        },
    }
    return pipelines, results


def main():
    pipelines, results = train_naive_bayes()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for feature_name, pipeline in pipelines.items():
        path = MODEL_PATHS[feature_name]
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, path, compress=3)
        print(f"Saved {feature_name} model and vectorizer: {path}")
    REPORT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"Saved metrics: {REPORT_PATH}")
    selected = results["selected"]
    print(f"Selected by validation macro-F1: {selected['features']}, "
          f"alpha={selected['alpha']}, fit_prior={selected['fit_prior']}")
    comparisons = [
        ("Naive Bayes + Bag of Words", results["test_models"]["bag_of_words"]),
        ("Naive Bayes + TF-IDF", results["test_models"]["tfidf"]),
        ("Majority-class baseline", results["majority_baseline"]),
    ]
    for name, result in comparisons:
        report = result["classification_report"]
        print(f"\n{name}: accuracy={report['accuracy']:.4f}")
        print(f"{'class':<13}{'precision':>11}{'recall':>10}{'F1':>10}{'support':>10}")
        for label in (*LABELS, "macro avg", "weighted avg"):
            row = report[label]
            print(f"{label:<13}{row['precision']:>11.4f}{row['recall']:>10.4f}"
                  f"{row['f1-score']:>10.4f}{int(row['support']):>10}")
        print("Confusion matrix (actual rows, predicted columns):")
        print(pd.DataFrame(result["confusion_matrix"], index=LABELS,
                           columns=LABELS).to_string())


if __name__ == "__main__":
    main()
