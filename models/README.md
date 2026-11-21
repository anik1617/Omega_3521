# Models

Saved models and fitted vectorizers go here (for example with `joblib.dump`), named
by model and features, such as `naive_bayes_tfidf.joblib`. Save the vectorizer with
the model so the Streamlit app applies the same preprocessing used in training.

Current files, written by `scripts/train_naive_bayes.py`:

- `naive_bayes_bow_seed46.joblib`: Bag of Words + Multinomial Naive Bayes (selected model)
- `naive_bayes_tfidf_seed46.joblib`: TF-IDF + Multinomial Naive Bayes

Each is a scikit-learn `Pipeline` that takes raw review text (Summary and Text
joined by a space). The vectorizers call `data_utils.clean_text`, so put the
repository root on `sys.path` before `joblib.load`.
