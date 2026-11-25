# Models

Saved models and fitted vectorizers go here (for example with `joblib.dump`), named
by model and features, such as `naive_bayes_tfidf.joblib`. Save the vectorizer with
the model so the Streamlit app applies the same preprocessing used in training.

Current files, written by `scripts/train_naive_bayes.py`:

- `naive_bayes_bow_seed46.joblib`: Bag of Words + Multinomial Naive Bayes
- `naive_bayes_tfidf_seed46.joblib`: TF-IDF + Multinomial Naive Bayes

Written by `scripts/train_logistic_regression.py` (Sophiya, M3 task 1):

- `logistic_regression_tfidf_seed46.joblib`: TF-IDF + Logistic Regression (best by validation macro-F1)
- `logistic_regression_bow_seed46.joblib`: Bag of Words + Logistic Regression
- `vectorizer_bow_seed46.joblib`, `vectorizer_tfidf_seed46.joblib`: the shared
  vectorizers fitted on the training split alone, for models (such as the neural
  network) that need features without a classifier attached

Each model file is a scikit-learn `Pipeline` that takes raw review text (Summary and Text
joined by a space). The vectorizers call `data_utils.clean_text`, so put the
repository root on `sys.path` before `joblib.load`.
