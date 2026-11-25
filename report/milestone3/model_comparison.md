# Model comparison (Milestone 3)

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Neutral F1 | Training time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Multinomial Naive Bayes (BoW)** (best) | 0.860 | 0.674 | 0.672 | 0.673 | 0.364 | 3.24 |
| Multinomial Naive Bayes (TF-IDF) | 0.859 | 0.668 | 0.668 | 0.667 | 0.352 | 3.71 |
| Majority-class baseline | 0.779 | 0.260 | 0.333 | 0.292 | 0.000 | n/a |
| Logistic Regression (BoW) | pending | pending | pending | pending | pending | pending |
| Logistic Regression (TF-IDF) | pending | pending | pending | pending | pending | pending |
| Feedforward neural network (TF-IDF) | pending | pending | pending | pending | pending | pending |
All numbers are on the same 20% test split (9,269 reviews, seed 46). Precision, recall, and F1 are macro averages over negative, neutral, and positive.

Training time is the fit time on the training split (vectorizer plus the selected model), not counting hyperparameter search.

Best model by validation macro F1: Multinomial Naive Bayes (BoW).

Timed on: Linux-6.18.44-fc-v880-x86_64-with-glibc2.39, 4 CPUs.
