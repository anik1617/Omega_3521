# Milestone 3: Logistic Regression and feature work (Sophiya Khurana)

Covers M3 task 1 (Logistic Regression on Bag of Words and TF-IDF, saved
vectorizers), carried-over task C3 (top TF-IDF terms), and Sophiya's part of
task 7 (Logistic Regression and feature representation write-up).

Run from the repository root:

```powershell
python scripts/train_logistic_regression.py
```

It takes about 10 minutes on a 4-core CPU (most of it is the validation grid).

## Files

| File | What it is |
| --- | --- |
| `scripts/train_logistic_regression.py` | Fits vectorizers, tunes and trains both models, writes everything below |
| `models/vectorizer_bow_seed46.joblib` | Fitted Bag of Words vectorizer (train split only) |
| `models/vectorizer_tfidf_seed46.joblib` | Fitted TF-IDF vectorizer (train split only) |
| `models/logistic_regression_bow_seed46.joblib` | Pipeline: Bag of Words + Logistic Regression |
| `models/logistic_regression_tfidf_seed46.joblib` | Pipeline: TF-IDF + Logistic Regression |
| `report/model_results/logistic_regression_sophiya_seed46.json` | Test metrics, confusion matrices, training time, tuning results (app.py loads this) |
| `validation_tuning.csv` | Validation macro-F1 and fit time for every C and class weight tried |
| `confusion_matrices_logistic_regression.png` | Test confusion matrices for both models |
| `top_terms_logistic_regression.csv` | Top 10 positive, negative, and neutral terms per representation |

The vectorizers are `data_utils.make_vectorizers()` fitted on the 30,123 training
reviews: `clean_text` preprocessing, unigrams plus bigrams, `min_df=2`, and
sublinear TF for TF-IDF. Both have 192,646 features. They are the same
vectorizers the Naive Bayes pipelines use, so the neural network can load them
with `joblib.load` instead of refitting. Put the repository root on `sys.path`
first, because they call `data_utils.clean_text`.

## Method

Multinomial (softmax) Logistic Regression, L2 penalty, lbfgs solver,
`max_iter=3000`. Two settings were tuned on the validation split by macro-F1:

- `C` (inverse regularization strength): 0.01, 0.03, 0.1, 0.3, 1, 3, 10.
- `class_weight`: none, or `balanced` (weights each class by 1 / its frequency, so
  the 7% neutral class counts as much as the 78% positive class in the loss).

The best setting for each representation was refitted on the training split and
evaluated once on the test split. Training time is vectorizer fit plus
classifier fit on the training split, measured on a 4-core x86_64 Linux machine
(Python 3.13, scikit-learn 1.8.0).

## Results (test set, 9,269 reviews)

| Model | Selected setting | Val macro-F1 | Accuracy | Macro precision | Macro recall | Macro F1 | Neutral F1 | Training time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LR + TF-IDF | C=3, balanced | 0.698 | 0.869 | 0.690 | 0.717 | 0.702 | 0.420 | 21.0 s |
| LR + Bag of Words | C=0.1, balanced | 0.689 | 0.864 | 0.679 | 0.700 | 0.689 | 0.394 | 43.4 s |
| NB + Bag of Words (M2, for reference) | alpha 0.3, equal priors | | 0.860 | | | 0.673 | 0.364 | |

Confusion matrices (rows actual, columns predicted; order negative, neutral, positive):

| LR + TF-IDF | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 1,074 | 151 | 142 |
| neutral | 179 | 299 | 201 |
| positive | 245 | 296 | 6,682 |

| LR + Bag of Words | negative | neutral | positive |
| --- | ---: | ---: | ---: |
| negative | 1,053 | 156 | 158 |
| neutral | 180 | 274 | 225 |
| positive | 258 | 281 | 6,684 |

Logistic Regression + TF-IDF has the best validation macro-F1 of the classical
models so far, so app.py now marks it as the best model.

## Top TF-IDF terms (task C3)

Ranked by Logistic Regression coefficient: positive and negative use
coef(positive) minus coef(negative); neutral uses the neutral class's own
coefficients. Full list with scores, and the Bag of Words version, in
`top_terms_logistic_regression.csv`.

| Rank | Positive | Negative | Neutral |
| ---: | --- | --- | --- |
| 1 | great | not | but |
| 2 | delicious | disappointed | ok |
| 3 | best | terrible | however |
| 4 | good | not worth | okay |
| 5 | excellent | awful | not bad |
| 6 | perfect | disappointing | not great |
| 7 | the best | horrible | stars |
| 8 | love | bad | update |
| 9 | wonderful | yuck | not |
| 10 | loves | unfortunately | decent |

## Report text: Logistic Regression and feature representation

We trained Logistic Regression on the same two feature sets as Naive Bayes. Both
use the shared cleaning step (lowercase, HTML tags removed, "n't" expanded to
"not") and keep unigrams and bigrams that appear in at least two training
reviews, which gives 192,646 features. Bag of Words stores raw counts. TF-IDF
uses 1 + log of the count, scales each term by how rare it is across reviews,
and normalizes every review to unit length, so long reviews and very common
words carry less weight. We tuned the regularization strength C and the class
weighting on the validation set. Balanced class weights won for both feature
sets, because without them the model rarely predicts neutral: at C=1 on TF-IDF,
validation macro-F1 was 0.619 unweighted and 0.686 balanced. TF-IDF preferred
weaker regularization (C=3) than Bag of Words (C=0.1), since its feature values
are small fractions while counts can be large.

On the test set, Logistic Regression with TF-IDF reached 86.9% accuracy and a
macro-F1 of 0.702, compared with 86.0% and 0.673 for our best Naive Bayes model.
The largest gain is on the neutral class (F1 0.420 against 0.364). Naive Bayes
treats every word as independent evidence and multiplies the probabilities, so
a three-star review that mentions both "great" and "disappointed" gets pushed
hard toward one side. Logistic Regression learns one weight per feature
directly from the labels and can give mixed-review markers such as "but",
"however", "ok", and "not bad" their own weight toward neutral, as the top
terms show. Bigrams also matter: "not worth", "not bad", and "not great" are
among the strongest terms, and they only exist because we kept two-word phrases
and expanded "n't". TF-IDF beat counts for Logistic Regression (0.702 against
0.689) and trained in half the time (21 s against 43 s), because the
normalized features make the optimization converge in fewer steps. Neutral is
still the weak class: 296 positive and 151 negative test reviews are predicted
neutral, and only 299 of 679 neutral reviews are found. A bag of words model
cannot tell which part of "good flavor, but the box arrived crushed" decides
the rating, which is where a model that uses word order may help.

## Progress log (Sophiya Khurana)

Contribution since M2: Trained Logistic Regression on Bag of Words and TF-IDF
(task 1), tuning C and class weighting on the validation set, and saved both
models and both fitted vectorizers to `models/` so the app and the neural network
can load them. Logistic Regression with TF-IDF is our best classical model so
far (test accuracy 0.869, macro-F1 0.702, neutral F1 0.420), and app.py picks it
up automatically. Listed the top TF-IDF terms per class (task C3) and wrote the
Logistic Regression and feature representation part of the model comparison
section.

Remaining tasks: help compare Logistic Regression with the neural network once
it is trained, and write the dataset and features section of the final report.
