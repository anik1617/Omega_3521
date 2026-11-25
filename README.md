# Group Omega — CSE 3521

Amazon food review sentiment analysis project. Planned deliverables include text
feature engineering, classical and deep/modern model comparisons, and a deployed
Streamlit application. Multinomial Naive Bayes training is implemented; the
application and other models are not implemented yet.

## Team

| Member | Primary role |
| --- | --- |
| Aniket Chaudhari | Project Lead |
| Sophiya Khurana | Data Engineer / Writer |
| Ethan Brown | Data Engineer / Writer |
| Lianyu Yang | App Engineer |
| Shilin Dong | ML Engineer |

## Data

The team dataset is committed: `data/raw/group_sample_seed46.csv` (50,000 reviews) and
`data/processed/group_seed46_clean.csv` (46,344 reviews with a `Sentiment` column).
Load the clean file directly. The commands below only rebuild it from the full
Kaggle download. Python 3.11+.

```powershell
pip install -r requirements.txt
python scripts/setup_data.py
python scripts/prepare_sample.py
```

`setup_data.py` downloads the full Kaggle `Reviews.csv`. `prepare_sample.py` draws
Group Omega's 50,000-review sample with the instructor's method and our assigned
seed, 46 (`df.sample(n=50000, random_state=46)`), then writes the cleaned
three-class dataset and the summary statistics. Labels: negative (1–2 stars),
neutral (3 stars), positive (4–5 stars). See [data setup instructions](data/README.md).

## Train / validation / test split

Everyone uses one shared split, committed in `data/splits/split_seed46.csv` (review
`Id` and `split`). It is stratified by sentiment, 65/15/20, with seed 46 (our group seed, same as the sample):

| Split | Negative | Neutral | Positive | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 4,444 | 2,206 | 23,473 | 30,123 |
| Validation | 1,026 | 509 | 5,417 | 6,952 |
| Test | 1,367 | 679 | 7,223 | 9,269 |

Load it in code or notebooks with:

```python
from data_utils import load_splits   # in notebooks/: import sys; sys.path.insert(0, '..')
train, val, test = load_splits()      # model input is the `text` column, label is `Sentiment`
```

Fit vectorizers and models on `train`, tune on `val`, and report numbers from `test`.
Do not rerun `scripts/make_splits.py`; it already made the committed split.

## Multinomial Naive Bayes

Run from the repository root using a local Python environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\train_naive_bayes.py
```

The script rebuilds the Bag-of-Words and TF-IDF features from
`notebooks/02_features_sophiya.ipynb` with the shared `data_utils.make_vectorizers()`:
text cleaned by `data_utils.clean_text` (lowercase, HTML tags such as `<br />`
removed, `n't` expanded to `not`), unigrams plus bigrams, and terms in at least
two training reviews (`min_df=2`). TF-IDF also uses sublinear term frequency.
Vectorizers are fitted only on training text. For each representation the script
tries smoothing values 0.01, 0.03, 0.1, 0.3, and 1, with learned class priors and
with equal priors (`fit_prior=False`), and keeps the setting with the best
validation macro-F1. It then reports test-set accuracy, per-class precision,
recall, F1, and confusion matrices for both representations, plus the
training-set majority-class baseline on the same test set. Each representation's
best vectorizer and classifier are saved together in
`models/naive_bayes_bow_seed46.joblib` and `models/naive_bayes_tfidf_seed46.joblib`;
all test metrics, vocabulary sizes, and validation results are saved in
`report/model_results/naive_bayes_sophiya_features_seed46.json`.

| Test set | Accuracy | Macro-F1 | Neutral F1 | Training time |
| --- | ---: | ---: | ---: | ---: |
| Majority-class baseline | 0.779 | 0.292 | 0.000 | n/a |
| Naive Bayes + Bag of Words (alpha 0.3, equal priors; selected) | 0.860 | 0.673 | 0.364 | about 3.5 s |
| Naive Bayes + TF-IDF (alpha 0.03, equal priors) | 0.859 | 0.667 | 0.353 | about 3.7 s |

Training time is the vectorizer fit plus the selected classifier's fit on the
training split, measured on a 4-CPU Linux machine; almost all of it is building
the 192,646-term vocabulary. The exact seconds are in the results JSON.

TF-IDF values are small fractions, so Multinomial Naive Bayes needs much smaller
smoothing on TF-IDF than on counts. With the earlier alpha grid (0.1 to 10)
TF-IDF looked far worse than Bag of Words; with small alpha the two are close.

## Streamlit App
Run `streamlit run app.py` to open the app. It loads every trained model listed in
`report/model_results/*.json` (each entry's `model_path`), marks the one with the
best validation macro-F1 as the best model, and shows a test-set metrics table.
Models not trained yet appear as placeholders with random predictions. To add a
model, have its training script write a results file in the same format as
`scripts/train_naive_bayes.py`.

## Model comparison

After training any model, rebuild the comparison table and confusion matrices:

```powershell
python scripts/compare_models.py
```

It reads every `report/model_results/*.json` and writes
`report/milestone3/model_comparison.md` (and `.csv`) plus
`report/milestone3/confusion_matrices.png`. The best model is picked by
validation macro-F1, the same rule the app uses. Models that are planned but not
trained yet show as `pending`. A training script adds its model by writing a
results file in the same format as `scripts/train_naive_bayes.py`, including
`training_time_seconds` for each saved model (the format is described at the top
of `scripts/compare_models.py`).

## Reproducing our results

1. **Get the code and data.** Clone this repository. The seed-46 dataset and the
   split are committed (`data/processed/group_seed46_clean.csv`,
   `data/splits/split_seed46.csv`), so no download is needed. To rebuild the
   dataset from scratch, download the Kaggle
   [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
   data with `python scripts/setup_data.py`, then run
   `python scripts/prepare_sample.py` (see [data/README.md](data/README.md)).
2. **Install dependencies** (Python 3.11 or newer; versions are pinned):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Retrain the models.** Each script fits on the training split, tunes on the
   validation split, reports on the test split, saves the model with its
   vectorizer to `models/`, and writes metrics to `report/model_results/`.
   ```powershell
   python scripts/train_naive_bayes.py          # Multinomial Naive Bayes
   # Logistic Regression: command added when its training script is merged
   # Feedforward neural network: command added when its training script is merged
   python scripts/compare_models.py             # comparison table and confusion matrices
   ```
   Do not rerun `scripts/make_splits.py`; the committed split is the one every
   model is evaluated on.
4. **Run the app:** `streamlit run app.py`. The saved models in `models/` are
   loaded directly, so the app does not retrain anything.

Live app URL: pending deployment.

## Milestone 3

- [M3 tracker](https://docs.google.com/document/d/18krXrnA5Ti16ZI3yS_n51BFG3RM6v0pROyJ6z9O1ceA/edit): tasks, owners, deadlines, progress log.

## Milestone 2

- [M2 tracker](https://docs.google.com/document/d/1NkAAW1ialDPXoOkk2UPxT6xhLj0Yuoiy5e14s_X0dLo/edit): tasks, owners, deadlines, progress log.
- Notebooks go in `notebooks/`, saved models and vectorizers in `models/`.

## Milestone 1

- [Working charter](https://docs.google.com/document/d/1wIlXwJSuGBldoeKHOvrOfHx-uvpAIfMrQg1aW8XH1gY/edit)
- [Draft charter PDF](report/milestone1_draft.pdf)
- [Dataset measurements](report/dataset_summary/SUMMARY.md)
- [Sentiment chart](report/dataset_summary/sentiment_seed46.png)

The PDF is a draft snapshot. Check the working charter for current edits and
remaining placeholders before submission.
