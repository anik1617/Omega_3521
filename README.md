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
`notebooks/02_features_sophiya.ipynb`: default scikit-learn unigram vectorizers
fitted only on training text. The notebook does not save fitted vectorizers or
matrices. The script compares smoothing values 0.1, 1, and 10 using validation
macro-F1 separately for each representation, then reports test-set accuracy,
per-class precision, recall, F1, and confusion matrices for both BoW and TF-IDF.
It reports the training-set majority-class baseline on the same test set. The
overall validation winner and its fitted vectorizer are saved together in
`models/naive_bayes_sophiya_features_seed46.joblib`; all test metrics, vocabulary
sizes, and validation results are saved in
`report/model_results/naive_bayes_sophiya_features_seed46.json`.

## Streamlit App
Run `streamlit run app.py` to open the app
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

Live app URL: pending deployment.
Other model training instructions: pending implementation.
