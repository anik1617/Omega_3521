"""Load Group Omega's cleaned reviews with the shared train/validation/test split.

    from data_utils import load_splits
    train, val, test = load_splits()
    X_train, y_train = train['text'], train['Sentiment']

Each DataFrame has every column of the clean CSV plus `text` (Summary and Text
joined by a space, missing values as empty strings) and `split`. Labels are in
`Sentiment` (negative / neutral / positive). Never use `Score` as a feature.

From a notebook in notebooks/, add the repo root to the path first:

    import sys; sys.path.insert(0, '..')

Build features with make_vectorizers() so every notebook, script, and the app use
the same text cleaning and settings:

    from data_utils import make_vectorizers
    bow, tfidf = make_vectorizers()
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / 'data/processed/group_seed46_clean.csv'
SPLIT = ROOT / 'data/splits/split_seed46.csv'

# Shared feature settings. min_df=2 drops terms seen in only one training review,
# which removes most typos while keeping two-word phrases such as "not good".
NGRAM_RANGE = (1, 2)
MIN_DF = 2

_HTML_TAG = re.compile(r'<[^>]+>')
_NOT_SUFFIX = re.compile(r"n['\u2019]t\b")


def clean_text(text):
    """Lowercase a review, replace HTML tags with spaces, and expand n't to not.

    About a quarter of reviews contain tags such as <br />, which otherwise become
    the token "br". The default tokenizer also turns "don't" into "don" and drops
    the "t", so expanding the suffix keeps the negation as the word "not".
    """
    text = _HTML_TAG.sub(' ', str(text)).lower()
    return _NOT_SUFFIX.sub(' not', text)


def make_vectorizers():
    """Return unfitted (Bag of Words, TF-IDF) vectorizers with the team settings.

    Both use clean_text, unigrams plus bigrams, and min_df=2. TF-IDF also uses
    sublinear term frequency (1 + log count), so a word repeated many times in one
    review does not dominate it. Fit them on training text only.
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    settings = dict(preprocessor=clean_text, ngram_range=NGRAM_RANGE, min_df=MIN_DF)
    return CountVectorizer(**settings), TfidfVectorizer(sublinear_tf=True, **settings)


def load_splits():
    df = pd.read_csv(CLEAN).merge(pd.read_csv(SPLIT), on='Id', how='inner', validate='one_to_one')
    df['text'] = df['Summary'].fillna('') + ' ' + df['Text'].fillna('')
    return tuple(df[df['split'] == name].reset_index(drop=True) for name in ('train', 'val', 'test'))
