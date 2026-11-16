"""Load Group Omega's cleaned reviews with the shared train/validation/test split.

    from data_utils import load_splits
    train, val, test = load_splits()
    X_train, y_train = train['text'], train['Sentiment']

Each DataFrame has every column of the clean CSV plus `text` (Summary and Text
joined by a space, missing values as empty strings) and `split`. Labels are in
`Sentiment` (negative / neutral / positive). Never use `Score` as a feature.

From a notebook in notebooks/, add the repo root to the path first:

    import sys; sys.path.insert(0, '..')
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / 'data/processed/group_seed46_clean.csv'
SPLIT = ROOT / 'data/splits/split_seed46.csv'


def load_splits():
    df = pd.read_csv(CLEAN).merge(pd.read_csv(SPLIT), on='Id', how='inner', validate='one_to_one')
    df['text'] = df['Summary'].fillna('') + ' ' + df['Text'].fillna('')
    return tuple(df[df['split'] == name].reset_index(drop=True) for name in ('train', 'val', 'test'))
