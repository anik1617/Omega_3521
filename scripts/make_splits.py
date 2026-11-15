"""Create the team's train/validation/test split (65/15/20, stratified, seed 42).

Run once from the repository root:
    python scripts/make_splits.py

Writes data/splits/split_seed42.csv with one row per review: Id, split.
The split is committed to Git so everyone trains and tests on the same rows.
Do not rerun it to make a new split; load it with data_utils.load_splits().
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / 'data/processed/group_seed46_clean.csv'
OUT = ROOT / 'data/splits/split_seed42.csv'
SEED = 42
TEST, VAL = 0.20, 0.15


def main():
    df = pd.read_csv(CLEAN, usecols=['Id', 'Sentiment'])
    if df['Id'].duplicated().any():
        raise ValueError('Duplicate Ids in the clean CSV')
    # The clean CSV already keeps one copy of each repeated review (see
    # prepare_sample.dedup_key), so a repeat cannot end up in two splits.
    rest, test = train_test_split(df, test_size=TEST, stratify=df['Sentiment'], random_state=SEED)
    # 15% of the whole dataset is 0.15 / 0.80 of what is left after the test split.
    train, val = train_test_split(rest, test_size=VAL / (1 - TEST), stratify=rest['Sentiment'],
                                  random_state=SEED)
    split = pd.concat([train.assign(split='train'), val.assign(split='val'), test.assign(split='test')])
    split = split.sort_values('Id')[['Id', 'split']]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(OUT, index=False, lineterminator='\n')

    summary = (df.merge(split, on='Id').groupby(['split', 'Sentiment']).size()
               .unstack(fill_value=0).loc[['train', 'val', 'test']])
    summary['total'] = summary.sum(axis=1)
    print(summary.to_string())
    print(json.dumps({s: round(len(g) / len(df), 4) for s, g in split.groupby('split')}))


if __name__ == '__main__':
    main()
