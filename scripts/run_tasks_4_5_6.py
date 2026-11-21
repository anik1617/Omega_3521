"""Generate Milestone 2 Tasks 4-6 from the shared split and saved pipeline.

Run: python scripts/run_tasks_4_5_6.py
No fitting, resampling, or split changes are performed. The script loads the
pipelines saved by scripts/train_naive_bayes.py and checks them against that
script's metrics file, so it keeps working when the models are retrained.
"""
from pathlib import Path
import json
import sys
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils import load_splits

HERE = ROOT / 'report/milestone2'
RESULTS_PATH = ROOT / 'report/model_results/naive_bayes_sophiya_features_seed46.json'
LABELS = ['negative', 'neutral', 'positive']
FEATURE_NAMES = {'bag_of_words': 'Bag of Words', 'tfidf': 'TF-IDF'}
MIN_TERM_COUNT = 150  # rank only terms with at least this many training occurrences


def top_terms(model, terms, counts, feature_name):
    """Rank terms by log P(term | class) minus the largest other-class value."""
    logp = model.feature_log_prob_
    eligible = np.where(counts >= MIN_TERM_COUNT)[0]
    rows = []
    for target in LABELS:
        idx = list(model.classes_).index(target)
        other_idx = [i for i in range(len(model.classes_)) if i != idx]
        contrast = logp[idx] - np.max(logp[other_idx], axis=0)
        chosen = eligible[np.argsort(contrast[eligible])[-10:][::-1]]
        for rank, col in enumerate(chosen, 1):
            rows.append({'features': feature_name, 'class': target, 'rank': rank,
                         'term': str(terms[col]),
                         'log_probability_difference': round(float(contrast[col]), 4),
                         'training_term_count': int(counts[col])})
    return rows


def main():
    train, val, test = load_splits()
    sizes = [len(train), len(val), len(test)]
    print('Rows:', sizes)
    assert sizes == [30123, 6952, 9269]
    print('Test classes:', test['Sentiment'].value_counts().to_dict())
    assert test['Sentiment'].value_counts().to_dict() == {'positive':7223,'negative':1367,'neutral':679}

    assert not (set(train.Id) & set(val.Id) or set(train.Id) & set(test.Id)
                or set(val.Id) & set(test.Id))
    results = json.loads(RESULTS_PATH.read_text(encoding='utf-8'))
    best_features = results['selected']['features']
    HERE.mkdir(parents=True, exist_ok=True)
    loaded, rows = {}, []
    for feature_name, saved in results['test_models'].items():
        pipeline = joblib.load(ROOT / saved['model_path'])
        vectorizer = pipeline.named_steps['vectorizer']
        model = pipeline.named_steps['classifier']
        assert isinstance(model, MultinomialNB) and model.alpha == saved['alpha']
        assert len(vectorizer.vocabulary_) == results['vocabulary_sizes'][feature_name]
        xtrain = vectorizer.transform(train['text'])
        # Verify the artifact's training counts against the shared training rows.
        assert np.array_equal(model.class_count_, [sum(train.Sentiment == c) for c in model.classes_])
        for i, label in enumerate(model.classes_):
            counts = np.asarray(xtrain[(train.Sentiment == label).to_numpy()].sum(axis=0)).ravel()
            assert np.allclose(counts, model.feature_count_[i])
        pred = model.predict(vectorizer.transform(test['text']))
        accuracy = accuracy_score(test['Sentiment'], pred)
        macro_f1 = f1_score(test['Sentiment'], pred, average='macro', labels=LABELS)
        cm = confusion_matrix(test['Sentiment'], pred, labels=LABELS)
        # The saved pipeline must reproduce the metrics the training script reported.
        assert np.isclose(accuracy, saved['classification_report']['accuracy'], rtol=0, atol=1e-12)
        assert np.isclose(macro_f1, saved['classification_report']['macro avg']['f1-score'],
                          rtol=0, atol=1e-12)
        assert cm.tolist() == saved['confusion_matrix']
        print(f"{FEATURE_NAMES[feature_name]}: vocabulary={len(vectorizer.vocabulary_)}, "
              f"alpha={model.alpha}, fit_prior={model.fit_prior}, "
              f"accuracy={accuracy:.4f}, macro F1={macro_f1:.4f}")
        print('Confusion matrix:\n', cm)

        # For Task 4: compare learned log likelihoods P(term | class) for every
        # class. Term counts are raw training occurrences for both representations.
        counter = CountVectorizer(vocabulary=vectorizer.vocabulary_,
                                  preprocessor=vectorizer.preprocessor,
                                  ngram_range=vectorizer.ngram_range)
        counts = np.asarray(counter.transform(train['text']).sum(axis=0)).ravel()
        rows += top_terms(model, vectorizer.get_feature_names_out(), counts, feature_name)
        loaded[feature_name] = dict(vectorizer=vectorizer, model=model, pred=pred,
                                    accuracy=accuracy, macro_f1=macro_f1, cm=cm)
    top = pd.DataFrame(rows)
    top.to_csv(HERE/'task4_top_model_terms.csv', index=False)

    # Tasks 5 and 6 use the model selected on validation macro-F1.
    vectorizer, model, pred = (loaded[best_features][k] for k in ('vectorizer', 'model', 'pred'))
    accuracy, macro_f1, cm = (loaded[best_features][k] for k in ('accuracy', 'macro_f1', 'cm'))
    xtrain = vectorizer.transform(train['text'])
    top = top[top['features'] == best_features]

    # For Task 5: show document frequency by class for selected model-informative terms.
    # Use a binary counter so every chart term (including bigrams) is counted once per review.
    # Frequency is % reviews within a class containing a given word at least once.
    chart_terms = (top.query('rank <= 6')['term'].tolist())
    chart_columns = [vectorizer.vocabulary_[t] for t in chart_terms]
    xterms = (xtrain[:, chart_columns] > 0).astype(int)
    stats = []
    ytrain = train['Sentiment'].to_numpy()
    for i, word in enumerate(chart_terms):
        for label in LABELS:
            subset = xterms[ytrain == label, i]
            percent = 100 * subset.getnnz() / (ytrain == label).sum()
            stats.append({'term': word, 'sentiment': label, 'percent_of_reviews': round(float(percent), 2)})
    frequency = pd.DataFrame(stats)
    frequency.to_csv(HERE/'task5_word_frequency.csv', index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, layout='constrained')
    for ax, target in zip(axes, LABELS):
        words = top[(top['class']==target) & (top['rank']<=6)]['term'].tolist()
        y = np.arange(len(words))
        offsets = [-.23, 0, .23]
        colors = ['#ad5754','#9e9e9e','#39846d']
        for label, off, color in zip(LABELS, offsets, colors):
            v = [frequency[(frequency.term==w)&(frequency.sentiment==label)]['percent_of_reviews'].iloc[0] for w in words]
            ax.barh(y+off, v, height=.22, label=label.title(), color=color)
        ax.set_yticks(y, words)
        ax.invert_yaxis()
        ax.set_title(f'Terms associated with {target} reviews')
        ax.set_xlabel('Reviews containing term (%)')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    axes[2].legend(frameon=False, loc='lower right')
    fig.suptitle('Selected words in training reviews, by sentiment', fontsize=13)
    fig.savefig(HERE/'task5_feature_chart.png', dpi=300)
    plt.close(fig)

    # For Task 6: choose 8 wrong predictions, spread over real/predicted combinations.
    errors = test[['Id','Score','Sentiment','Summary','Text']].copy()
    errors['predicted'] = pred
    errors = errors[errors['Sentiment'] != errors['predicted']].copy()
    errors['words'] = errors['Text'].fillna('').str.split().str.len()
    print('Total test errors:',len(errors))
    assert len(errors) == int((cm.sum() - np.trace(cm)))
    # One to two examples from different pairs: deterministic choice sorted by Id.
    selected = []
    pairs = [('negative','positive'), ('positive','negative'), ('neutral','positive'),
             ('positive','neutral'), ('negative','neutral'), ('neutral','negative')]
    for actual, predicted in pairs:
        options = errors[(errors['Sentiment']==actual)&(errors['predicted']==predicted)]
        if len(options):
            selected.append(options.sort_values('Id').iloc[0])
    # Complete up to 8 using shortest remaining reviews; break ties by Id.
    used = {int(x['Id']) for x in selected}
    remaining = errors[~errors.Id.isin(used)].sort_values(['words','Id'])
    for _, item in remaining.iterrows():
        if len(selected)>=8:break
        selected.append(item)
    chosen = pd.DataFrame(selected)
    assert len(chosen) == 8 and chosen.Id.nunique() == 8
    assert (chosen.Sentiment != chosen.predicted).all()
    chosen['review_excerpt'] = chosen['Text'].fillna('').str.replace(r'\s+',' ',regex=True).str.slice(0,350)
    chosen[['Id','Score','Sentiment','predicted','Summary','words','review_excerpt','Text']].to_csv(HERE/'task6_eight_errors.csv',index=False)
    pair_counts = errors.groupby(['Sentiment','predicted']).size().rename('count').reset_index()
    pair_counts.to_csv(HERE/'task6_error_counts.csv',index=False)
    print('Error pairs:',pair_counts.to_dict('records'))
    print('Example errors:',chosen[['Id','Score','Sentiment','predicted','Summary','words','review_excerpt']].to_string(index=False,max_colwidth=100))

    summary={'source':'data_utils.load_splits(): committed seed-46 dataset and split',
      'model_path':results['test_models'][best_features]['model_path'],
      'model':'MultinomialNB', 'features':FEATURE_NAMES[best_features], 'alpha':float(model.alpha),
      'fit_prior':bool(model.fit_prior),
      'note':'Saved pipelines loaded without fitting; training feature counts and test metrics verified against the training script output',
      'task4_representations':[FEATURE_NAMES[f] for f in results['test_models']],
      'rows':dict(zip(['train','validation','test'],sizes)),
      'vocabulary':len(vectorizer.vocabulary_), 'test_accuracy':float(accuracy),
      'test_macro_f1':float(macro_f1), 'labels':LABELS,
      'confusion_matrix_axes':'rows=actual, columns=predicted', 'confusion_matrix':cm.tolist(),
      'test_errors':len(errors), 'top_features_method':f'for each representation and class: model class log probability minus maximum other-class log probability; minimum {MIN_TERM_COUNT} training occurrences',
      'chart_method':'class-normalized document frequency for the top six terms of each class in the selected model (training only)'}
    (HERE/'verification.json').write_text(json.dumps(summary,indent=2)+'\n', encoding='utf-8')
    print('Wrote output files to:',HERE)


if __name__=='__main__':
    main()
