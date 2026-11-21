"""Streamlit demo for Group Omega's Amazon food review sentiment models.

Run from the repository root:

    streamlit run app.py
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
# Saved vectorizers call data_utils.clean_text, so the repo root must be importable.
sys.path.insert(0, str(ROOT))
LABELS = ['negative', 'neutral', 'positive']
COLORS = {'negative': '#d9534f', 'neutral': '#9e9e9e', 'positive': '#5cb85c'}
RESULTS_DIR = ROOT / 'report/model_results'
FEATURE_NAMES = {'bag_of_words': 'BoW', 'tfidf': 'TF-IDF'}


def placeholder_predict(model_name, text):
    """Fake class probabilities, deterministic for a given model and text."""
    seed = int(hashlib.md5(f'{model_name}|{text}'.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).dirichlet(np.ones(len(LABELS)))


@st.cache_resource
def load_pipeline(model_path):
    pipeline = joblib.load(ROOT / model_path)
    assert list(pipeline.classes_) == LABELS
    return pipeline


@st.cache_data
def trained_models():
    """Find every saved model listed in report/model_results/*.json.

    A training script registers a model by writing a results file with a
    `model` name and a `test_models` dict whose entries have `model_path`,
    `validation_macro_f1`, and a scikit-learn `classification_report`.
    """
    found = {}
    for path in sorted(RESULTS_DIR.glob('*.json')):
        results = json.loads(path.read_text(encoding='utf-8'))
        model_name = results.get('model', 'Multinomial Naive Bayes')
        for features, saved in results.get('test_models', {}).items():
            if 'model_path' not in saved or not (ROOT / saved['model_path']).exists():
                continue
            report = saved['classification_report']
            name = f'{model_name} ({FEATURE_NAMES.get(features, features)})'
            found[name] = {
                'model_path': saved['model_path'],
                'Validation macro-F1': saved['validation_macro_f1'],
                'Test accuracy': report['accuracy'],
                'Test macro-F1': report['macro avg']['f1-score'],
                'Neutral F1': report['neutral']['f1-score'],
            }
    return found


def real_predictor(model_path):
    return lambda text: load_pipeline(model_path).predict_proba([text])[0]


TRAINED = trained_models()
# The best model is chosen on validation macro-F1, the same rule used in training,
# so the test set is not used to pick it.
BEST = max(TRAINED, key=lambda name: TRAINED[name]['Validation macro-F1']) if TRAINED else None

# Display name -> function(text) returning probabilities in LABELS order.
# Trained models are loaded automatically; placeholders mark models not built yet.
MODELS = {name: real_predictor(info['model_path']) for name, info in TRAINED.items()}
PLACEHOLDERS = {
    'Logistic Regression (TF-IDF)': 'logreg',
    'Linear SVM (TF-IDF)': 'svm',
    'Transformer (DistilBERT)': 'bert',
}
for name, key in PLACEHOLDERS.items():
    MODELS.setdefault(name, lambda text, key=key: placeholder_predict(key, text))
PLACEHOLDER_NAMES = set(MODELS) - set(TRAINED)


def display_name(name):
    if name == BEST:
        return f'★ {name} (best)'
    return f'{name} (placeholder)' if name in PLACEHOLDER_NAMES else name

EXAMPLES = {
    'Positive': ('Best coffee ever', 'Smooth, rich flavor and no bitterness. I order it every month.'),
    'Neutral': ('It is okay', 'Tastes fine but nothing special. Probably will not buy again at this price.'),
    'Negative': ('Stale and broken', 'Arrived crushed and tasted stale. Very disappointed.'),
}


def probability_chart(probs):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    bars = ax.barh(LABELS, probs, color=[COLORS[label] for label in LABELS])
    ax.bar_label(bars, labels=[f'{p:.1%}' for p in probs], padding=3)
    ax.set_xlim(0, 1.15)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


st.set_page_config(page_title='Omega Review Sentiment', layout='wide')
st.title('Amazon Food Review Sentiment')
st.caption('CSE 3521 · Group Omega')
if BEST:
    best = TRAINED[BEST]
    st.success(f'**Best model: {BEST}**, chosen by validation macro-F1 '
               f'({best["Validation macro-F1"]:.3f}). Test accuracy {best["Test accuracy"]:.1%}, '
               f'test macro-F1 {best["Test macro-F1"]:.3f}.')
if PLACEHOLDER_NAMES:
    st.warning('Models marked (placeholder) are not trained yet and return random predictions: '
               + ', '.join(sorted(PLACEHOLDER_NAMES)) + '.')

with st.expander('Model performance on the test set', expanded=False):
    if TRAINED:
        table = pd.DataFrame([{'Model': display_name(name),
                               **{k: v for k, v in info.items() if k != 'model_path'}}
                              for name, info in TRAINED.items()])
        table = table.sort_values('Validation macro-F1', ascending=False)
        metric_columns = ['Validation macro-F1', 'Test macro-F1', 'Neutral F1']
        st.dataframe(table.style.format({'Test accuracy': '{:.1%}',
                                         **{c: '{:.3f}' for c in metric_columns}}),
                     hide_index=True, width='stretch')
        st.caption('The best model is picked on validation data; test scores are reported once.')
    else:
        st.info('No trained models found in report/model_results/.')

left, right = st.columns(2, gap='large')

with left:
    st.subheader('Input')
    options = ([BEST] if BEST else []) + [name for name in MODELS if name != BEST]
    model_name = st.selectbox('Model', options, format_func=display_name)
    compare = st.checkbox('Compare all models')

    example = st.radio('Load an example', ['None', *EXAMPLES], horizontal=True)
    summary_default, text_default = EXAMPLES.get(example, ('', ''))
    summary = st.text_input('Review summary (optional)', value=summary_default)
    body = st.text_area('Review text', value=text_default, height=220,
                        placeholder='Paste or type a food review...')

# Same input format as training: Summary and Text joined by a space.
text = f'{summary} {body}'.strip()

with right:
    st.subheader('Prediction')
    if not text:
        st.info('Enter a review on the left to see the prediction.')
    else:
        probs = MODELS[model_name](text)
        label = LABELS[int(np.argmax(probs))]

        st.markdown(
            f'<div style="padding:0.8rem 1rem;border-radius:0.5rem;background:{COLORS[label]};'
            f'color:white;font-size:1.4rem;font-weight:600;">{label.capitalize()}</div>',
            unsafe_allow_html=True,
        )
        st.metric('Confidence', f'{probs.max():.1%}')
        st.pyplot(probability_chart(probs), clear_figure=True)

        if compare:
            st.markdown('**All models**')
            rows = []
            for name, predict in MODELS.items():
                p = predict(text)
                rows.append({'Model': display_name(name), 'Prediction': LABELS[int(np.argmax(p))],
                             **dict(zip(LABELS, p))})
            st.dataframe(
                pd.DataFrame(rows).style.format({label: '{:.1%}' for label in LABELS}),
                hide_index=True, width='stretch',
            )
