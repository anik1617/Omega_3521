"""Streamlit demo for Group Omega's Amazon food review sentiment models.

Run from the repository root:

    streamlit run app.py
"""
import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
# Saved vectorizers call data_utils.clean_text, so the repo root must be importable.
sys.path.insert(0, str(ROOT))
LABELS = ['negative', 'neutral', 'positive']
RESULTS_DIR = ROOT / 'report/model_results'
FEATURE_NAMES = {'bag_of_words': 'BoW', 'tfidf': 'TF-IDF', 'distilbert_tokens': 'WordPiece tokens'}
# DistilBERT reads at most this many tokens, the same cut used in training.
DISTILBERT_MAX_LENGTH = 256
EXAMPLES = {
    'Positive': ('Best coffee ever', 'Smooth, rich flavor and no bitterness. I order it every month.'),
    'Neutral': ('It is okay', 'Tastes fine but nothing special. Probably will not buy again at this price.'),
    'Negative': ('Stale and broken', 'Arrived crushed and tasted stale. Very disappointed.'),
}


@st.cache_resource
def load_pipeline(model_path):
    pipeline = joblib.load(ROOT / model_path)
    assert list(pipeline.classes_) == LABELS
    return pipeline


@st.cache_data
def load_results():
    """Discover saved models and the training-set majority baseline."""
    models = {}
    baseline = None
    for path in sorted(RESULTS_DIR.glob('*.json')):
        results = json.loads(path.read_text(encoding='utf-8'))
        if baseline is None and results.get('test_models') and 'majority_baseline' in results:
            baseline = results['majority_baseline']
        model_name = results.get('model', 'Multinomial Naive Bayes')
        for features, saved in results.get('test_models', {}).items():
            required = ('model_path', 'validation_macro_f1', 'classification_report', 'confusion_matrix')
            if not all(key in saved for key in required) or not (ROOT / saved['model_path']).exists():
                continue
            if saved['model_path'].endswith('.pt'):
                continue  # raw PyTorch weights from train_torch_models.py; not served in the app
            weights = ROOT / saved['model_path'] / 'model.safetensors'
            if (ROOT / saved['model_path']).is_dir() and (not weights.exists() or weights.stat().st_size < 1_000_000):
                continue  # Git LFS file not downloaded yet (run `git lfs pull`)
            name = f'{model_name} ({FEATURE_NAMES.get(features, features)})'
            models[name] = {
                'model_path': saved['model_path'],
                'validation_macro_f1': saved['validation_macro_f1'],
                'report': saved['classification_report'],
                'confusion_matrix': saved['confusion_matrix'],
            }
    return models, baseline


@st.cache_resource
def load_transformer(model_path):
    """Load a fine-tuned DistilBERT folder saved by scripts/train_distilbert.py."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ROOT / model_path)
    # Weights are stored in float16 to fit GitHub; run in float32, which every CPU supports.
    model = AutoModelForSequenceClassification.from_pretrained(ROOT / model_path, dtype=torch.float32).eval()
    assert [model.config.id2label[i] for i in range(len(LABELS))] == LABELS
    return tokenizer, model


def transformer_predict(model_path, text):
    import torch

    tokenizer, model = load_transformer(model_path)
    inputs = tokenizer(text, truncation=True, max_length=DISTILBERT_MAX_LENGTH, return_tensors='pt')
    with torch.no_grad():
        return torch.softmax(model(**inputs).logits[0], dim=0).numpy()


@st.cache_data
def load_data_overview():
    stats = json.loads((ROOT / 'report/dataset_summary/statistics.json').read_text(encoding='utf-8'))
    counts = pd.read_csv(ROOT / 'report/dataset_summary/class_counts.csv')
    split_counts = pd.read_csv(ROOT / 'data/splits/split_seed46.csv', usecols=['split'])['split'].value_counts()
    return stats, counts, split_counts


def model_probabilities(name, text, models):
    model_path = models[name]['model_path']
    if (ROOT / model_path).is_dir():
        probabilities = transformer_predict(model_path, text)
    else:
        pipeline = load_pipeline(model_path)
        probabilities = pipeline.predict_proba([text])[0]
    return {label: float(probability) for label, probability in zip(LABELS, probabilities)}


def load_example():
    summary, body = EXAMPLES.get(st.session_state.example, ('', ''))
    st.session_state.review_summary = summary
    st.session_state.review_body = body


st.set_page_config(page_title='Omega Review Sentiment', layout='wide')
st.title('Amazon Food Review Sentiment')
st.caption('CSE 3521 · Group Omega')
models, baseline = load_results()
best_name = max(models, key=lambda name: models[name]['validation_macro_f1']) if models else None
predict_tab, comparison_tab, overview_tab = st.tabs(['Predict', 'Model Comparison', 'Data Overview'])

with predict_tab:
    input_col, result_col = st.columns(2, gap='large')

    with input_col:
        st.subheader('Predict a review')
        st.caption('Both saved models receive the same review. Summary and review text are joined as in training.')
        st.session_state.setdefault('review_summary', '')
        st.session_state.setdefault('review_body', '')
        st.selectbox('Load an example', ['None', *EXAMPLES], key='example', on_change=load_example)
        summary = st.text_input('Review summary (optional)', key='review_summary')
        body = st.text_area('Review text', key='review_body', height=180,
                            placeholder='Paste or type a food review...')
        review_text = f'{summary} {body}'.strip()

        if len(models) < 2:
            st.error('At least two saved models are needed. Check models/ and report/model_results/.')
            first_name = second_name = None
        else:
            names = list(models)
            first_default = 'Logistic Regression (TF-IDF)'
            first_default = first_default if first_default in models else best_name
            first_name = st.selectbox('First model', names, index=names.index(first_default))
            second_choices = [name for name in names if name != first_name]
            second_default = 'Multinomial Naive Bayes (BoW)'
            if second_default not in second_choices:
                second_default = second_choices[0]
            second_name = st.selectbox('Second model', second_choices,
                                       index=second_choices.index(second_default))

    with result_col:
        st.subheader('Predictions')
        if len(models) < 2:
            st.info('Predictions appear here once two models are available.')
        elif not review_text:
            st.info('Enter or load a review to see both predictions.')
        else:
            for name in (first_name, second_name):
                with st.container(border=True):
                    probabilities = model_probabilities(name, review_text, models)
                    prediction = max(probabilities, key=probabilities.get)
                    st.subheader(name)
                    st.metric('Predicted sentiment', prediction.title())
                    st.caption(f'Highest class probability: {probabilities[prediction]:.1%}')
                    for label in LABELS:
                        st.progress(probabilities[label], text=f'{label.title()}: {probabilities[label]:.1%}')

with comparison_tab:
    st.subheader('Model comparison')
    if not models:
        st.info('No saved models found in report/model_results/.')
    else:
        best = models[best_name]
        st.success(f'Best on validation macro-F1: **{best_name}** ({best["validation_macro_f1"]:.3f}).')
        rows = []
        for name, info in models.items():
            report = info['report']
            rows.append({'Model': name, 'Validation macro-F1': info['validation_macro_f1'],
                         'Test accuracy': report['accuracy'],
                         'Test macro precision': report['macro avg']['precision'],
                         'Test macro recall': report['macro avg']['recall'],
                         'Test macro-F1': report['macro avg']['f1-score'],
                         'Neutral F1': report['neutral']['f1-score']})
        if baseline:
            report = baseline['classification_report']
            rows.append({'Model': f'Majority-class baseline ({baseline["predicted_class"]})',
                         'Validation macro-F1': None, 'Test accuracy': report['accuracy'],
                         'Test macro precision': report['macro avg']['precision'],
                         'Test macro recall': report['macro avg']['recall'],
                         'Test macro-F1': report['macro avg']['f1-score'],
                         'Neutral F1': report['neutral']['f1-score']})
        table = pd.DataFrame(rows).sort_values('Validation macro-F1', ascending=False, na_position='last')
        st.dataframe(table.style.format({column: '{:.3f}' for column in table.columns if column != 'Model'},
                                        na_rep='—'), hide_index=True, width='stretch')
        st.caption('The best model is selected using validation macro-F1. All other scores use the test set.')

        choices = [*models, *(['Majority-class baseline'] if baseline else [])]
        selected = st.selectbox('Inspect test results for', choices)
        selected_report = baseline['classification_report'] if selected == 'Majority-class baseline' else models[selected]['report']
        selected_matrix = baseline['confusion_matrix'] if selected == 'Majority-class baseline' else models[selected]['confusion_matrix']
        st.markdown('**Per-class test scores**')
        class_table = pd.DataFrame([
            {'Sentiment': label.title(), 'Precision': selected_report[label]['precision'],
             'Recall': selected_report[label]['recall'], 'F1': selected_report[label]['f1-score'],
             'Support': selected_report[label]['support']} for label in LABELS
        ])
        st.dataframe(class_table.style.format({'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1': '{:.3f}'}),
                     hide_index=True, width='stretch')
        st.markdown('**Confusion matrix**')
        matrix = pd.DataFrame(selected_matrix, index=[f'Actual {label}' for label in LABELS],
                              columns=[f'Predicted {label}' for label in LABELS])
        st.dataframe(matrix, width='stretch')

with overview_tab:
    st.subheader('Data overview')
    stats, counts, split_counts = load_data_overview()
    sample_metric, clean_metric, removed_metric = st.columns(3)
    sample_metric.metric('Sampled reviews', f'{stats["sample_rows"]:,}')
    clean_metric.metric('Clean reviews', f'{stats["clean_rows"]:,}')
    removed_metric.metric('Removed during cleaning', f'{stats["sample_rows"] - stats["clean_rows"]:,}')
    st.caption('The sample is drawn with seed 46. Duplicate and unusable reviews are removed before splitting.')
    st.markdown('**Clean review sentiments**')
    clean_counts = counts[counts['stage'] == 'clean'].set_index('sentiment').reindex(LABELS)
    sentiment_chart = (
        alt.Chart(clean_counts.reset_index())
        .mark_bar()
        .encode(
            x=alt.X('sentiment:N', sort=LABELS, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y('count:Q', title=None),
        )
    )
    st.altair_chart(sentiment_chart, width='stretch')
    st.dataframe(clean_counts.reset_index().rename(columns={'sentiment': 'Sentiment',
                                                              'count': 'Reviews', 'percent': 'Percent (%)'})
                 [['Sentiment', 'Reviews', 'Percent (%)']], hide_index=True, width='stretch')
    st.markdown('**Shared train / validation / test split**')
    split_table = pd.DataFrame({'Split': ['train', 'validation', 'test'],
                                'Reviews': [int(split_counts.get(name, 0))
                                            for name in ('train', 'val', 'test')]})
    st.dataframe(split_table, hide_index=True, width='stretch')
    st.caption('Labels: 1–2 stars → negative; 3 stars → neutral; 4–5 stars → positive. '
               'The models are fitted on the training split and evaluated on the test split.')
