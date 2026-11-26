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
# One muted color per class, reused by every chart in the app.
COLORS = {'negative': '#b4533a', 'neutral': '#9a8f6f', 'positive': '#3f7a5c'}
CLASS_SCALE = alt.Scale(domain=LABELS, range=[COLORS[label] for label in LABELS])


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


def probability_chart(probabilities):
    frame = pd.DataFrame({'sentiment': LABELS, 'probability': [probabilities[label] for label in LABELS]})
    bars = alt.Chart(frame).encode(
        y=alt.Y('sentiment:N', sort=LABELS, title=None,
                axis=alt.Axis(labelAngle=0, ticks=False, domain=False, labelOverlap=False, labelFontSize=12)),
        x=alt.X('probability:Q', scale=alt.Scale(domain=[0, 1.12]), title=None, axis=None),
    )
    chart = bars.mark_bar(height=16, cornerRadiusEnd=2).encode(
        color=alt.Color('sentiment:N', scale=CLASS_SCALE, legend=None))
    labels = bars.mark_text(align='left', dx=4, fontSize=12).encode(text=alt.Text('probability:Q', format='.0%'))
    return (chart + labels).properties(height=96)


def confusion_chart(matrix):
    # Shade by share of each actual class, so the small neutral row is as readable as the large positive one.
    cells = pd.DataFrame([{'actual': actual, 'predicted': predicted, 'count': matrix[i][j],
                           'share': matrix[i][j] / sum(matrix[i])}
                          for i, actual in enumerate(LABELS) for j, predicted in enumerate(LABELS)])
    base = alt.Chart(cells).encode(
        x=alt.X('predicted:N', sort=LABELS, title='Predicted', axis=alt.Axis(labelAngle=0, orient='top')),
        y=alt.Y('actual:N', sort=LABELS, title='Actual'),
    )
    heat = base.mark_rect().encode(
        color=alt.Color('share:Q', scale=alt.Scale(domain=[0, 1], range=['#f1efe9', '#2f5d50']), legend=None))
    text = base.mark_text(fontSize=14).encode(
        text='count:Q',
        color=alt.condition(alt.datum.share > 0.5, alt.value('white'), alt.value('#222')))
    return (heat + text).properties(height=240)


st.set_page_config(page_title='Food Review Sentiment', layout='wide')
st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2.2rem;}
h1 {font-size: 2rem !important; margin-bottom: 0 !important;}
.verdict {font-size: 1.6rem; font-weight: 600; margin: 0.1rem 0 0.2rem;}
.model-name {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #6b6b6b;}
.agree {color: #6b6b6b; font-size: 0.95rem; margin-top: 0.5rem;}
</style>
""", unsafe_allow_html=True)

models, baseline = load_results()
best_name = max(models, key=lambda name: models[name]['validation_macro_f1']) if models else None

st.title('Food review sentiment')
st.markdown(f'CSE 3521, Group Omega. {len(models)} models trained on Amazon Fine Food reviews, '
            'each labeling a review as negative, neutral or positive.')
predict_tab, comparison_tab, overview_tab = st.tabs(['Predict', 'Model Comparison', 'Data Overview'])

with predict_tab:
    input_col, result_col = st.columns([1.1, 1], gap='large')

    with input_col:
        st.session_state.setdefault('review_summary', '')
        st.session_state.setdefault('review_body', '')
        st.selectbox('Example', ['None', *EXAMPLES], key='example', on_change=load_example)
        summary = st.text_input('Summary', key='review_summary', placeholder='Optional, like a review title')
        body = st.text_area('Review', key='review_body', height=170, placeholder='Type or paste a review')
        review_text = f'{summary} {body}'.strip()

        if len(models) < 2:
            st.error('Need at least two saved models. Check models/ and report/model_results/.')
            first_name = second_name = None
        else:
            names = list(models)
            first_default = 'Logistic Regression (TF-IDF)'
            first_default = first_default if first_default in models else best_name
            first_name = st.selectbox('Model A', names, index=names.index(first_default))
            second_choices = [name for name in names if name != first_name]
            second_default = 'Multinomial Naive Bayes (BoW)'
            if second_default not in second_choices:
                second_default = second_choices[0]
            second_name = st.selectbox('Model B', second_choices,
                                       index=second_choices.index(second_default))

    with result_col:
        if len(models) < 2:
            pass
        elif not review_text:
            st.caption('Predictions show up here once there is some review text.')
        else:
            verdicts = []
            for name in (first_name, second_name):
                probabilities = model_probabilities(name, review_text, models)
                prediction = max(probabilities, key=probabilities.get)
                verdicts.append(prediction)
                with st.container(border=True):
                    st.markdown(f'<div class="model-name">{name}</div>'
                                f'<div class="verdict" style="color:{COLORS[prediction]}">{prediction.title()}</div>',
                                unsafe_allow_html=True)
                    st.altair_chart(probability_chart(probabilities), width='stretch')
            if verdicts[0] == verdicts[1]:
                st.markdown(f'<div class="agree">Both models say {verdicts[0]}.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="agree">The models disagree: {verdicts[0]} vs. {verdicts[1]}.</div>',
                            unsafe_allow_html=True)

with comparison_tab:
    if not models:
        st.caption('No saved models found in report/model_results/.')
    else:
        best = models[best_name]
        st.markdown(f'{best_name} has the highest validation macro-F1 ({best["validation_macro_f1"]:.3f}). '
                    'We pick models on validation; every other column is the test set.')
        rows = []
        for name, info in models.items():
            report = info['report']
            rows.append({'Model': name, 'Val macro-F1': info['validation_macro_f1'],
                         'Accuracy': report['accuracy'],
                         'Precision': report['macro avg']['precision'],
                         'Recall': report['macro avg']['recall'],
                         'Macro-F1': report['macro avg']['f1-score'],
                         'Neutral F1': report['neutral']['f1-score']})
        if baseline:
            report = baseline['classification_report']
            rows.append({'Model': f'Majority class ({baseline["predicted_class"]})',
                         'Val macro-F1': None, 'Accuracy': report['accuracy'],
                         'Precision': report['macro avg']['precision'],
                         'Recall': report['macro avg']['recall'],
                         'Macro-F1': report['macro avg']['f1-score'],
                         'Neutral F1': report['neutral']['f1-score']})
        table = pd.DataFrame(rows).astype({'Val macro-F1': float}).sort_values('Val macro-F1', ascending=False, na_position='last')
        # Format as text so the baseline's missing validation score shows as a dash, not "None".
        for column in table.columns[1:]:
            table[column] = table[column].map(lambda value: '–' if pd.isna(value) else f'{value:.3f}')
        st.dataframe(
            table.style.set_properties(subset=table.columns[1:], **{'text-align': 'right'})
            .apply(lambda row: ['font-weight: 600' if row['Model'] == best_name else '' for _ in row], axis=1),
            hide_index=True, width='stretch')
        st.caption('Precision, recall and F1 are macro averages over the three classes.')

        st.markdown('#### Per-class results')
        choices = [*models, *(['Majority class'] if baseline else [])]
        selected = st.selectbox('Model', choices, label_visibility='collapsed')
        is_baseline = selected == 'Majority class'
        selected_report = baseline['classification_report'] if is_baseline else models[selected]['report']
        selected_matrix = baseline['confusion_matrix'] if is_baseline else models[selected]['confusion_matrix']
        class_col, matrix_col = st.columns([1, 1], gap='large')
        with class_col:
            class_table = pd.DataFrame([
                {'Class': label, 'Precision': selected_report[label]['precision'],
                 'Recall': selected_report[label]['recall'], 'F1': selected_report[label]['f1-score'],
                 'Reviews': int(selected_report[label]['support'])} for label in LABELS
            ])
            st.dataframe(class_table.style.format({'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1': '{:.3f}'}),
                         hide_index=True, width='stretch')
            st.caption('Test set scores for each class. Matrix cells are shaded by the share of each actual class.')
        with matrix_col:
            st.altair_chart(confusion_chart(selected_matrix), width='stretch')

with overview_tab:
    stats, counts, split_counts = load_data_overview()
    sample_metric, clean_metric, removed_metric = st.columns(3)
    sample_metric.metric('Sampled', f'{stats["sample_rows"]:,}')
    clean_metric.metric('After cleaning', f'{stats["clean_rows"]:,}')
    removed_metric.metric('Removed', f'{stats["sample_rows"] - stats["clean_rows"]:,}')
    st.caption('A 50,000 review sample drawn with seed 46. Duplicates and unusable rows are dropped before '
               'splitting. 1 to 2 stars is negative, 3 is neutral, 4 to 5 is positive.')

    chart_col, split_col = st.columns([1.4, 1], gap='large')
    clean_counts = counts[counts['stage'] == 'clean'].set_index('sentiment').reindex(LABELS).reset_index()
    with chart_col:
        st.markdown('#### Sentiment after cleaning')
        bars = alt.Chart(clean_counts).encode(
            x=alt.X('sentiment:N', sort=LABELS, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y('count:Q', title=None, axis=alt.Axis(grid=False)),
        )
        sentiment_chart = (
            bars.mark_bar(width=60).encode(color=alt.Color('sentiment:N', scale=CLASS_SCALE, legend=None))
            + bars.transform_calculate(label="format(datum.percent, '.1f') + '%'").mark_text(dy=-8).encode(text='label:N')
        ).properties(height=280)
        st.altair_chart(sentiment_chart, width='stretch')
        st.caption('Most reviews are positive, so neutral is the hard class.')
    with split_col:
        st.markdown('#### Train, validation, test')
        split_table = pd.DataFrame({'Split': ['Train', 'Validation', 'Test'],
                                    'Reviews': [int(split_counts.get(name, 0)) for name in ('train', 'val', 'test')]})
        split_table['Share'] = split_table['Reviews'] / split_table['Reviews'].sum()
        st.dataframe(split_table.style.format({'Reviews': '{:,}', 'Share': '{:.0%}'}),
                     hide_index=True, width='stretch')
        st.caption('Every model uses this same split. Models are fit on train and scored on test.')
