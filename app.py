"""Streamlit demo for Group Omega's Amazon food review sentiment models.

Run from the repository root:

    streamlit run app.py
"""
import hashlib
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
LABELS = ['negative', 'neutral', 'positive']
COLORS = {'negative': '#d9534f', 'neutral': '#9e9e9e', 'positive': '#5cb85c'}


def placeholder_predict(model_name, text):
    """Fake class probabilities, deterministic for a given model and text."""
    seed = int(hashlib.md5(f'{model_name}|{text}'.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).dirichlet(np.ones(len(LABELS)))


@st.cache_resource
def load_naive_bayes():
    pipeline = joblib.load(ROOT / 'models/naive_bayes_sophiya_features_seed46.joblib')
    assert list(pipeline.classes_) == LABELS
    return pipeline


# Display name -> function(text) returning probabilities in LABELS order.
# Replace placeholders with real models loaded from models/*.joblib.
MODELS = {
    'Multinomial Naive Bayes (BoW)': lambda text: load_naive_bayes().predict_proba([text])[0],
    'Logistic Regression (TF-IDF)': lambda text: placeholder_predict('logreg', text),
    'Linear SVM (TF-IDF)': lambda text: placeholder_predict('svm', text),
    'Transformer (DistilBERT)': lambda text: placeholder_predict('bert', text),
}

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
st.warning('Only Multinomial Naive Bayes is trained. The other models are placeholders '
           'that return random predictions.')

left, right = st.columns(2, gap='large')

with left:
    st.subheader('Input')
    model_name = st.selectbox('Model', list(MODELS))
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
                rows.append({'Model': name, 'Prediction': LABELS[int(np.argmax(p))],
                             **dict(zip(LABELS, p))})
            st.dataframe(
                pd.DataFrame(rows).style.format({label: '{:.1%}' for label in LABELS}),
                hide_index=True, width='stretch',
            )
