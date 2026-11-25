# Reproducibility (draft for the M3 report)

All code, the seed-46 dataset, and the shared split are in our GitHub repository
(github.com/anik1617/Omega_3521). The cleaned dataset
(data/processed/group_seed46_clean.csv, 46,344 reviews) and the 65/15/20
stratified split (data/splits/split_seed46.csv) are committed, so the models can
be retrained without downloading anything. To rebuild the dataset from the
original source, scripts/setup_data.py downloads the Kaggle Amazon Fine Food
Reviews file and scripts/prepare_sample.py draws our 50,000-review sample with
random_state=46 and assigns the three sentiment labels.

Dependencies are pinned in requirements.txt and installed with
pip install -r requirements.txt on Python 3.11 or newer. Each model has its own
training script (python scripts/train_naive_bayes.py for Naive Bayes, with the
Logistic Regression and neural network commands listed in the README). Every
script fits on the training split, tunes on the validation split, reports on the
test split, saves the fitted vectorizer and model together in models/, and writes
its metrics and training time to report/model_results/. Running
python scripts/compare_models.py then rebuilds the comparison table and the
confusion matrices from those files. The app runs locally with
streamlit run app.py and loads the saved models without retraining. The same
steps are in README.md under "Reproducing our results".
