# Milestone 2: Tasks 4-6 (Ethan Brown)

These outputs adapt the supplied `Omega_Milestone2_Tasks4-6_Local.zip` analysis
to the shared repository. They load Shilin's saved pipelines and the committed
seed-46 split through `data_utils.load_splits()`; nothing is resampled or fitted.
Input text is Summary plus Text. Score is used only to display the true rating
in the error examples, never as a feature.

## Outputs and methods

- **Task 4:** `task4_top_model_terms.csv` contains the top ten terms for each
  class (negative, neutral, positive) and each representation (Bag of Words and
  TF-IDF), from the two saved MultinomialNB models. Terms can be single words or
  two-word phrases. For each target class, the ranking score is
  `log P(term | target) - max(log P(term | other class))` using the saved
  classifier's `feature_log_prob_`. Terms must have at least 150 total token
  occurrences across training reviews. Higher scores indicate stronger relative
  association with the target class. This is a model probability contrast,
  not a causal explanation or a per-review contribution.
- **Task 5:** `task5_feature_chart.png` compares the top six terms from each
  Task 4 ranking across all three classes. `task5_word_frequency.csv` stores
  the values: `100 * reviews containing term / reviews in class`. Each review
  counts once per term, so positive-class size and repeated words do not inflate
  the bars. Selection, ranking, and frequencies use training data only.
  The chart uses the selected Bag of Words model, with one panel per class, and
  the shared preprocessing from `data_utils.make_vectorizers()`, matching
  Sophiya's notebook: HTML tags removed, `n't` expanded to `not`, lowercase
  unigrams and bigrams in at least two training reviews, no stop-word removal.
- **Task 6:** `task6_eight_errors.csv` contains eight actual incorrect test
  predictions, with IDs, true ratings/labels, predictions, summaries, excerpts,
  and full Text. One example is selected for each of the six actual/predicted
  error pairs by smallest ID; the last two are the shortest remaining Text
  fields (whitespace word count), breaking ties by ID. The examples come from
  the selected Bag of Words model. `task6_error_counts.csv` counts every test error by pair.
- `verification.json` records split sizes, vocabulary, model configuration,
  metrics, confusion matrix, and methods.

## Reproduce and verify

From the repository root with Python 3.11+:

```bash
python -m pip install -r requirements.txt
python scripts/train_naive_bayes.py   # only if the models need retraining
python scripts/run_tasks_4_5_6.py
```

Outputs are written to `report/milestone2/` regardless of the working directory.
The script checks disjoint split IDs and split sizes, then, for each saved
pipeline, checks the vocabulary size, alpha, and per-class training feature
counts against the shared training rows. It also checks that test accuracy,
macro-F1, and the confusion matrix match the values in
`report/model_results/naive_bayes_sophiya_features_seed46.json`, so it fails if a
saved model and its metrics file disagree. It never calls fit.

## Evidence to review for the error discussion

- Reviews 2806 and 1902 are four-star reviews with both favorable language and
  complaints (a fishy smell, one bitter flavor). Review 6814 is two-star despite
  a "decent tasting" summary. These examples suggest difficulty combining mixed
  assessments into one sentiment label.
- Review 3037 opens with praise ("I love these cookies") before describing
  headaches; it is two-star but predicted positive. Bag of words does not know
  which part of the review decides the rating.
- Reviews 1535 and 295161 are three-star reviews that start with praise ("Not
  bad", "very delicious") and end with a reservation; 1058 is three-star with
  only complaints. Review 560618 ("tasty but a little hotter than I like") is
  four-star but predicted neutral. These illustrate ambiguity between
  star-derived labels and the sentiment expressed in the text, especially
  around the neutral class.

Across all test reviews, neutral recall is 239/679 (35.2%); 277 neutral reviews
are predicted positive and 163 negative. Positive recall is 6,725/7,223 (93.1%).
Neutral is the weakest class for three measured reasons: it is 7.3% of training
reviews; three-star reviews usually mix praise and complaints ("but" appears in
72% of neutral reviews versus 41% of positive ones); and four-star and two-star
reviews often use the same hedged wording. The strongest neutral terms in
Task 4 ("not bad", "ok", "ok but", "good but") show the model has learned this
hedging, but the same phrases also occur in neighboring ratings.
The eight examples are deliberately selected for coverage and brevity; they are
not a random sample and cannot establish the prevalence or cause of a pattern.

## Report use and limits

Suggested caption: **Training-set prevalence of selected model-associated
words.** Bars show the percentage of reviews within each sentiment class
containing a term at least once. The top six terms for each class were
selected by learned class log-probability contrast with a minimum of 150
training occurrences. Bar heights measure document frequency, not learned
feature importance.

The local assignment's M2 requirements call for top-ten positive/negative
features for each representation, at least one visualization, and 5-10 error
examples. This contribution covers top-ten terms for both representations and
all three classes, one visualization (from the selected BoW model), and eight
examples. Label the chart as Bag of Words, not TF-IDF. Baseline
tables, timeline, progress logs, and the final 3-5 page PDF remain team work.

Students should read all eight full reviews, review these factual notes, and
write their own final interpretation. The assignment requires disclosure of AI
assistance in the final report; these notes do not replace that responsibility.
