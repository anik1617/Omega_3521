# Milestone 2: Tasks 4-6 (Ethan Brown)

These outputs adapt the supplied `Omega_Milestone2_Tasks4-6_Local.zip` analysis
to the shared repository. They load Shilin's saved pipeline and the committed
seed-46 split through `data_utils.load_splits()`; nothing is resampled or fitted.
Input text is Summary plus Text. Score is used only to display the true rating
in the error examples, never as a feature.

## Outputs and methods

- **Task 4:** `task4_top_model_terms.csv` contains ten negative and ten positive
  terms from the selected Bag of Words MultinomialNB model (alpha=0.1).
  For each target class, the ranking score is
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
  The chart uses the saved default CountVectorizer preprocessing, matching
  Sophiya's notebook: lowercase unigrams, tokens of at least two word characters,
  no stop-word removal or vocabulary pruning. Negation words remain in the
  vocabulary, although default tokenization does not preserve contraction scope.
- **Task 6:** `task6_eight_errors.csv` contains eight actual incorrect test
  predictions, with IDs, true ratings/labels, predictions, summaries, excerpts,
  and full Text. One example is selected for each of the six actual/predicted
  error pairs by smallest ID; the last two are the shortest remaining Text
  fields (whitespace word count), breaking ties by ID. This preserves the
  supplied examples. `task6_error_counts.csv` counts every test error by pair.
- `verification.json` records split sizes, vocabulary, model configuration,
  metrics, confusion matrix, and methods.

## Reproduce and verify

From the repository root with Python 3.11+:

```bash
python -m pip install -r requirements.txt
python scripts/run_tasks_4_5_6.py
```

Outputs are written to `report/milestone2/` regardless of the working directory.
The script checks disjoint split IDs, sizes 30,123/6,952/9,269, vocabulary 35,788,
and agreement between the saved classifier's feature counts and the shared
training matrix. It also checks accuracy 0.8258711835, macro-F1 0.6247747233,
1,614 test errors, and eight distinct incorrect examples. It never calls fit.
The generated CSV values match the supplied ZIP; the chart is regenerated at
300 DPI, and verification metadata now identifies the original saved pipeline.
The original misleading 30-occurrence comment was corrected to 150.

## Evidence to review for the error discussion

- Reviews 2933 and 1902 are four-star reviews with both favorable language and
  complaints (freshness or one flavor). Review 6814 is two-star despite some
  favorable taste comments. These examples suggest difficulty combining mixed
  assessments into one sentiment label.
- Review 123381 praises Dingos but says the dogs will no longer touch other
  treats. Its positive-to-negative error suggests a possible limitation of
  unigrams for negation and the target of a comparison.
- Reviews 2601 and 3152 have three-star labels but negative wording; review 61
  mixes praise with a price complaint. These illustrate ambiguity between
  star-derived labels and sentiment expressed in the text.
- Review 907 complains about plastic in baby food despite a positive prediction.
  The example establishes the error; the words responsible have not been tested.

Across all test reviews, neutral recall is 236/679 (34.76%); 270 neutral reviews
are predicted positive and 173 negative. Positive recall is 6,545/7,223 (90.61%).
The eight examples are deliberately selected for coverage and brevity; they are
not a random sample and cannot establish the prevalence or cause of a pattern.
Possible follow-up experiments include bigrams for phrases and comparisons of
alternative models. No improvement from these ideas has been validated here.

## Report use and limits

Suggested caption: **Training-set prevalence of selected model-associated
words.** Bars show the percentage of reviews within each sentiment class
containing a term at least once. The top six negative and positive terms were
selected by learned class log-probability contrast with a minimum of 150
training occurrences. Bar heights measure document frequency, not learned
feature importance.

The local assignment's M2 requirements call for top-ten positive/negative
features for each representation, at least one visualization, and 5-10 error
examples. This contribution covers the selected BoW model, one visualization,
and eight examples. TF-IDF-specific feature interpretation remains a team
report item; these BoW rankings must not be labeled as TF-IDF rankings. Baseline
tables, timeline, progress logs, and the final 3-5 page PDF remain team work.

Students should read all eight full reviews, review these factual notes, and
write their own final interpretation. The assignment requires disclosure of AI
assistance in the final report; these notes do not replace that responsibility.
