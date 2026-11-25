# Milestone 3 Task 2 — Feedforward neural network

The network reads 192,646 TF-IDF features from the team's fitted vectorizer.
It has two ReLU hidden layers with 32 and 16 neurons and a three-class softmax
output. The input is the review summary and body joined with a space, using
the shared seed-46 split (30,123 training, 6,952 validation, 9,269 test).

Adam updates the model in batches of up to 1,024 reviews. The script runs 18
epochs with `partial_fit` and seed 46. The training and validation matrices
stay sparse and use float32 to reduce training memory. After each epoch the
script measures multiclass log loss on both splits and computes validation
macro-F1. The checkpoint with the highest validation macro-F1 is saved, with
earlier epochs winning ties. Test metrics are computed after checkpoint
selection.

From the repository root, with the pinned dependencies installed:

```bash
python scripts/train_neural_network.py
```

The script saves the raw-text pipeline to
`models/neural_network_tfidf_seed46.joblib` and the comparison and app metrics
to `report/model_results/neural_network_seed46.json`. The epoch measurements
and loss plot are in `neural_network_training_history.csv` and
`neural_network_loss.png` in this directory.

## Results

Epoch 11 had the highest validation macro-F1 (0.6444). The selected network
scored 0.8746 accuracy on the test set. Test macro precision was 0.7379,
macro recall was 0.6082, and macro-F1 was 0.6484.

| Test class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Negative | 0.8221 | 0.6357 | 0.7170 |
| Neutral | 0.4965 | 0.2062 | 0.2914 |
| Positive | 0.8951 | 0.9827 | 0.9368 |

The 18 calls to `partial_fit` took 20.967 seconds in total on arm64 macOS
with 8 logical CPUs and Python 3.13.7. This measures training updates only.
The time excludes TF-IDF transformation, epoch scoring, and saving files.
The JSON includes the machine and library versions and the test confusion matrix.

Validation loss reached its minimum at epoch 8 (0.3411) and then rose while
training loss kept falling, a sign of overfitting. The checkpoint was picked
by validation macro-F1, which weights all three classes equally. The confusion
matrix shows that 421 of 679 neutral reviews were predicted positive. This
helps explain why test accuracy is high while macro-F1 trails the existing
TF-IDF Logistic Regression result (0.702).
