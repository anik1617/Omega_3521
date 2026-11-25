# GPU models (Aniket)

Trained on an RTX 3060 Laptop GPU with `scripts/train_torch_models.py` and
`scripts/train_distilbert.py`. Same seed-46 split as every other model; each
checkpoint was picked by validation macro-F1 and the test split was used once.

| Model | Accuracy | Macro F1 | Neutral F1 | Training time (s) |
| --- | ---: | ---: | ---: | ---: |
| DistilBERT fine-tuned (3 epochs) | 0.908 | 0.753 | 0.476 | 676 |
| Feedforward network, dropout + early stopping (TF-IDF) | 0.877 | 0.700 | 0.402 | 73 |
| Feedforward network, dropout + early stopping (BoW) | 0.865 | 0.698 | 0.408 | 55 |
| Averaged word embeddings | 0.853 | 0.679 | 0.389 | 107 |
| 1D CNN on word embeddings | 0.861 | 0.678 | 0.400 | 170 |
| Logistic Regression (TF-IDF), for reference | 0.869 | 0.702 | 0.420 | 21 |

Notes:

- DistilBERT is the only model clearly ahead of Logistic Regression: +0.051
  macro-F1 and +0.056 neutral F1. Its validation macro-F1 was still rising at
  epoch 3 (0.719, 0.720, 0.752), so a fourth epoch might help a little.
- The feedforward network has hidden layers of 64 and 32 with dropout 0.5 and
  stops after 3 epochs without a validation gain. With dropout, BoW and TF-IDF
  give almost the same macro-F1 (0.698 vs 0.700). The earlier scikit-learn MLP
  without dropout scored 0.684 on BoW and 0.648 on TF-IDF, so the gap there came
  mostly from overfitting, not from BoW being the better input.
- Every model here kept the square-root class-weighted loss over plain
  cross-entropy on validation, which mainly helps the neutral class.
- The CNN and averaged-embedding models learn embeddings from scratch on 30k
  reviews and land below Logistic Regression. Pretrained word vectors were not tried.
- Model weights are not committed (too large for GitHub). Rerun the scripts to
  get them back in `models/torch/` and `models/distilbert_seed46/`.

Files: `history_*.csv` (per-epoch losses and validation macro-F1),
`loss_*.png` (loss curves).
