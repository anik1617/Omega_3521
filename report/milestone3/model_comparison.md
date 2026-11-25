# Model comparison (Milestone 3)

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Neutral F1 | Training time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multinomial Naive Bayes (BoW) | 0.860 | 0.674 | 0.672 | 0.673 | 0.364 | 3.24 |
| Multinomial Naive Bayes (TF-IDF) | 0.859 | 0.668 | 0.668 | 0.667 | 0.352 | 3.71 |
| Logistic Regression (BoW) | 0.864 | 0.679 | 0.700 | 0.689 | 0.394 | 43.42 |
| Logistic Regression (TF-IDF) | 0.869 | 0.690 | 0.717 | 0.702 | 0.420 | 21.01 |
| Feedforward neural network (BoW) | 0.879 | 0.726 | 0.654 | 0.684 | 0.370 | 46.86* |
| Feedforward neural network (TF-IDF) | 0.875 | 0.738 | 0.608 | 0.648 | 0.291 | 20.97* |
| Feedforward neural network, dropout (TF-IDF) | 0.877 | 0.711 | 0.690 | 0.700 | 0.402 | 73.40* |
| Feedforward neural network, dropout (BoW) | 0.865 | 0.685 | 0.715 | 0.698 | 0.408 | 55.32* |
| Averaged word embeddings (learned embeddings) | 0.853 | 0.666 | 0.696 | 0.679 | 0.389 | 107.47* |
| 1D CNN (learned embeddings) | 0.861 | 0.692 | 0.667 | 0.678 | 0.400 | 169.94* |
| **DistilBERT fine-tuned (WordPiece tokens)** (best) | 0.908 | 0.758 | 0.748 | 0.753 | 0.478 | 676.20* |
| Majority-class baseline | 0.779 | 0.260 | 0.333 | 0.292 | 0.000 | n/a |

All numbers are on the same 20% test split (9,269 reviews, seed 46). Precision, recall, and F1 are macro averages over negative, neutral, and positive.

Training time is the fit time on the training split (vectorizer plus the selected model), not counting hyperparameter search.

Best model by validation macro F1: DistilBERT fine-tuned (WordPiece tokens).

\* Feedforward neural network (BoW): Total time in partial_fit over all epochs, in seconds.

\* Feedforward neural network (TF-IDF): Total time in partial_fit over all epochs, in seconds.

\* Feedforward neural network, dropout (TF-IDF): GPU time in the training loop for the kept loss weighting, over every epoch run including the ones after the best epoch; excludes validation passes and the weighting search.

\* Feedforward neural network, dropout (BoW): GPU time in the training loop for the kept loss weighting, over every epoch run including the ones after the best epoch; excludes validation passes and the weighting search.

\* Averaged word embeddings (learned embeddings): GPU time in the training loop for the kept loss weighting, over every epoch run including the ones after the best epoch; excludes validation passes and the weighting search.

\* 1D CNN (learned embeddings): GPU time in the training loop for the kept loss weighting, over every epoch run including the ones after the best epoch; excludes validation passes and the weighting search.

\* DistilBERT fine-tuned (WordPiece tokens): GPU time in the fine-tuning loop over all 3 epochs; excludes tokenization and validation passes.

Timed on: Linux-6.18.44-fc-v880-x86_64-with-glibc2.39 (Logistic Regression (BoW), Logistic Regression (TF-IDF)); Linux-6.18.44-fc-v880-x86_64-with-glibc2.39, 4 CPUs (Multinomial Naive Bayes (BoW), Multinomial Naive Bayes (TF-IDF), Feedforward neural network (BoW)); Windows-10-10.0.26200-SP0, 20 CPUs, NVIDIA GeForce RTX 3060 Laptop GPU (Feedforward neural network, dropout (TF-IDF), Feedforward neural network, dropout (BoW), Averaged word embeddings (learned embeddings), 1D CNN (learned embeddings), DistilBERT fine-tuned (WordPiece tokens)); macOS-15.7.3-arm64-arm-64bit-Mach-O, 8 CPUs (Feedforward neural network (TF-IDF)). Times from different machines are only roughly comparable.
