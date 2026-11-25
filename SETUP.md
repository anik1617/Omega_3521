# Setup after pulling the DistilBERT update

Follow these steps in order from the repository root. They update your Python
packages and download the fine-tuned DistilBERT model, which is stored with
Git LFS. Every step is safe to rerun.

## What changed

- `models/distilbert_seed46/` holds the fine-tuned DistilBERT model, the best model
  by validation macro-F1. Its weights file, `model.safetensors` (about 134 MB), is
  stored with Git LFS. Without Git LFS you get a text pointer file of about 130 bytes,
  and the app quietly leaves DistilBERT out.
- `requirements.txt` now also pins `torch==2.10.0` and `transformers==5.3.0`, which the
  app needs to run DistilBERT. It points pip at the CPU-only torch build (about 200 MB
  instead of several GB). A CUDA build of torch that is already installed also works.

## 1. Install Git LFS (once per computer)

Check whether it is installed:

```
git lfs version
```

If that fails, install it:

- Windows: Git for Windows usually includes it. Otherwise `winget install GitHub.GitLFS`.
- macOS: `brew install git-lfs`
- Debian/Ubuntu: `sudo apt-get install git-lfs`

Then turn it on for your user account (once per computer):

```
git lfs install
```

## 2. Get the latest code and the model file

```
git checkout main
git pull
git lfs pull
```

Check that the real model file arrived. It should be about 134 MB (133,923,966 bytes),
not about 130 bytes:

- Windows PowerShell: `(Get-Item models/distilbert_seed46/model.safetensors).Length`
- macOS/Linux: `ls -l models/distilbert_seed46/model.safetensors`

If it is still small, run `git lfs install` and then `git lfs pull` again.

## 3. Update Python packages

Use the same environment you already run the app in (for example `.venv`), with
Python 3.11 or newer:

```
pip install -r requirements.txt
```

Check that everything imports:

```
python -c "import torch, transformers, streamlit, sklearn; print(torch.__version__, transformers.__version__)"
```

Expected output starts with `2.10.0` (any suffix such as `+cpu` or `+cu128` is fine)
followed by `5.3.0`.

## 4. Check the model loads

```
python -c "from transformers import AutoModelForSequenceClassification as M; m = M.from_pretrained('models/distilbert_seed46'); print(m.config.id2label)"
```

Expected output: `{0: 'negative', 1: 'neutral', 2: 'positive'}`.

## 5. Run the app

```
streamlit run app.py
```

The model menu should list `★ DistilBERT fine-tuned (WordPiece tokens) (best)` first.
The first DistilBERT prediction takes a few seconds while the model loads, and later
ones are fast.

## Notes

- Do not retrain anything to use the app. All models in `models/` are already
  trained. Do not rerun `scripts/make_splits.py`.
- Avoid cloning the repository again and again, for example in loops or fresh
  environments. Every download of the 134 MB model counts against the repository
  owner's Git LFS bandwidth (1 GB a month on a free GitHub account). One `git lfs pull`
  per computer is enough.
- Retraining DistilBERT (`python scripts/train_distilbert.py`) needs an NVIDIA GPU
  with a CUDA build of torch. See `requirements-gpu.txt`. It is not needed for
  the app.
- Troubleshooting:
  - `safetensors` header or "invalid file" error: the LFS file was not downloaded. Repeat step 2.
  - `No module named torch`: repeat step 3 in the environment that runs Streamlit.
  - `This repository is over its data quota`: the LFS bandwidth for the month is used
    up. Ask Aniket for the model folder directly.
