# Notebooks

One notebook per task, named with the task number from the M2 tracker and your name,
for example `02_features_sophiya.ipynb` or `03_naive_bayes_shilin.ipynb`.

Load the data the same way in every notebook:

```python
import sys; sys.path.insert(0, '..')
from data_utils import load_splits

train, val, test = load_splits()
```

Fit vectorizers and models on `train`, tune on `val`, and use `test` only for the
numbers that go in the report.
