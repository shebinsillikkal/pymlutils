# PyMLUtils

> Reusable ML utilities I kept rewriting across projects — so I packaged them properly.

## Install
```bash
pip install pymlutils
```

## What's Included
- `preprocessing` — Smart imputers, encoders, scalers with fit/transform API
- `evaluation` — Classification/regression report generators with visualisations
- `explainability` — SHAP wrappers, feature importance plots
- `pipelines` — Pre-built sklearn pipeline templates for common use cases
- `validation` — Time-series aware cross-validators

## Quick Example
```python
from pymlutils.evaluation import ClassificationReport
from pymlutils.preprocessing import SmartImputer
from sklearn.ensemble import RandomForestClassifier

imputer = SmartImputer(strategy='auto')
X_clean = imputer.fit_transform(X_train)

clf = RandomForestClassifier()
clf.fit(X_clean, y_train)

report = ClassificationReport(clf, X_test, y_test)
report.show()
report.plot_roc()
report.plot_confusion()
```

## Why This Exists
I was writing the same preprocessing boilerplate and evaluation helpers on every project. Packaged them up, put them on GitHub. More people found it than I expected.

**Built by Shebin S Illikkal** — Shebinsillikkal@gmail.com | Kerala, India
