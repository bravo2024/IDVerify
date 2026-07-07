# IDVerify

Identity-document verification with liveness checks and fraud signals.

Trains four classifiers on synthetic identity verification data (face match score, liveness score, document authenticity, MRZ confidence, microprint score) to predict verification outcome. Dashboard provides document authenticity breakdowns, face match analysis, and liveness detection metrics.

## Results (holdout)

Best model: Gradient Boosting.

| Metric | Value |
|---|---|
| ROC AUC | 0.808 |
| Gini | 0.617 |
| KS Statistic | 0.483 |
| F1 Score | 0.553 |
| Accuracy | 0.756 |

5-fold CV AUC: 0.814 ± 0.006. Four models compared.

## Setup

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Layout

```
IDVerify/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Verification tabs

| Tab | What it does |
|---|---|
| **Explorer** | Verification records, outcome distribution, feature descriptions |
| **Model Lab** | Multi-model comparison, ROC curves, calibration plots, CV results |
| **Fraud Analysis** | Document authenticity distribution, face match and liveness score distributions by verification outcome |
| **Liveness** | Liveness detection theory, microprint score analysis, MRZ confidence by document type |

## Data & license

Synthetic identity verification dataset: face match score, liveness score, document authenticity (genuine/forged/altered/suspicious), MRZ confidence, microprint score, UV verification, and document type.

MIT.
