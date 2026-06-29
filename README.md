# IDVerify

> Identity document verification with liveness detection and fraud analysis.

Trains four classifiers on synthetic identity verification data (face match score, liveness score, document authenticity, MRZ confidence, microprint score) to predict verification outcome. Dashboard provides document authenticity breakdowns, face match analysis, and liveness detection metrics.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Gradient Boosting) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.808 |
| Gini | 0.617 |
| KS Statistic | 0.483 |
| F1 Score | 0.553 |
| Accuracy | 0.756 |

5-fold CV AUC: 0.814 ± 0.006. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | Verification records, outcome distribution, feature descriptions |
| **Model Lab** | Multi-model comparison, ROC curves, calibration plots, CV results |
| **Fraud Analysis** | Document authenticity distribution, face match and liveness score distributions by verification outcome |
| **Liveness** | Liveness detection theory, microprint score analysis, MRZ confidence by document type |

## Repo Structure

```
IDVerify/
  src/         data, model, evaluate, persist modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic identity verification dataset: face match score, liveness score, document authenticity (genuine/forged/altered/suspicious), MRZ confidence, microprint score, UV verification, and document type.

## License

MIT
