# Survey Classification Pipeline

End-to-end binary classification pipeline for predicting customer
propensity based on survey response data.

Trains a classifier to identify which customers are likely to respond
positively to a given survey variable — enabling targeted outreach
and personalised marketing. Built for Databricks with full MLflow
experiment tracking and Unity Catalog model registration.

---

## Business context

Survey data captures customer attitudes and preferences that are not
visible in transactional data alone. By joining survey responses onto
a customer feature table and training a propensity model, this
pipeline makes survey-derived insights scalable — scoring the full
customer base, not just survey respondents.

Typical use cases include predicting interest in a product category,
brand preference, or likelihood to respond to a campaign.

---

## Project structure

    survey-classification-pipeline/
    ├── config.py             # Country config, paths, and training defaults
    ├── base_pipeline.py      # Shared predict() and register() logic
    ├── survey_pipeline.py    # Full training pipeline with MLflow logging
    └── preprocessing.py      # Custom sklearn transformers (not included)

---

## How it works

| Step | What happens |
|---|---|
| 1. Data cleaning | Initialises country base tables, curates features, joins survey targets |
| 2. Train/test split | Stratified split, default 80/20 |
| 3. Preprocessing | Encoding → imputation → dataset cleaning → VIF-based redundancy removal → SelectKBest |
| 4. Hyperparameter optimisation | Automated search via Hyperopt for LR or XGBoost |
| 5. MLflow logging | Model, metrics, SHAP beeswarm, ROC curve, confusion matrix |

---

## Usage

```python
from survey_pipeline import SurveyClassificationPipeline

pipeline = SurveyClassificationPipeline(
    country         = "no",
    report          = report,
    variable_name   = "product_interest",
    positive_labels = ["Very interested", "Interested"],
    negative_labels = ["Not interested", "Not at all interested"],
)

pipeline.fit(
    X,
    model_type    = "lr",     # or "xgb"
    n_features    = 10,
    vif_threshold = 10,
)

pipeline.register(
    model_name = "product_interest_model",
    schema     = "your_schema",
)
```

**Generating predictions:**

```python
scores = pipeline.predict(
    context = None,
    X       = X_new,
    params  = {"by_household": True, "problem_type": "classification"},
)
```

Output columns: `household_id`, `score`, `decile` (10 = highest propensity).

---

## MLflow artifacts logged per run

| Artifact | Description |
|---|---|
| `pipeline` | Fitted pyfunc model with input/output signature |
| `shap_beeswarm.png` | Feature importance via SHAP |
| `roc_curve.png` | ROC curve on test set |
| `confusion_matrix.png` | Count and normalised confusion matrices |
| `confusion_matrix_test.json` | Confusion matrix values as JSON |

## Metrics logged per run

| Metric | Description |
|---|---|
| `auc_train` / `auc_test` | Area under the ROC curve |
| `balanced_accuracy_train` / `_test` | Balanced accuracy |
| `precision_train` / `_test` | Precision |
| `recall_train` / `_test` | Recall |
| `true_positive` / `false_positive` / etc. | Confusion matrix values |

---

## Supported classifiers

| model_type | Classifier | Notes |
|---|---|---|
| `lr` | Logistic Regression | Fast, interpretable, good baseline |
| `xgb` | XGBoost | Higher capacity, better for non-linear patterns |

Both use automated hyperparameter optimisation via Hyperopt.

---

## Requirements

- Databricks Runtime 12.x or later
- MLflow (included in Databricks Runtime)
- scikit-learn, XGBoost, SHAP, scikit-plot, hyperopt

---

## License

MIT