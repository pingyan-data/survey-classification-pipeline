"""
config.py

Central configuration for the survey-based classification pipeline.
Replace placeholder paths with your actual Databricks paths
before running. Do not commit real paths to version control.
"""

# ── Country-specific identifiers ───────────────────────────────
COUNTRY_CONFIG = {
    "se": {
        "individual_id":  "individual_id",
        "household_id":   "household_id",
        "survey_file":    "/Volumes/your_catalog/your_schema/survey/se_survey_responses.txt",
        "feature_file":   "/Volumes/your_catalog/your_schema/features/se_features.csv",
    },
    "no": {
        "individual_id":  "individual_id",
        "household_id":   "household_id",
        "survey_file":    "/Volumes/your_catalog/your_schema/survey/no_survey_responses.csv",
        "feature_file":   "/Volumes/your_catalog/your_schema/features/no_features.csv",
    },
}

SUPPORTED_COUNTRIES   = list(COUNTRY_CONFIG.keys())
SUPPORTED_MODEL_TYPES = ["lr", "xgb"]

# ── MLflow / model registry ────────────────────────────────────
REGISTRY_URI  = "databricks-uc"
MODEL_CATALOG = "your_catalog"

VALID_SCHEMAS = [
    "your_schema_country_a",
    "your_schema_country_b",
]

# ── Training defaults ──────────────────────────────────────────
DEFAULT_TEST_RATIO    = 0.2
DEFAULT_VIF_THRESHOLD = 10
DEFAULT_N_FEATURES    = 10

# ── Package versions pinned for MLflow logging ─────────────────
PIP_REQUIREMENTS = [
    "pandas==1.5.3",
    "numpy==1.23.5",
    "scikit-learn==1.5.1",
    "pyspark==3.5.0",
    "matplotlib==3.7.0",
    "scikit-plot==0.3.7",
    "statsmodels==0.13.5",
    "shap==0.44.0",
    "hyperopt==0.2.7",
    "python-docx==1.1.2",
]