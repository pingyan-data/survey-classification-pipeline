"""
base_pipeline.py

Base class for survey-based classification pipelines.

Provides shared prediction and model registration logic.
Subclasses implement fit() for their specific data source and
model type, but must not override predict() or register()
unless there is a compelling reason to do so.
"""

import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc

from config import REGISTRY_URI, MODEL_CATALOG, VALID_SCHEMAS


class BasePipeline(mlflow.pyfunc.PythonModel):
    """
    Base class for survey-based classification pipelines.

    Provides standardised prediction (with household-level aggregation
    and decile scoring) and MLflow model registration. All subclasses
    must call super().__init__() and implement fit().

    Attributes:
        individual_id:          Name of the individual identifier column.
        household_id:           Name of the household identifier column.
        run_id:                 MLflow run ID set after fit() is called.
        preprocessing_pipelines: List of fitted sklearn Pipeline objects
                                 applied sequentially at inference time.
        classifier:             Fitted sklearn-compatible classifier.
    """

    def __init__(self) -> None:
        self.individual_id:           str | None  = None
        self.household_id:            str | None  = None
        self.run_id:                  str | None  = None
        self.preprocessing_pipelines: list        = []
        self.classifier:              object| None = None

    def predict(
        self,
        context,
        X: pd.DataFrame,
        params: dict | None = None,
    ) -> pd.DataFrame:
        """
        Generates propensity scores and deciles for a feature DataFrame.

        Applies all fitted preprocessing pipelines sequentially, then
        runs the classifier. Optionally aggregates predictions to
        household level by taking the maximum individual score per
        household.

        Args:
            context:      MLflow PythonModel context (unused, required by API).
            X:            Feature DataFrame. Must contain individual_id and
                          household_id columns alongside model features.
            params:       Optional dict with keys:
                          - by_household (bool, default True): aggregate to
                            household level using max score.
                          - problem_type (str, default "classification"):
                            "classification" uses predict_proba,
                            "regression" uses predict.

        Returns:
            DataFrame with columns:
            - individual_id or household_id (depending on by_household)
            - score: predicted propensity probability
            - decile: score decile (10 = highest propensity, 1 = lowest)

        Raises:
            ValueError: If problem_type is not a supported value.
        """
        params        = params or {}
        by_household  = params.get("by_household", True)
        problem_type  = params.get("problem_type", "classification")

        # Preserve ID columns before preprocessing transforms them away
        id_cols = X[[self.individual_id, self.household_id]].copy()

        # Apply preprocessing pipelines in sequence
        X_transformed = X.copy()
        for pipe in self.preprocessing_pipelines:
            X_transformed = pipe.transform(X_transformed)

        # Score
        if problem_type == "classification":
            positive_class_idx = list(self.classifier.classes_).index(1)
            scores = self.classifier.predict_proba(X_transformed)[:, positive_class_idx]
        elif problem_type == "regression":
            scores = self.classifier.predict(X_transformed)
        else:
            raise ValueError(
                f"Unsupported problem_type='{problem_type}'. "
                "Choose 'classification' or 'regression'."
            )

        id_cols["score"] = scores

        # Aggregate to household level if requested
        if by_household:
            id_cols = (
                id_cols
                .groupby(self.household_id)["score"]
                .max()
                .reset_index()
            )

        # Assign deciles (10 = highest score, 1 = lowest)
        id_cols["decile"] = self._assign_deciles(id_cols["score"])

        id_col = self.household_id if by_household else self.individual_id
        result = id_cols[[id_col, "score", "decile"]]

        if hasattr(self, "report"):
            self.report.result = result

        return result

    def register(
        self,
        model_name: str,
        schema: str,
        catalog: str = MODEL_CATALOG,
    ) -> None:
        """
        Registers the fitted model in the MLflow Model Registry.

        Must be called after fit(). The model is registered under
        catalog.schema.model_name in Databricks Unity Catalog.

        Args:
            model_name: Name for the registered model.
            schema:     Target schema. Must be one of VALID_SCHEMAS
                        defined in config.py.
            catalog:    Target catalog (default: MODEL_CATALOG from config).

        Raises:
            RuntimeError: If called before fit().
            ValueError:   If schema is not in VALID_SCHEMAS.
        """
        if self.run_id is None:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before register()."
            )
        if schema not in VALID_SCHEMAS:
            raise ValueError(
                f"Invalid schema '{schema}'. "
                f"Valid options: {VALID_SCHEMAS}"
            )

        mlflow.set_registry_uri(REGISTRY_URI)
        registered_name = f"{catalog}.{schema}.{model_name}"
        mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pipeline",
            name=registered_name,
        )
        print(f"Model registered as: {registered_name}")

    @staticmethod
    def _assign_deciles(scores: pd.Series) -> pd.Series:
        """
        Assigns decile labels (10 = highest, 1 = lowest) to a score series.

        Uses a small random jitter to handle ties in qcut when score
        distributions have many identical values.

        Args:
            scores: Series of propensity scores.

        Returns:
            Series of integer decile labels.
        """
        labels = np.arange(10, 0, -1)
        try:
            return pd.qcut(scores, 10, labels=labels)
        except ValueError:
            # Non-unique bin edges — add tiny jitter to break ties
            std = scores.std()
            jitter = (np.random.random(len(scores)) * std / 1e6) - (std / 2e6)
            return pd.qcut(scores + jitter, 10, labels=labels)