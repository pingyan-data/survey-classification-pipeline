"""
survey_pipeline.py

Classification pipeline trained on survey response data.

Loads a customer feature table and joins survey responses as targets,
then trains a classification model to predict propensity for a given
survey variable. Supports Logistic Regression and XGBoost classifiers
with automated hyperparameter optimisation.

Designed to run in a Databricks environment with MLflow tracking.
"""

import shap
import mlflow
import mlflow.pyfunc
import scikitplot as skplot
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection    import train_test_split
from sklearn.linear_model       import LogisticRegression
from sklearn.pipeline           import Pipeline
from sklearn.feature_selection  import SelectKBest
from sklearn.metrics            import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from pyspark.sql import SparkSession
import xgboost as xgb

from base_pipeline import BasePipeline
from config import (
    COUNTRY_CONFIG,
    SUPPORTED_COUNTRIES,
    SUPPORTED_MODEL_TYPES,
    DEFAULT_TEST_RATIO,
    DEFAULT_VIF_THRESHOLD,
    DEFAULT_N_FEATURES,
    PIP_REQUIREMENTS,
)

# These are custom preprocessing steps 
from preprocessing import (
    BaseTableInitializer,
    FeatureCurator,
    SurveyTargetJoiner,
    get_encoder,
    ColumnNameResetter,
    CustomSimpleImputer,
    DatasetCleaner,
    RedundancyDropper,
)
from optimizers import LogisticRegressionOptimizer, XGBOptimizer


class SurveyClassificationPipeline(BasePipeline):
    """
    Classification pipeline for predicting survey response propensity.

    Trains a binary classifier to identify customers likely to respond
    positively to a given survey variable. Handles the full training
    lifecycle: data loading, preprocessing, feature selection,
    hyperparameter optimisation, evaluation, and MLflow logging.

    Args:
        country:         Country code — 'se' or 'no'.
        report:          Report object for storing metrics and outputs.
        variable_name:   Survey variable to predict (e.g. 'Q5_brand_pref').
        positive_labels: List of survey response values treated as positive class.
        negative_labels: List of survey response values treated as negative class.
        feature_file:    Path to the curated feature list file. If None,
                         falls back to the country default in config.

    Example:
        pipeline = SurveyClassificationPipeline(
            country        = "no",
            report         = report,
            variable_name  = "product_interest",
            positive_labels= ["Very interested", "Interested"],
            negative_labels= ["Not interested", "Not at all interested"],
            feature_file   = None,
        )
        pipeline.fit(X)
        pipeline.register("product_interest_model", schema="product_schema")
    """

    def __init__(
        self,
        country:         str,
        report:          object,
        variable_name:   str,
        positive_labels: list[str],
        negative_labels: list[str],
        feature_file:    str | None = None,
    ) -> None:
        super().__init__()

        country = country.lower()
        if country not in SUPPORTED_COUNTRIES:
            raise ValueError(
                f"Invalid country '{country}'. "
                f"Supported: {SUPPORTED_COUNTRIES}"
            )

        cfg = COUNTRY_CONFIG[country]
        self.country          = country
        self.individual_id    = cfg["individual_id"]
        self.household_id     = cfg["household_id"]
        self.survey_file      = cfg["survey_file"]
        self.feature_file     = feature_file or cfg["feature_file"]

        self.variable_name    = variable_name
        self.positive_labels  = positive_labels
        self.negative_labels  = negative_labels

        self.report                  = report
        self.report.positive_labels  = positive_labels
        self.report.negative_labels  = negative_labels

    def fit(
        self,
        X,
        y                = None,
        test_ratio:  float = DEFAULT_TEST_RATIO,
        vif_threshold: int = DEFAULT_VIF_THRESHOLD,
        n_features:    int = DEFAULT_N_FEATURES,
        model_type:    str = "lr",
        spark_session       = None,
    ) -> None:
        """
        Trains the classification pipeline end-to-end.

        Steps:
        1. Clean and join survey targets onto the feature table
        2. Encode, impute, and select features
        3. Optimise hyperparameters and train classifier
        4. Evaluate on held-out test set
        5. Log model, metrics, SHAP values, and plots to MLflow

        Args:
            X:             Input feature DataFrame or Spark DataFrame.
            y:             Unused — targets are joined from the survey file.
            test_ratio:    Fraction of data held out for evaluation.
            vif_threshold: VIF threshold for removing multicollinear features.
            n_features:    Number of features to retain after selection.
            model_type:    Classifier type — 'lr' (Logistic Regression)
                           or 'xgb' (XGBoost).
            spark_session: Active SparkSession. Uses active session if None.

        Raises:
            ValueError: If model_type is not supported.
        """
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type='{model_type}'. "
                f"Choose from {SUPPORTED_MODEL_TYPES}."
            )

        spark = spark_session or SparkSession.builder.getOrCreate()

        # ── Step 1: Data cleaning and target joining ───────────
        print("Step 1/5: Data cleaning and target joining...")
        data_cleaning_pipeline = Pipeline(steps=[
            ("init_base_tables",  BaseTableInitializer(self.country)),
            ("curate_features",   FeatureCurator(self.country,
                                                 feature_file=self.feature_file)),
            ("join_survey_targets", SurveyTargetJoiner(
                report          = self.report,
                survey_file     = self.survey_file,
                id_column       = self.individual_id,
                variable_name   = self.variable_name,
                positive_labels = self.positive_labels,
                negative_labels = self.negative_labels,
            )),
        ])
        X = data_cleaning_pipeline.fit_transform(X, y=None)
        print(f"  Dataset shape after cleaning: {X.shape}")

        # ── Step 2: Train/test split ───────────────────────────
        print("Step 2/5: Splitting into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X.drop(columns=["target"]),
            X["target"],
            stratify   = X["target"],
            test_size  = test_ratio,
            random_state = 42,
            shuffle    = True,
        )
        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        print(f"  Target rate (train): {y_train.mean():.1%}")

        # ── Step 3: Preprocessing and feature selection ────────
        print("Step 3/5: Preprocessing and feature selection...")
        preprocessing_pipeline = Pipeline(steps=[
            ("encode",          get_encoder(X_train)),
            ("reset_cols_1",    ColumnNameResetter()),
            ("impute",          CustomSimpleImputer()),
            ("reset_cols_2",    ColumnNameResetter()),
            ("clean_dataset",   DatasetCleaner(
                                    id_column           = self.individual_id,
                                    household_id_column = self.household_id,
                                )),
            ("drop_multicollinear", RedundancyDropper(
                                    self.report,
                                    threshold = vif_threshold,
                                )),
            ("select_features", SelectKBest(k=n_features)),
        ])
        preprocessing_pipeline.fit(X_train, y_train)
        self.preprocessing_pipelines.append(preprocessing_pipeline)

        X_train = preprocessing_pipeline.transform(X_train)
        X_test  = preprocessing_pipeline.transform(X_test)
        print(f"  Features after selection: {X_train.shape[1]}")

        # ── Step 4: Hyperparameter optimisation and training ───
        print(f"Step 4/5: Training {model_type.upper()} classifier...")
        if model_type == "lr":
            optimizer = LogisticRegressionOptimizer(X_train, y_train)
            best_params = optimizer.find_best_hyperparameters()
            self.classifier = LogisticRegression(**best_params)
        else:
            optimizer = XGBOptimizer(X_train, y_train)
            best_params = optimizer.find_best_hyperparameters()
            self.classifier = xgb.XGBClassifier(**best_params)

        self.classifier.fit(X_train, y_train)

        y_pred_train       = self.classifier.predict(X_train)
        y_pred_proba_train = self.classifier.predict_proba(X_train)
        y_pred_test        = self.classifier.predict(X_test)
        y_pred_proba_test  = self.classifier.predict_proba(X_test)

        # Store evaluation data on the report object
        self.report.classifier          = self.classifier
        self.report.X_train             = X_train
        self.report.y_train             = y_train
        self.report.X_test              = X_test
        self.report.y_test              = y_test
        self.report.y_pred_proba_train  = y_pred_proba_train
        self.report.y_pred_proba_test   = y_pred_proba_test
        self.report.y_pred_train        = y_pred_train
        self.report.y_pred_test         = y_pred_test

        # ── Step 5: MLflow logging ─────────────────────────────
        print("Step 5/5: Logging to MLflow...")
        self._log_to_mlflow(
            X_train, X_test, y_train, y_test,
            y_pred_train, y_pred_test,
            y_pred_proba_train, y_pred_proba_test,
            best_params,
        )
        print("Pipeline training complete.")

    def _log_to_mlflow(
        self,
        X_train, X_test,
        y_train, y_test,
        y_pred_train, y_pred_test,
        y_pred_proba_train, y_pred_proba_test,
        best_params: dict,
    ) -> None:
        """
        Logs the fitted model, metrics, and diagnostic plots to MLflow.

        Logs:
        - Fitted pipeline as a pyfunc model with input/output signature
        - Hyperparameters
        - Balanced accuracy, AUC, precision, recall (train and test)
        - Confusion matrix values and JSON artifact
        - SHAP beeswarm plot
        - ROC curve
        - Confusion matrix visualisation

        Args:
            X_train, X_test:                 Preprocessed feature DataFrames.
            y_train, y_test:                 Target Series.
            y_pred_train, y_pred_test:       Hard predictions.
            y_pred_proba_train/test:         Probability predictions.
            best_params:                     Hyperparameters from optimiser.
        """
        signature      = self._build_signature(X_train)
        conf_matrix    = confusion_matrix(y_test, y_pred_test)
        tn, fp, fn, tp = conf_matrix.ravel()

        with mlflow.start_run() as run:
            self.run_id = run.info.run_id

            # Log model
            mlflow.pyfunc.log_model(
                "pipeline",
                python_model    = self,
                pip_requirements= PIP_REQUIREMENTS,
                signature       = signature,
                code_path       = ["../src"],
            )

            # Log hyperparameters
            mlflow.log_params(best_params)

            # Log evaluation metrics
            metrics = {
                "balanced_accuracy_train": balanced_accuracy_score(y_train, y_pred_train),
                "balanced_accuracy_test":  balanced_accuracy_score(y_test,  y_pred_test),
                "auc_train": roc_auc_score(y_train, y_pred_proba_train[:, 1]),
                "auc_test":  roc_auc_score(y_test,  y_pred_proba_test[:, 1]),
                "precision_train": precision_score(y_train, y_pred_train),
                "precision_test":  precision_score(y_test,  y_pred_test),
                "recall_train": recall_score(y_train, y_pred_train),
                "recall_test":  recall_score(y_test,  y_pred_test),
                "true_positive":  int(tp),
                "true_negative":  int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
            }
            mlflow.log_metrics(metrics)
            mlflow.log_dict(
                np.array(conf_matrix).tolist(),
                "confusion_matrix_test.json",
            )

            # Log SHAP beeswarm plot
            explainer   = shap.Explainer(
                self.classifier.predict,
                X_train,
                feature_names = X_train.columns.tolist(),
            )
            shap_values = explainer(X_test)
            shap.plots.beeswarm(shap_values, show=False)
            fig = plt.gcf()
            fig.tight_layout()
            mlflow.log_figure(fig, "shap_beeswarm.png")
            plt.close(fig)

            # Log ROC curve
            skplot.metrics.plot_roc(
                y_true  = y_test,
                y_probas= y_pred_proba_test,
                title   = "ROC Curve",
            )
            fig = plt.gcf()
            fig.tight_layout()
            mlflow.log_figure(fig, "roc_curve.png")
            plt.close(fig)

            # Log confusion matrix
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ConfusionMatrixDisplay(conf_matrix).plot(
                ax=axes[0], cmap="Blues", colorbar=False
            )
            ConfusionMatrixDisplay(
                confusion_matrix(y_test, y_pred_test, normalize="true") * 100
            ).plot(ax=axes[1], cmap="Blues", colorbar=False)
            axes[0].set_title("Counts")
            axes[1].set_title("Normalised (%)")
            fig.tight_layout()
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

        print(f"  MLflow run ID: {self.run_id}")
        print(f"  AUC (test): {metrics['auc_test']:.3f}")
        print(f"  Balanced accuracy (test): {metrics['balanced_accuracy_test']:.3f}")

    def _build_signature(self, X_train) -> mlflow.models.ModelSignature:
        """
        Builds an MLflow model signature with fixed output schema.

        Args:
            X_train: Preprocessed training DataFrame used to infer
                     input schema.

        Returns:
            MLflow ModelSignature with input features, output scores,
            and inference params.
        """
        signature      = mlflow.models.infer_signature(X_train)
        signature_dict = signature.to_dict()

        output_schema = (
            '[{"type": "string", "name": "'
            + self.individual_id
            + '"}, {"type": "float", "name": "score"}, '
            '{"type": "integer", "name": "decile"}]'
        )
        params_schema = (
            '[{"name": "by_household", "type": "boolean", "default": true}, '
            '{"name": "problem_type", "type": "string", "default": "classification"}]'
        )

        signature_dict["outputs"] = output_schema
        signature_dict["params"]  = params_schema

        return mlflow.models.ModelSignature.from_dict(signature_dict)
