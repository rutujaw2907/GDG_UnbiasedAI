# backend/model/train.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from backend.utils.preprocessing import build_preprocessor, get_feature_names


@dataclass
class TrainArtifacts:
    model_pipeline: Pipeline
    feature_names: List[str]
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray
    y_prob_test: np.ndarray
    metrics: Dict


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_weight=None,
) -> TrainArtifacts:
    """
    Train baseline logistic regression model using preprocessing pipeline.
    """
    preprocessor = build_preprocessor(X_train)

    model = LogisticRegression(
        max_iter=1000,
        class_weight=None,
        solver="liblinear",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    if sample_weight is not None:
        pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    else:
        pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob_test = y_pred_test.astype(float)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    report = classification_report(y_test, y_pred_test, output_dict=True)

    fitted_preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(fitted_preprocessor)

    metrics = {
        "train_accuracy": round(float(train_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "classification_report": report,
    }

    return TrainArtifacts(
        model_pipeline=pipeline,
        feature_names=feature_names,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
        y_prob_test=y_prob_test,
        metrics=metrics,
    )


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """
    Wrapper returning plain dict for easy API serialization.
    """
    artifacts = train_logistic_regression(X_train, y_train, X_test, y_test)

    return {
        "model_pipeline": artifacts.model_pipeline,
        "feature_names": artifacts.feature_names,
        "y_pred_train": artifacts.y_pred_train,
        "y_pred_test": artifacts.y_pred_test,
        "y_prob_test": artifacts.y_prob_test,
        "metrics": artifacts.metrics,
    }