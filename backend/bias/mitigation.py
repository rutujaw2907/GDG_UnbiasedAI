# backend/bias/mitigation.py

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from backend.model.train import train_logistic_regression


def compute_reweighing_sample_weights(
    y_train: pd.Series,
    sens_train: pd.Series,
) -> np.ndarray:
    """
    Simple reweighing:
    weight = P(y) * P(s) / P(y,s)

    This helps reduce imbalance between sensitive group and label combinations.
    """
    y_train = y_train.reset_index(drop=True)
    sens_train = sens_train.astype(str).reset_index(drop=True)

    df = pd.DataFrame({"y": y_train, "s": sens_train})

    total = len(df)
    if total == 0:
        raise ValueError("Training data is empty.")

    p_y = df["y"].value_counts(normalize=True).to_dict()
    p_s = df["s"].value_counts(normalize=True).to_dict()
    p_ys = df.groupby(["y", "s"]).size() / total

    weights = []
    for _, row in df.iterrows():
        y_val = row["y"]
        s_val = row["s"]
        joint = p_ys[(y_val, s_val)]
        w = (p_y[y_val] * p_s[s_val]) / joint if joint > 0 else 1.0
        weights.append(w)

    return np.array(weights, dtype=float)


def mitigate_with_reweighing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sens_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """
    Train a mitigated model using sample reweighing.
    """
    sample_weights = compute_reweighing_sample_weights(y_train, sens_train)

    artifacts = train_logistic_regression(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_weight=sample_weights,
    )

    return {
        "method": "reweighing",
        "model_pipeline": artifacts.model_pipeline,
        "feature_names": artifacts.feature_names,
        "y_pred_test": artifacts.y_pred_test,
        "y_prob_test": artifacts.y_prob_test,
        "metrics": artifacts.metrics,
    }


def mitigate_by_dropping_sensitive_proxy(
    original_df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    proxy_columns: list[str],
) -> pd.DataFrame:
    """
    Optional helper for future use:
    remove manually chosen proxy columns from the original dataset.
    """
    missing = [col for col in proxy_columns if col not in original_df.columns]
    if missing:
        raise ValueError(f"Proxy columns not found: {missing}")

    protected = {target_col, sensitive_col}
    removable = [col for col in proxy_columns if col not in protected]

    return original_df.drop(columns=removable)