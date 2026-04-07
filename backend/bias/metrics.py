# backend/bias/metrics.py

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score


def _to_series(values, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(values, name=name)


def _selection_rate(y_pred_group: pd.Series) -> float:
    if len(y_pred_group) == 0:
        return 0.0
    return float((y_pred_group == 1).mean())


def _safe_accuracy(y_true_group: pd.Series, y_pred_group: pd.Series) -> float:
    if len(y_true_group) == 0:
        return 0.0
    return float(accuracy_score(y_true_group, y_pred_group))


def _safe_recall(y_true_group: pd.Series, y_pred_group: pd.Series) -> float:
    if len(y_true_group) == 0:
        return 0.0

    if len(set(y_true_group.tolist())) < 2 and y_true_group.sum() == 0:
        return 0.0

    try:
        return float(recall_score(y_true_group, y_pred_group, zero_division=0))
    except Exception:
        return 0.0


def compute_group_metrics(
    y_true,
    y_pred,
    sensitive_values,
) -> List[Dict]:
    """
    Metrics per sensitive group.
    """
    y_true = _to_series(y_true, "y_true")
    y_pred = _to_series(y_pred, "y_pred")
    sensitive_values = _to_series(sensitive_values, "sensitive").astype(str)

    groups = sorted(sensitive_values.dropna().unique().tolist())
    results = []

    for group in groups:
        mask = sensitive_values == group
        yt = y_true[mask]
        yp = y_pred[mask]

        results.append(
            {
                "group": group,
                "count": int(mask.sum()),
                "selection_rate": round(_selection_rate(yp), 4),
                "accuracy": round(_safe_accuracy(yt, yp), 4),
                "true_positive_rate": round(_safe_recall(yt, yp), 4),
            }
        )

    return results


def compute_bias_summary(
    y_true,
    y_pred,
    sensitive_values,
) -> Dict:
    """
    Compute high-level fairness summary.
    For MVP we use:
    - selection rate difference
    - demographic parity ratio
    - equal opportunity difference
    """
    group_metrics = compute_group_metrics(y_true, y_pred, sensitive_values)

    if len(group_metrics) < 2:
        return {
            "group_metrics": group_metrics,
            "selection_rate_difference": 0.0,
            "demographic_parity_ratio": 1.0,
            "equal_opportunity_difference": 0.0,
            "bias_detected": False,
            "severity": "LOW",
        }

    selection_rates = [g["selection_rate"] for g in group_metrics]
    tprs = [g["true_positive_rate"] for g in group_metrics]

    max_sr = max(selection_rates)
    min_sr = min(selection_rates)
    sr_diff = round(max_sr - min_sr, 4)

    if max_sr == 0:
        dp_ratio = 1.0
    else:
        dp_ratio = round(min_sr / max_sr, 4)

    eo_diff = round(max(tprs) - min(tprs), 4)

    # very simple severity thresholding for hackathon demo
    if sr_diff >= 0.25 or eo_diff >= 0.25:
        severity = "HIGH"
        bias_detected = True
    elif sr_diff >= 0.10 or eo_diff >= 0.10:
        severity = "MEDIUM"
        bias_detected = True
    else:
        severity = "LOW"
        bias_detected = False

    return {
        "group_metrics": group_metrics,
        "selection_rate_difference": sr_diff,
        "demographic_parity_ratio": dp_ratio,
        "equal_opportunity_difference": eo_diff,
        "bias_detected": bias_detected,
        "severity": severity,
    }