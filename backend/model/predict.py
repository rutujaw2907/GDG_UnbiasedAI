# backend/model/predict.py

from __future__ import annotations

from typing import Dict

import pandas as pd


def predict_with_model(model_pipeline, X: pd.DataFrame) -> Dict:
    """
    Run prediction on given data using trained pipeline.
    """
    if model_pipeline is None:
        raise ValueError("Model pipeline is not available.")

    predictions = model_pipeline.predict(X).tolist()

    if hasattr(model_pipeline, "predict_proba"):
        probabilities = model_pipeline.predict_proba(X)[:, 1].tolist()
    else:
        probabilities = [float(p) for p in predictions]

    return {
        "predictions": predictions,
        "probabilities": probabilities,
    }