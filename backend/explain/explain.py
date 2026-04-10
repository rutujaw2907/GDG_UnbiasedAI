def get_feature_importance(model):
    if not hasattr(model, "coef_"):
        return {"error": "Model does not support feature importance"}

    importance = model.coef_[0]

    result = {}
    for i, val in enumerate(importance):
        result[f"feature_{i}"] = float(val)

    return {
        "feature_importance": result
    }