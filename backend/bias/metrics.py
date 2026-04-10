import numpy as np

def calculate_bias(model, X_test, y_test):
    y_pred = model.predict(X_test)

    sensitive_feature = X_test.iloc[:, 0]

    group_0 = y_pred[sensitive_feature == 0]
    group_1 = y_pred[sensitive_feature == 1]

    if len(group_0) == 0 or len(group_1) == 0:
        return {"error": "Not enough data for bias calculation"}

    rate_0 = np.mean(group_0)
    rate_1 = np.mean(group_1)

    bias_score = abs(rate_0 - rate_1)

    return {
        "group_0_rate": float(rate_0),
        "group_1_rate": float(rate_1),
        "bias_score": float(bias_score),
        "bias_detected": bias_score > 0.1
    }