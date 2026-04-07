import pandas as pd


def calculate_selection_rate(y_pred, sensitive_data):
    results = pd.DataFrame({
        "sensitive": sensitive_data.values,
        "prediction": y_pred
    })

    group_rates = results.groupby("sensitive")["prediction"].mean().to_dict()
    return group_rates


def calculate_bias_gap(group_rates):
    rates = list(group_rates.values())
    if len(rates) < 2:
        return 0.0
    return abs(max(rates) - min(rates))


def get_bias_severity(bias_gap):
    if bias_gap < 0.1:
        return "Low"
    elif bias_gap < 0.2:
        return "Medium"
    else:
        return "High"


def generate_bias_report(y_pred, sensitive_data):
    group_rates = calculate_selection_rate(y_pred, sensitive_data)
    bias_gap = calculate_bias_gap(group_rates)
    severity = get_bias_severity(bias_gap)

    return {
        "group_selection_rates": group_rates,
        "bias_gap": round(bias_gap, 4),
        "severity": severity
    }