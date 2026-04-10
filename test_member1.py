from backend.utils.preprocessing import load_dataset, prepare_train_test_data
from backend.model.train import train_baseline_model
from backend.bias.metrics import compute_bias_summary
from backend.bias.mitigation import mitigate_with_reweighing

DATA_PATH = "data/sample_dataset.csv"
TARGET_COL = "approved"
SENSITIVE_COL = "gender"

df = load_dataset(DATA_PATH)

prepared = prepare_train_test_data(
    df=df,
    target_col=TARGET_COL,
    sensitive_col=SENSITIVE_COL,
)

baseline = train_baseline_model(
    X_train=prepared.X_train,
    y_train=prepared.y_train,
    X_test=prepared.X_test,
    y_test=prepared.y_test,
)

baseline_bias = compute_bias_summary(
    y_true=prepared.y_test,
    y_pred=baseline["y_pred_test"],
    sensitive_values=prepared.sens_test,
)

mitigated = mitigate_with_reweighing(
    X_train=prepared.X_train,
    y_train=prepared.y_train,
    sens_train=prepared.sens_train,
    X_test=prepared.X_test,
    y_test=prepared.y_test,
)

mitigated_bias = compute_bias_summary(
    y_true=prepared.y_test,
    y_pred=mitigated["y_pred_test"],
    sensitive_values=prepared.sens_test,
)

print("\nBASELINE MODEL METRICS")
print(baseline["metrics"])

print("\nBASELINE BIAS SUMMARY")
print(baseline_bias)

print("\nMITIGATED MODEL METRICS")
print(mitigated["metrics"])

print("\nMITIGATED BIAS SUMMARY")
print(mitigated_bias)