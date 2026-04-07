from backend.utils.preprocessing import load_and_preprocess_data
from backend.model.train import train_model, evaluate_model
from backend.bias.metrics import generate_bias_report
from backend.bias.mitigation import remove_sensitive_feature

file_path = "data/sample_dataset.csv"
target_col = "selected"
sensitive_col = "gender"

X_train, X_test, y_train, y_test, sens_train, sens_test, encoders = load_and_preprocess_data(
    file_path, target_col, sensitive_col
)

print("Data loaded successfully")

model = train_model(X_train, y_train)
y_pred, accuracy = evaluate_model(model, X_test, y_test)

print("Accuracy before mitigation:", accuracy)

report_before = generate_bias_report(y_pred, sens_test)
print("Bias report before mitigation:", report_before)

X_train_m, X_test_m = remove_sensitive_feature(X_train.copy(), X_test.copy(), sensitive_col)

model_m = train_model(X_train_m, y_train)
y_pred_m, accuracy_m = evaluate_model(model_m, X_test_m, y_test)

print("Accuracy after mitigation:", accuracy_m)

report_after = generate_bias_report(y_pred_m, sens_test)
print("Bias report after mitigation:", report_after)