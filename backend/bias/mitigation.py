from model.train import train_model

def mitigate_bias(df):
    df = df.copy()

    # Remove sensitive feature (first column)
    sensitive_column = df.columns[0]
    df = df.drop(columns=[sensitive_column])

    model, X_test, y_test = train_model(df)
    y_pred = model.predict(X_test)

    return {
        "message": "Bias mitigation applied",
        "new_prediction_mean": float(y_pred.mean())
    }