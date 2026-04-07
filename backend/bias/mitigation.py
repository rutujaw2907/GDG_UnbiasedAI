def remove_sensitive_feature(X_train, X_test, sensitive_col):
    if sensitive_col in X_train.columns:
        X_train = X_train.drop(columns=[sensitive_col])
    if sensitive_col in X_test.columns:
        X_test = X_test.drop(columns=[sensitive_col])
    return X_train, X_test