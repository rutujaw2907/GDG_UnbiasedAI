import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(file_path, target_col, sensitive_col):
    df = pd.read_csv(file_path)

    # Drop rows with missing target or sensitive values
    df = df.dropna(subset=[target_col, sensitive_col])

    # Fill missing values for other columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Keep original sensitive column separately
    sensitive_data = df[sensitive_col].copy()

    # Encode categorical columns
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive_data, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, sens_train, sens_test, label_encoders