from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    sens_train: pd.Series
    sens_test: pd.Series
    feature_columns: List[str]


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load CSV dataset and return DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise ValueError(f"Dataset file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"Dataset file '{file_path}' is empty. Please add valid CSV data with headers."
        )
    except Exception as e:
        raise ValueError(f"Failed to read dataset '{file_path}': {str(e)}")

    if df.empty:
        raise ValueError("Uploaded dataset is empty.")

    if len(df.columns) == 0:
        raise ValueError("Dataset has no columns. Please check the CSV format.")

    return df


def validate_columns(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
) -> None:
    """
    Validate required columns.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found in dataset.")

    if target_col == sensitive_col:
        raise ValueError("Target column and sensitive column cannot be the same.")


def clean_target(y: pd.Series) -> pd.Series:
    """
    Convert target column to binary numeric values if possible.
    Supported:
      - already numeric 0/1
      - yes/no, true/false, approved/rejected, selected/rejected, etc.
    """
    if pd.api.types.is_numeric_dtype(y):
        unique_vals = sorted(y.dropna().unique().tolist())
        if len(unique_vals) > 2:
            raise ValueError("Target column must be binary for this MVP.")
        return y.astype(int)

    y_str = y.astype(str).str.strip().str.lower()

    positive_map = {
        "1", "yes", "true", "approved", "approve", "selected", "accept", "accepted"
    }
    negative_map = {
        "0", "no", "false", "rejected", "reject", "denied", "not selected", "declined"
    }

    mapped = []
    for val in y_str:
        if val in positive_map:
            mapped.append(1)
        elif val in negative_map:
            mapped.append(0)
        else:
            raise ValueError(
                f"Unsupported target value '{val}'. Target must be binary."
            )

    return pd.Series(mapped, index=y.index, name=y.name)


def split_features_target_sensitive(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Separate X, y, and sensitive attribute.
    Sensitive column is removed from X for baseline training.
    """
    validate_columns(df, target_col, sensitive_col)

    y = clean_target(df[target_col].copy())
    sens = df[sensitive_col].astype(str).fillna("Unknown")

    X = df.drop(columns=[target_col, sensitive_col]).copy()

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns left after removing target and sensitive columns.")

    return X, y, sens


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline:
    - numeric: median impute
    - categorical: most frequent impute + one-hot encode
    """
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extract transformed feature names after fit.
    """
    try:
        feature_names = preprocessor.get_feature_names_out()
        return feature_names.tolist()
    except Exception:
        return []


def prepare_train_test_data(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    """
    Split dataset into train/test.
    """
    from sklearn.model_selection import train_test_split

    X, y, sens = split_features_target_sensitive(df, target_col, sensitive_col)

    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X,
        y,
        sens,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        sens_train=sens_train,
        sens_test=sens_test,
        feature_columns=X.columns.tolist(),
    )


def dataset_bias_overview(df: pd.DataFrame, sensitive_col: str) -> Dict:
    """
    Simple dataset distribution overview for frontend display.
    """
    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found.")

    counts = df[sensitive_col].astype(str).fillna("Unknown").value_counts(dropna=False)
    total = int(counts.sum())

    distribution = []
    for group, count in counts.items():
        distribution.append(
            {
                "group": str(group),
                "count": int(count),
                "percentage": round((count / total) * 100, 2),
            }
        )

    return {
        "total_rows": total,
        "sensitive_distribution": distribution,
    }