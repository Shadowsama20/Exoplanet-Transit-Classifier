"""Load the Kepler transit dataset, impute missing values and scale features."""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "kepler_data.csv"

FEATURE_COLUMNS = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq"]
TARGET_COLUMN = "label"


def load_raw(path: str | Path = DEFAULT_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Read the CSV and return (features, target) without any preprocessing."""
    df = pd.read_csv(path)

    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")

    # Rows without a label cannot be used for supervised training.
    df = df.dropna(subset=[TARGET_COLUMN])

    X = df[FEATURE_COLUMNS].astype(float)
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def build_preprocessor() -> Pipeline:
    """Median-impute missing feature values, then standardise to zero mean and unit variance.

    Fit this on training data only, then reuse it with transform() on other data
    to avoid leaking test-set statistics.
    """
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )


def load_kepler_data(
    path: str | Path = DEFAULT_PATH,
) -> tuple[pd.DataFrame, pd.Series, Pipeline]:
    """Load, impute and scale the dataset.

    Returns
    -------
    X : DataFrame of imputed, scaled features (columns = FEATURE_COLUMNS)
    y : Series of binary labels (1 = CONFIRMED, 0 = FALSE POSITIVE)
    preprocessor : fitted Pipeline, reusable on new data via .transform()
    """
    X_raw, y = load_raw(path)

    preprocessor = build_preprocessor()
    X_scaled = preprocessor.fit_transform(X_raw)

    X = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS, index=X_raw.index)
    return X, y, preprocessor


if __name__ == "__main__":
    X, y, _ = load_kepler_data()
    print(X.describe().round(3).to_string())
    print(y.value_counts().to_string())
