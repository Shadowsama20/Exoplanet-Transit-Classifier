"""Train a RandomForestClassifier on the Kepler transit data and save artifacts.

Usage (from the project root):
    python src/train.py
"""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from data_loader import build_preprocessor, load_raw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "random_forest.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42


def main() -> None:
    X, y = load_raw()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Fit the imputer and scaler on training features only, then apply the
    # same fitted transform to the test features to avoid data leakage.
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification report (test set):")
    print(
        classification_report(
            y_test, y_pred, target_names=["FALSE POSITIVE", "CONFIRMED"], digits=3
        )
    )
    print(f"ROC-AUC (test set): {roc_auc_score(y_test, y_proba):.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved preprocessor to {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()
