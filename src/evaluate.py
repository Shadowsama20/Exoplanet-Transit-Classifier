"""Evaluate the saved RandomForest on the held-out test split.

Loads models/random_forest.joblib and models/preprocessor.joblib, recreates the
same 80/20 split used in train.py, and writes two charts to results/:
    roc_curve.png            ROC curve with AUC in the legend
    feature_importance.png   RandomForest impurity-based feature importances

Usage (from the project root):
    python src/evaluate.py
"""

from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # render to files; no display needed

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import RocCurveDisplay, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from data_loader import FEATURE_COLUMNS, load_raw  # noqa: E402
from train import MODEL_PATH, PREPROCESSOR_PATH, RANDOM_STATE, TEST_SIZE  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
ROC_PATH = RESULTS_DIR / "roc_curve.png"
IMPORTANCE_PATH = RESULTS_DIR / "feature_importance.png"


def main() -> None:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Same split as train.py, so the evaluation set is the true held-out test set.
    X, y = load_raw()
    _, X_test_raw, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_test = preprocessor.transform(X_test_raw)  # transform only; never refit

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC (test set): {auc:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(
        y_test, y_proba, name="RandomForest", ax=ax, plot_chance_level=True
    )
    ax.set_title("ROC Curve (test set)")
    fig.tight_layout()
    fig.savefig(ROC_PATH, dpi=150)
    plt.close(fig)

    # Feature importance. The preprocessor keeps column order, so the importances
    # line up with FEATURE_COLUMNS.
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    importances.plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(IMPORTANCE_PATH, dpi=150)
    plt.close(fig)

    print(f"Saved ROC curve to {ROC_PATH}")
    print(f"Saved feature importance chart to {IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
