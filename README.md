# Kepler Exoplanet Transit Classifier

A machine learning pipeline that classifies Kepler-style transit candidates as
**CONFIRMED** planets (label 1) or **FALSE POSITIVES** (label 0) using a
RandomForest classifier.

> The data is synthetic for now. `scripts/generate_mock_kepler.py` generates a
> dataset that mimics the shape and feature names of the Kepler KOI catalogue.
> Metrics reflect the generator, not real Kepler performance.

## Features

Each candidate is described by five features:

| Feature        | Description                        | Unit          |
|----------------|------------------------------------|---------------|
| `koi_period`   | Orbital period                     | days          |
| `koi_duration` | Transit duration                   | hours         |
| `koi_depth`    | Transit depth                      | ppm           |
| `koi_prad`     | Planetary radius                   | Earth radii   |
| `koi_teq`      | Equilibrium temperature            | K             |

## Pipeline

```
scripts/generate_mock_kepler.py  ->  data/kepler_data.csv
src/data_loader.py               ->  load, impute (median), scale (StandardScaler)
src/train.py                     ->  80/20 stratified split, fit preprocessor on train,
                                     train RandomForest, save to models/
src/evaluate.py                  ->  test-set metrics, ROC curve, feature importance -> results/
```

## Project layout

```
scripts/generate_mock_kepler.py   Synthetic data generator
src/data_loader.py                Loading, imputation and scaling
src/train.py                      Training and evaluation metrics
src/evaluate.py                   Charts for the saved model
data/                             Generated dataset (git-ignored)
models/                           Saved model and preprocessor (git-ignored)
results/                          Generated charts (git-ignored)
requirements.txt                  Python dependencies
```

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

Run these from the project root, in order:

```bash
# 1. Generate the synthetic dataset -> data/kepler_data.csv
python scripts/generate_mock_kepler.py

# 2. Train the model -> models/random_forest.joblib, models/preprocessor.joblib
python src/train.py

# 3. Evaluate on the test split -> results/roc_curve.png, results/feature_importance.png
python src/evaluate.py
```

`src/train.py` and `src/evaluate.py` import sibling modules directly, so run
them as scripts (`python src/train.py`), not as modules (`python -m src.train`).

To inspect the scaled feature summary, run `python src/data_loader.py`.

## Method notes

- The imputer and scaler are fitted on the training split only, then applied to
  the test split. Fitting on the full dataset would leak test statistics.
- The split is stratified and seeded (`random_state=42`), so results are reproducible.
- The RandomForest uses 300 trees. Feature importance is the mean decrease in impurity.

## Results

On the synthetic test split, the model reaches a ROC-AUC of about 0.80 and an
accuracy of about 74%. Exact numbers come from `src/train.py` and
`src/evaluate.py`.

## License

Not specified yet.
