# Kepler Exoplanet Transit Classifier

A machine learning pipeline that classifies Kepler transit candidates as
**CONFIRMED** planets (label 1) or **FALSE POSITIVES** (label 0) with a
RandomForest classifier.

**The model is trained on real NASA telescope data:** the cumulative Kepler
Objects of Interest (KOI) table from the NASA Exoplanet Archive. It was
trained on 7,587 labelled candidates, and on the held-out 20% test split it
reaches a **ROC-AUC of 0.958** and **89.7% accuracy**.

## Features

Each candidate is described by five features from the KOI table:

| Feature        | Description                        | Unit          |
|----------------|------------------------------------|---------------|
| `koi_period`   | Orbital period                     | days          |
| `koi_duration` | Transit duration                   | hours         |
| `koi_depth`    | Transit depth                      | ppm           |
| `koi_prad`     | Planetary radius                   | Earth radii   |
| `koi_teq`      | Equilibrium temperature            | K             |

Missing values (about 3% of rows for depth, radius and temperature) are filled
with the training-set median.

## Pipeline

```
scripts/fetch_nasa_kepler.py  ->  data/kepler_data.csv
                                  (NASA Exoplanet Archive KOI table, CONFIRMED / FALSE POSITIVE only)
src/data_loader.py            ->  load, impute (median), scale (StandardScaler)
src/train.py                  ->  80/20 stratified split, fit preprocessor on train,
                                  train RandomForest, save to models/
src/evaluate.py               ->  test-set metrics, ROC curve, feature importance -> results/
app.py                        ->  Gradio web UI for single-candidate predictions
```

## Project layout

```
scripts/fetch_nasa_kepler.py      Downloads and cleans the real KOI data
scripts/generate_mock_kepler.py   Synthetic data generator (offline fallback)
src/data_loader.py                Loading, imputation and scaling
src/train.py                      Training and evaluation metrics
src/evaluate.py                   Charts for the saved model
app.py                            Gradio interface
data/                             Dataset (git-ignored; folder kept via .gitkeep)
models/                           Saved model and preprocessor (git-ignored)
results/                          Generated charts (git-ignored)
requirements.txt                  Python dependencies
```

## Setup

Requires Python 3.10 or newer and an internet connection for the data download.

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
# 1. Download the KOI table from the NASA Exoplanet Archive -> data/kepler_data.csv
python scripts/fetch_nasa_kepler.py

# 2. Train the model -> models/random_forest.joblib, models/preprocessor.joblib
python src/train.py

# 3. Evaluate on the test split -> results/roc_curve.png, results/feature_importance.png
python src/evaluate.py

# 4. Launch the web interface (http://127.0.0.1:7860)
python app.py
```

`src/train.py` and `src/evaluate.py` import sibling modules directly, so run
them as scripts (`python src/train.py`), not as modules (`python -m src.train`).

To inspect the scaled feature summary, run `python src/data_loader.py`.

### Offline fallback

`scripts/generate_mock_kepler.py` produces a synthetic dataset with the same
columns. It writes to `data/mock_kepler_data.csv`, so it never touches the real
`data/kepler_data.csv`. The training and evaluation scripts still read the real
file; the mock file is only for experiments and tests.

## Method notes

- The imputer and scaler are fitted on the training split only, then applied to
  the test split. Fitting on the full dataset would leak test statistics.
- The split is stratified and seeded (`random_state=42`), so results are reproducible.
- The RandomForest uses 300 trees. Feature importance is the mean decrease in impurity.
- Predictions in `app.py` are probabilities from the same model; they are not
  calibrated scientific claims about a specific star.

## License

Not specified yet.
