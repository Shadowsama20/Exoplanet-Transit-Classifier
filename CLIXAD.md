# CLIXAD.md

Guidance for coding agents working in this repository.

## What the project is

A Python machine learning pipeline that classifies Kepler-style exoplanet transit candidates as CONFIRMED (1) or FALSE POSITIVE (0). The data is currently synthetic: `scripts/generate_mock_kepler.py` produces it. There is no real Kepler catalogue in the repo yet.

## Layout

- `scripts/generate_mock_kepler.py`: writes `data/kepler_data.csv` (5,000 rows, fixed seed).
- `src/data_loader.py`: `load_raw()` returns unscaled `(X, y)`. `build_preprocessor()` returns an unfitted sklearn Pipeline (median impute, then StandardScaler). `load_kepler_data()` fits on the full dataset; it is for exploration only, not for training.
- `src/train.py`: stratified 80/20 split, preprocessor fitted on train only, RandomForest (300 trees), prints a classification report and ROC-AUC, saves artifacts.
- `data/`: input CSV. `models/`: joblib artifacts (`random_forest.joblib`, `preprocessor.joblib`).
- `requirements.txt`: pandas, scikit-learn, matplotlib, joblib. No lockfile, no pyproject.
- `.venv/`: local virtualenv (Python 3.14, Windows). Do not edit it.

## Build, test, run

There is no build step and no test suite. Verify changes by running the scripts.

Run from the project root, using the venv interpreter on Windows:

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\generate_mock_kepler.py   # regenerates data/kepler_data.csv
.venv\Scripts\python.exe src\data_loader.py                # prints scaled feature summary
.venv\Scripts\python.exe src\train.py                      # trains, evaluates, writes models/
```

`src/train.py` imports `data_loader` as a top-level module, so run it as a script (`python src\train.py`), not as `python -m src.train`.

## Conventions and gotchas

- Fit preprocessing on training data only. `train.py` is the reference for the correct order: split, then `fit_transform` on train, then `transform` on test. Never fit the scaler or imputer on the full dataset when evaluating.
- Label encoding: 1 = CONFIRMED, 0 = FALSE POSITIVE. Keep this in target names and reports.
- Feature names follow Kepler KOI column names (`koi_period`, `koi_duration`, `koi_depth`, `koi_prad`, `koi_teq`). Keep them in `FEATURE_COLUMNS` in `data_loader.py`; the saved preprocessor depends on the column order.
- Paths are resolved from `Path(__file__)`, not the working directory. Keep it that way.
- Random seeds are fixed (42). Keep them fixed so results are reproducible.
- Scripts are plain Python with docstrings and type hints; there is no linter or formatter configured. Match the existing style.
- The repo has no `.gitignore` and no commits yet. Don't commit `.venv/` or `models/`; add a `.gitignore` first if committing.
- Metrics on the synthetic data (ROC-AUC about 0.80) reflect the generator, not real Kepler performance. Don't tune the model to chase that number without saying so.
- Before changing files, check `git status`; the workspace was not yet under version control at the time of writing.

## Working on this project

- Keep edits minimal and scoped to the requested files.
- After changing code, rerun the affected script and report the output.
- Add dependencies to `requirements.txt` and install them into `.venv`.
