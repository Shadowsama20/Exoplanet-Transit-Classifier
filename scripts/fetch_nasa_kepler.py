"""Download the Kepler Objects of Interest (KOI) table from the NASA Exoplanet Archive.

Queries the archive's TAP service for the cumulative KOI table, keeps only
CONFIRMED and FALSE POSITIVE dispositions, maps them to a binary label
(1 = CONFIRMED, 0 = FALSE POSITIVE) and writes data/kepler_data.csv.

Usage (from the project root):
    python scripts/fetch_nasa_kepler.py
"""

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
FEATURE_COLUMNS = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq"]
DISPOSITION_COLUMN = "koi_disposition"
LABEL_MAP = {"CONFIRMED": 1, "FALSE POSITIVE": 0}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "kepler_data.csv"
TIMEOUT_SECONDS = 120


def build_query() -> str:
    columns = ", ".join(FEATURE_COLUMNS + [DISPOSITION_COLUMN])
    return f"select {columns} from cumulative"


def download_koi_table() -> pd.DataFrame:
    """Return the KOI cumulative table as a DataFrame, restricted to the needed columns."""
    response = requests.get(
        TAP_URL,
        params={"query": build_query(), "format": "csv"},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only CONFIRMED / FALSE POSITIVE rows, add a numeric label, drop the text column."""
    df = df[df[DISPOSITION_COLUMN].isin(LABEL_MAP)].copy()
    df["label"] = df[DISPOSITION_COLUMN].map(LABEL_MAP).astype(int)
    return df.drop(columns=[DISPOSITION_COLUMN]).reset_index(drop=True)


def main() -> None:
    raw = download_koi_table()
    print(f"Downloaded {len(raw)} KOI rows")

    cleaned = clean(raw)
    print(cleaned["label"].value_counts().rename({1: "CONFIRMED", 0: "FALSE POSITIVE"}).to_string())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(cleaned)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
