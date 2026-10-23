"""Generate a synthetic Kepler-like transit dataset.

Features (named after the Kepler KOI table columns):
    koi_period   orbital period [days]
    koi_duration transit duration [hours]
    koi_depth    transit depth [ppm]
    koi_prad     planetary radius [Earth radii]
    koi_teq      equilibrium temperature [K]

Target:
    label  1 = CONFIRMED, 0 = FALSE POSITIVE

Output: data/kepler_data.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

N_SAMPLES = 5000
MISSING_RATE = 0.02  # fraction of values blanked out per feature, to mimic real catalogs
RANDOM_SEED = 42

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "kepler_data.csv"


def generate(n: int, rng: np.random.Generator) -> pd.DataFrame:
    label = rng.binomial(1, 0.5, size=n)
    confirmed = label == 1

    # Confirmed planets: smaller, shallower, slightly longer-period transits with
    # physically plausible radii. False positives: broader spread, often eclipsing
    # binaries (deep, large radius) or noise (shallow, short, random).
    period = np.where(
        confirmed,
        rng.lognormal(mean=np.log(10.0), sigma=0.8, size=n),
        rng.lognormal(mean=np.log(5.0), sigma=1.1, size=n),
    )

    radius = np.where(
        confirmed,
        rng.lognormal(mean=np.log(2.3), sigma=0.45, size=n),
        rng.lognormal(mean=np.log(4.0), sigma=0.9, size=n),
    )

    # Depth roughly scales with (Rp/Rs)^2; Rs ~ 1 Rsun, 1 Rearth/Rsun ~ 84 ppm.
    stellar_radius = rng.normal(1.0, 0.2, size=n).clip(0.4, 2.5)
    depth = (radius * 0.00917 / stellar_radius) ** 2 * 1e6
    depth = depth * rng.lognormal(0.0, 0.15, size=n)

    # Duration scales with period^(1/3) for a Sun-like star, plus noise.
    duration = 13.0 * (period / 365.25) ** (1 / 3) * stellar_radius * rng.lognormal(0.0, 0.2, size=n)

    # Equilibrium temperature: T_star * sqrt(R_star / 2a) * (1 - albedo)^0.25
    star_teff = rng.normal(5600, 500, size=n).clip(3500, 7500)
    a_au = (period / 365.25) ** (2 / 3)
    teq = star_teff * np.sqrt(stellar_radius * 0.00465 / (2 * a_au)) * (1 - 0.3) ** 0.25

    df = pd.DataFrame(
        {
            "koi_period": period,
            "koi_duration": duration,
            "koi_depth": depth,
            "koi_prad": radius,
            "koi_teq": teq,
            "label": label,
        }
    )

    feature_cols = [c for c in df.columns if c != "label"]
    for col in feature_cols:
        mask = rng.random(n) < MISSING_RATE
        df.loc[mask, col] = np.nan

    return df.round(4)


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    df = generate(N_SAMPLES, rng)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    print(df["label"].value_counts().rename({1: "CONFIRMED", 0: "FALSE POSITIVE"}).to_string())


if __name__ == "__main__":
    main()
