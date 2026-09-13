"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest

RNG = np.random.default_rng(0)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A small synthetic frame matching the raw olympics.csv schema."""
    n = 200
    sports = ["Swimming", "Athletics", "Judo", "Ice Hockey"]
    nocs = ["USA", "JPN", "KEN", "BRA"]
    df = pd.DataFrame(
        {
            "id": RNG.integers(1, 60, size=n),
            "name": [f"Athlete {i}" for i in range(n)],
            "sex": RNG.choice(["M", "F"], size=n),
            "age": RNG.normal(25, 4, size=n).round(),
            "height": RNG.normal(175, 10, size=n).round(),
            "weight": RNG.normal(70, 12, size=n).round(),
            "team": "Team",
            "noc": RNG.choice(nocs, size=n),
            "games": "2016 Summer",
            "year": 2016,
            "season": RNG.choice(["Summer", "Winter"], size=n),
            "city": "Rio",
            "sport": RNG.choice(sports, size=n),
            "event": "Some Event",
            "medal": RNG.choice([None, "Gold", "Silver", "Bronze"], size=n, p=[0.85, 0.05, 0.05, 0.05]),
        }
    )
    # Inject missing values for the imputation tests.
    df.loc[df.index[:10], "age"] = np.nan
    df.loc[df.index[5:15], "height"] = np.nan
    df.loc[df.index[12:20], "weight"] = np.nan
    return df


@pytest.fixture
def multi_year_df(raw_df) -> pd.DataFrame:
    """Same schema as ``raw_df`` but spread across several Games.

    ``raw_df`` is single-year by design, which cannot exercise a
    chronological split.
    """
    df = raw_df.copy()
    years = RNG.choice([2000, 2004, 2008, 2012, 2016], size=len(df))
    df["year"] = years
    df["games"] = [f"{y} Summer" for y in years]
    return df
