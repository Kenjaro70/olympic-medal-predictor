"""Data cleaning and feature engineering for the Olympic medal prediction model.

Every fit-style function (impute_missing, encode_categoricals, scale_features)
returns the fitted state alongside the transformed frame so the exact same
transformation can be replayed on the test set and at inference time without
re-fitting on unseen data (no leakage). All functions return copies and never
mutate the input dataframe.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

SEX_MAP = {"M": 0, "F": 1}
SEASON_MAP = {"Summer": 0, "Winter": 1}

IMPUTE_COLUMNS = ["age", "height", "weight"]

# Final numeric feature matrix consumed by every model.
FEATURE_COLUMNS = [
    "sex",
    "season",
    "age",
    "height",
    "weight",
    "year",
    "noc_medal_rate",
    "sport_medal_rate",
]

TARGET_COLUMN = "medal_won"


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw athlete_events / olympics CSV."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns, drop duplicates, and build the binary target.

    The target ``medal_won`` is 1 when the athlete won any medal (Gold,
    Silver or Bronze) in that event entry, 0 otherwise. Rows with an
    unrecognized sex or season code are dropped (a handful of rows).
    """
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    out = out.drop_duplicates()
    out[TARGET_COLUMN] = out["medal"].notna().astype(int)
    out = out[out["sex"].isin(SEX_MAP) & out["season"].isin(SEASON_MAP)]
    return out.reset_index(drop=True)


def split_by_athlete(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grouped train/test split so the same athlete never appears in both.

    An athlete typically has several event entries; a plain random split
    would leak athlete identity (height/weight/nationality) across the
    boundary and inflate test metrics.
    """
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(df, groups=df["id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def impute_missing(
    df: pd.DataFrame, medians: dict[str, float] | None = None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fill missing age/height/weight with medians.

    When ``medians`` is None the medians are computed from ``df`` (training
    fit); otherwise the provided values are applied unchanged (test /
    inference transform).
    """
    out = df.copy()
    if medians is None:
        medians = {c: float(out[c].median()) for c in IMPUTE_COLUMNS}
    for col in IMPUTE_COLUMNS:
        # to_numeric handles inference rows where missing values arrive as None
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(medians[col])
    return out, medians


def encode_categoricals(
    df: pd.DataFrame, encoders: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Encode sex/season as binary and NOC/sport as historical medal rates.

    NOC (country) and sport have hundreds of levels, so instead of one-hot
    encoding they are target-encoded with the medal rate observed in the
    *training* data only. Categories unseen at fit time fall back to the
    global training medal rate. When ``encoders`` is provided no target
    statistics are recomputed, so applying this to test/inference data
    cannot leak.
    """
    out = df.copy()
    if encoders is None:
        global_rate = float(out[TARGET_COLUMN].mean())
        encoders = {
            "sex_map": dict(SEX_MAP),
            "season_map": dict(SEASON_MAP),
            "global_rate": global_rate,
            "noc_rates": out.groupby("noc")[TARGET_COLUMN].mean().to_dict(),
            "sport_rates": out.groupby("sport")[TARGET_COLUMN].mean().to_dict(),
        }
    out["sex"] = out["sex"].map(encoders["sex_map"])
    out["season"] = out["season"].map(encoders["season_map"])
    out["noc_medal_rate"] = (
        out["noc"].map(encoders["noc_rates"]).fillna(encoders["global_rate"])
    )
    out["sport_medal_rate"] = (
        out["sport"].map(encoders["sport_rates"]).fillna(encoders["global_rate"])
    )
    return out, encoders


def scale_features(
    df: pd.DataFrame, scaler: StandardScaler | None = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize the feature columns (fit on train only)."""
    out = df.copy()
    if scaler is None:
        scaler = StandardScaler().fit(out[FEATURE_COLUMNS])
    out[FEATURE_COLUMNS] = scaler.transform(out[FEATURE_COLUMNS])
    return out, scaler


def prepare_datasets(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Full preprocessing pipeline: clean, split, fit on train, apply to test.

    Returns (train, test, artifacts) where ``artifacts`` holds everything
    needed to transform a single inference row identically.
    """
    cleaned = clean(df)
    train, test = split_by_athlete(cleaned, test_size, random_state)

    train, medians = impute_missing(train)
    test, _ = impute_missing(test, medians)

    train, encoders = encode_categoricals(train)
    test, _ = encode_categoricals(test, encoders)

    train, scaler = scale_features(train)
    test, _ = scale_features(test, scaler)

    artifacts = {"medians": medians, "encoders": encoders, "scaler": scaler}
    return train, test, artifacts


def features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS], df[TARGET_COLUMN]


def transform_inference(features: dict, artifacts: dict) -> pd.DataFrame:
    """Turn one raw feature dict from the LLM parser into a model-ready row.

    Expected keys: sex ('M'/'F'), age, height, weight, noc, sport, season
    ('Summer'/'Winter'), year. Missing age/height/weight fall back to the
    training medians; unseen NOC/sport fall back to the global medal rate.
    """
    row = pd.DataFrame(
        [
            {
                "sex": features.get("sex"),
                "season": features.get("season", "Summer"),
                "age": features.get("age"),
                "height": features.get("height"),
                "weight": features.get("weight"),
                "year": features.get("year"),
                "noc": features.get("noc"),
                "sport": features.get("sport"),
            }
        ]
    )
    row, _ = impute_missing(row, artifacts["medians"])
    if row.loc[0, "year"] is None or pd.isna(row.loc[0, "year"]):
        row["year"] = 2016
    row, _ = encode_categoricals(row, artifacts["encoders"])
    row, _ = scale_features(row, artifacts["scaler"])
    return row[FEATURE_COLUMNS]
