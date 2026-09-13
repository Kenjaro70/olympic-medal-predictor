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

# Missingness indicators. Whether height/weight were recorded is itself
# informative: marginally it looks useless (corr with the target is 0.0007),
# but that is Simpson's paradox -- missingness is driven by era
# (corr(year, height_missing) = -0.65), and *within* any given decade a
# missing measurement tracks a much lower medal rate (1980s: 15.4% when
# present vs 2.6% when missing). Median-filling silently destroys that
# signal, so the flags are kept as explicit features.
MISSING_FLAG_COLUMNS = ["height_missing", "weight_missing"]

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
    "height_missing",
    "weight_missing",
    "team_size",
    "field_size",
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

    # ``year`` is a feature and gets standardized downstream, which makes it
    # useless for slicing by calendar year afterwards. Keep an untouched copy.
    out["year_raw"] = out["year"].astype(int)

    # Flags must be computed here, before impute_missing() fills the gaps.
    for col in ("height", "weight"):
        out[f"{col}_missing"] = out[col].isna().astype(int)

    out = add_entry_size_features(out)
    return out.reset_index(drop=True)


def add_entry_size_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add roster size and field size for each (games, event) entry.

    A team gold medal produces one medal row per athlete -- up to 38 rows
    for a single 1908 gymnastics result -- so roster size correlates with
    the target (r = 0.10) without being skill. Small fields are the mirror
    image: events with <=8 entrants have a 55% medal rate versus 12% for
    fields of 50-100. Making both explicit lets the model account for
    opportunity instead of absorbing it into the athlete features.

    Both are computed from the entry list only (no medal information), so
    they are safe to derive before the train/test split.
    """
    out = df.copy()
    out["field_size"] = out.groupby(["games", "event"])["id"].transform("size")
    out["team_size"] = out.groupby(["games", "event", "team"])["id"].transform("size")
    return out


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


def split_by_year(
    df: pd.DataFrame, holdout_from: int = 2012
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: train on <= holdout_from - 1, test on the rest.

    The athlete-grouped split closes athlete leakage but still lets the
    model train on 2016 to predict 1924, which is not a prediction problem.
    It also ignores drift in the target itself: the medal rate falls from
    37.6% in the 1890s to ~14% from the 1960s on, as fields grew. A model
    scored on a random slice of all eras gets credit for interpolating a
    trend it would have to extrapolate in real use.

    With the default cutoff this yields ~233.8k train rows (1896-2008) and
    ~31.5k test rows (2012-2016).
    """
    train = df[df["year"] < holdout_from].copy()
    test = df[df["year"] >= holdout_from].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"holdout_from={holdout_from} leaves an empty split "
            f"({len(train)} train / {len(test)} test rows)"
        )
    return train, test


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
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    split_strategy: str = "athlete",
    holdout_from: int = 2012,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Full preprocessing pipeline: clean, split, fit on train, apply to test.

    ``split_strategy`` selects the holdout:

    - ``"athlete"``: grouped random split, no athlete spans the boundary.
    - ``"temporal"``: train on games before ``holdout_from``, test on the
      rest. Harder and more honest -- this is the number to report.

    Returns (train, test, artifacts) where ``artifacts`` holds everything
    needed to transform a single inference row identically.
    """
    cleaned = clean(df)
    if split_strategy == "temporal":
        train, test = split_by_year(cleaned, holdout_from)
    elif split_strategy == "athlete":
        train, test = split_by_athlete(cleaned, test_size, random_state)
    else:
        raise ValueError(
            f"Unknown split_strategy {split_strategy!r}; "
            "expected 'athlete' or 'temporal'"
        )

    train, medians = impute_missing(train)
    test, _ = impute_missing(test, medians)

    train, encoders = encode_categoricals(train)
    test, _ = encode_categoricals(test, encoders)

    train, scaler = scale_features(train)
    test, _ = scale_features(test, scaler)

    artifacts = {
        "entry_size_defaults": {
            "team_size": float(train["team_size"].median()),
            "field_size": float(train["field_size"].median()),
        },
        "medians": medians,
        "encoders": encoders,
        "scaler": scaler,
        "split_strategy": split_strategy,
        "holdout_from": holdout_from if split_strategy == "temporal" else None,
    }
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

    # Same flags the training rows carry, read before imputation fills them.
    for col in ("height", "weight"):
        row[f"{col}_missing"] = int(
            features.get(col) is None or pd.isna(features.get(col))
        )

    # A single inference row has no event field to count, so fall back to the
    # training medians unless the caller supplied a size explicitly.
    size_defaults = artifacts.get(
        "entry_size_defaults", {"team_size": 1.0, "field_size": 1.0}
    )
    for col in ("team_size", "field_size"):
        value = features.get(col)
        row[col] = size_defaults[col] if value is None else float(value)

    row, _ = impute_missing(row, artifacts["medians"])
    if row.loc[0, "year"] is None or pd.isna(row.loc[0, "year"]):
        row["year"] = 2016
    row, _ = encode_categoricals(row, artifacts["encoders"])
    row, _ = scale_features(row, artifacts["scaler"])
    return row[FEATURE_COLUMNS]
