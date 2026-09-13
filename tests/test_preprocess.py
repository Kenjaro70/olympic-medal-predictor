"""Unit tests for src/preprocess.py."""

import numpy as np
import pandas as pd

import pytest

from src.preprocess import (
    FEATURE_COLUMNS,
    MISSING_FLAG_COLUMNS,
    add_entry_size_features,
    clean,
    encode_categoricals,
    impute_missing,
    prepare_datasets,
    scale_features,
    split_by_year,
)


def test_clean_builds_binary_target(raw_df):
    cleaned = clean(raw_df)
    assert set(cleaned["medal_won"].unique()) <= {0, 1}
    medal_rows = cleaned["medal"].notna()
    assert (cleaned.loc[medal_rows, "medal_won"] == 1).all()
    assert (cleaned.loc[~medal_rows, "medal_won"] == 0).all()


def test_impute_fills_all_missing_values(raw_df):
    cleaned = clean(raw_df)
    assert cleaned[["age", "height", "weight"]].isna().any().any()
    imputed, medians = impute_missing(cleaned)
    assert not imputed[["age", "height", "weight"]].isna().any().any()
    # A second frame transformed with the fitted medians uses those exact values.
    other = cleaned.copy()
    other.loc[other.index[0], "age"] = np.nan
    other_imputed, _ = impute_missing(other, medians)
    assert other_imputed.loc[other.index[0], "age"] == medians["age"]


def test_encode_categoricals_maps_and_handles_unseen(raw_df):
    cleaned, _ = impute_missing(clean(raw_df))
    encoded, encoders = encode_categoricals(cleaned)
    assert set(encoded["sex"].unique()) <= {0, 1}
    assert set(encoded["season"].unique()) <= {0, 1}
    assert encoded["noc_medal_rate"].between(0, 1).all()
    # An NOC never seen at fit time falls back to the global medal rate.
    unseen = cleaned.iloc[[0]].copy()
    unseen["noc"] = "ZZZ"
    unseen_encoded, _ = encode_categoricals(unseen, encoders)
    assert unseen_encoded["noc_medal_rate"].iloc[0] == encoders["global_rate"]


def test_scale_features_standardizes(raw_df):
    cleaned, _ = impute_missing(clean(raw_df))
    encoded, _ = encode_categoricals(cleaned)
    scaled, _ = scale_features(encoded)
    means = scaled[FEATURE_COLUMNS].mean().abs()
    stds = scaled[FEATURE_COLUMNS].std(ddof=0)
    assert (means < 1e-9).all()
    # Constant columns (e.g. single year) stay at 0; others scale to unit std.
    assert ((stds - 1).abs() < 1e-9)[stds > 0].all()


def test_functions_do_not_mutate_input(raw_df):
    original = raw_df.copy(deep=True)
    cleaned = clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, original)

    before_impute = cleaned.copy(deep=True)
    imputed, medians = impute_missing(cleaned)
    pd.testing.assert_frame_equal(cleaned, before_impute)

    before_encode = imputed.copy(deep=True)
    encoded, encoders = encode_categoricals(imputed)
    pd.testing.assert_frame_equal(imputed, before_encode)

    before_scale = encoded.copy(deep=True)
    scale_features(encoded)
    pd.testing.assert_frame_equal(encoded, before_scale)


def test_prepare_datasets_keeps_athletes_separate(raw_df):
    train, test, artifacts = prepare_datasets(raw_df, test_size=0.3, random_state=0)
    assert set(train["id"]).isdisjoint(set(test["id"]))
    assert {"medians", "encoders", "scaler"} <= set(artifacts)
    assert not train[FEATURE_COLUMNS].isna().any().any()
    assert not test[FEATURE_COLUMNS].isna().any().any()


# --- missingness flags -------------------------------------------------


def test_clean_flags_missing_measurements_before_imputation(raw_df):
    cleaned = clean(raw_df)
    for col in ("height", "weight"):
        flag = cleaned[f"{col}_missing"]
        assert set(flag.unique()) <= {0, 1}
        # The flag must record the ORIGINAL gaps, not post-imputation state.
        assert (flag == cleaned[col].isna().astype(int)).all()
    assert cleaned["height_missing"].sum() > 0, "fixture should contain gaps"


def test_missing_flags_survive_imputation(raw_df):
    cleaned = clean(raw_df)
    imputed, _ = impute_missing(cleaned)
    assert not imputed["height"].isna().any()
    # Imputation fills the value but must not erase the evidence.
    assert imputed["height_missing"].sum() == cleaned["height_missing"].sum()


def test_missing_flags_are_model_features():
    assert set(MISSING_FLAG_COLUMNS) <= set(FEATURE_COLUMNS)


# --- entry size features -----------------------------------------------


def test_entry_size_counts_roster_and_field(raw_df):
    sized = add_entry_size_features(raw_df)
    grouped = raw_df.groupby(["games", "event"]).size()
    for (games, event), expected in grouped.items():
        rows = sized[(sized["games"] == games) & (sized["event"] == event)]
        assert (rows["field_size"] == expected).all()
    assert (sized["team_size"] <= sized["field_size"]).all()


# --- temporal split ----------------------------------------------------


def test_split_by_year_is_chronological(multi_year_df):
    cleaned = clean(multi_year_df)
    train, test = split_by_year(cleaned, holdout_from=2012)
    assert train["year"].max() < 2012
    assert test["year"].min() >= 2012
    assert len(train) + len(test) == len(cleaned)


def test_split_by_year_rejects_empty_holdout(multi_year_df):
    cleaned = clean(multi_year_df)
    with pytest.raises(ValueError, match="empty split"):
        split_by_year(cleaned, holdout_from=3000)


def test_prepare_datasets_temporal_has_no_future_in_train(multi_year_df):
    train, test, artifacts = prepare_datasets(
        multi_year_df, split_strategy="temporal", holdout_from=2012
    )
    # ``year`` is a feature column, so it arrives standardized here. The
    # scaler is monotonic and fitted once, so the ordering still proves the
    # holdout is strictly in the future.
    assert train["year"].max() < test["year"].min()
    assert artifacts["split_strategy"] == "temporal"
    assert artifacts["holdout_from"] == 2012
    assert not train[FEATURE_COLUMNS].isna().any().any()
    assert not test[FEATURE_COLUMNS].isna().any().any()


def test_prepare_datasets_rejects_unknown_strategy(raw_df):
    with pytest.raises(ValueError, match="Unknown split_strategy"):
        prepare_datasets(raw_df, split_strategy="nonsense")


def test_year_raw_survives_scaling(multi_year_df):
    """The calendar year must stay readable after ``year`` is standardized."""
    train, test, _ = prepare_datasets(
        multi_year_df, split_strategy="temporal", holdout_from=2012
    )
    assert train["year_raw"].max() < 2012 <= test["year_raw"].min()
    assert set(train["year_raw"]) <= {2000, 2004, 2008}
    assert "year_raw" not in FEATURE_COLUMNS, "raw year must not reach the model"


def test_walk_forward_folds_do_not_overlap(multi_year_df):
    """Each fold's training rows must predate its own validation window."""
    from src.train import build_walk_forward_folds

    train, _test, _ = prepare_datasets(
        multi_year_df, split_strategy="temporal", holdout_from=2012
    )
    folds = build_walk_forward_folds(
        train, window=4, n_folds=3, min_val_rows=5, min_fit_rows=5
    )
    assert folds, "fixture should yield at least one fold"
    for lo, hi, fit, val in folds:
        assert val["year_raw"].between(lo, hi).all()
        assert (fit["year_raw"] < lo).all(), "fit window must precede validation"


def test_walk_forward_stops_when_history_runs_out(multi_year_df):
    """Folds must not be produced from slices too small to be informative."""
    from src.train import build_walk_forward_folds

    train, _test, _ = prepare_datasets(
        multi_year_df, split_strategy="temporal", holdout_from=2012
    )
    with pytest.raises(ValueError, match="no usable walk-forward folds"):
        build_walk_forward_folds(train, window=4, n_folds=3)
