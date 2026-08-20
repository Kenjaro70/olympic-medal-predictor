"""Unit tests for src/preprocess.py."""

import numpy as np
import pandas as pd

from src.preprocess import (
    FEATURE_COLUMNS,
    clean,
    encode_categoricals,
    impute_missing,
    prepare_datasets,
    scale_features,
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
