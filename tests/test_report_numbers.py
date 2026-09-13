"""Unit tests for the table helpers in scripts/report_numbers.py."""

import pandas as pd
import pytest

from scripts.report_numbers import _agg, _table


def test_single_seed_reports_bare_value():
    assert _agg([0.4396]) == "0.4396"


def test_multiple_seeds_report_mean_and_spread():
    out = _agg([0.40, 0.44, 0.42])
    assert out.startswith("0.4200")
    assert "±" in out, "multi-seed results must carry an uncertainty"


def test_table_groups_seeds_into_one_row():
    df = pd.DataFrame(
        {
            "split": ["temporal"] * 3 + ["athlete"] * 3,
            "pr_auc": [0.43, 0.44, 0.45, 0.57, 0.58, 0.59],
        }
    )
    table = _table(df, ["split"], ["pr_auc"])
    rows = [r for r in table.splitlines() if r.startswith("| ")]
    # header + 2 data rows, not 6 -- seeds collapse into one line each.
    assert len(rows) == 3
    assert "temporal" in rows[1] and "athlete" in rows[2]


def test_table_tolerates_missing_metric():
    df = pd.DataFrame({"g": ["a", "a"], "m": [1.0, float("nan")]})
    assert "1.0000" in _table(df, ["g"], ["m"])


def test_agg_rejects_empty():
    """An empty metric column is a bug upstream, not something to paper over."""
    with pytest.raises(Exception):
        _agg([])


def test_table_renders_all_missing_column_without_crashing():
    """A metric undefined for an entire row must render, not raise."""
    df = pd.DataFrame(
        {"g": ["a", "a"], "defined": [1.0, 2.0], "absent": [float("nan")] * 2}
    )
    table = _table(df, ["g"], ["defined", "absent"])
    assert "—" in table
    assert "1.5000" in table
