import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import build_features


def make_daily(n_days=100):
    """Create a complete daily series with identifiable quantities."""
    return pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=n_days, freq="D"),
        "Category": "Test Category",
        "Quantity": np.arange(1, n_days + 1, dtype=float),
    })


@pytest.mark.parametrize(
    "X,Y",
    [(7, 7), (14, 1), (28, 14)],
)
def test_target_matches_future_window(X, Y):
    daily = make_daily()
    data = build_features(
        daily,
        "Test Category",
        X=X,
        Y=Y,
        holiday_dates=frozenset(),
    )

    target = config.target_name(X, Y)

    assert not data.empty

    for _, row in data.iterrows():
        t = daily.index[daily["Date"] == row["Date"]][0]

        expected = daily["Quantity"].iloc[
            t + Y : t + Y + X
        ].sum()

        assert row[target] == pytest.approx(expected)


def test_lags_and_rolling_features_use_past_only():
    daily = make_daily()
    data = build_features(
        daily,
        "Test Category",
        X=7,
        Y=7,
        holiday_dates=frozenset(),
    )

    row = data.iloc[0]
    t = daily.index[daily["Date"] == row["Date"]][0]

    assert row["lag_1"] == daily.loc[t - 1, "Quantity"]
    assert row["lag_7"] == daily.loc[t - 7, "Quantity"]

    expected_rolling_sum = daily["Quantity"].iloc[t - 7:t].sum()
    assert row["rolling_sum_7"] == pytest.approx(
        expected_rolling_sum
    )


def test_future_sales_do_not_change_past_features():
    daily = make_daily()

    original = build_features(
        daily,
        "Test Category",
        X=7,
        Y=7,
        holiday_dates=frozenset(),
    )

    reference_date = original.iloc[0]["Date"]

    modified_daily = daily.copy()
    modified_daily.loc[
        modified_daily["Date"] > reference_date,
        "Quantity",
    ] = 999999.0

    modified = build_features(
        modified_daily,
        "Test Category",
        X=7,
        Y=7,
        holiday_dates=frozenset(),
    )

    original_row = original.loc[
        original["Date"] == reference_date
    ].iloc[0]

    modified_row = modified.loc[
        modified["Date"] == reference_date
    ].iloc[0]

    feature_columns = [
        name
        for name in config.FEATURES
        if name in original.columns
    ]

    pd.testing.assert_series_equal(
        original_row[feature_columns],
        modified_row[feature_columns],
    )

    target = config.target_name(7, 7)

    assert original_row[target] != modified_row[target]