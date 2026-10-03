import numpy as np
import pandas as pd
import pytest

from src.features import _check_contiguous
from src.models import horizon_scaled_baseline
from src.splits import (
    purged_chronological_split,
    split_overlap_report,
    target_window_overlap_ratio,
)


def test_horizon_scaled_baseline():
    data = pd.DataFrame({"rolling_sum_7": [350.0, 700.0]})

    np.testing.assert_allclose(
        horizon_scaled_baseline(data, X=7),
        [350.0, 700.0],
    )
    np.testing.assert_allclose(
        horizon_scaled_baseline(data, X=14),
        [700.0, 1400.0],
    )
    np.testing.assert_allclose(
        horizon_scaled_baseline(data, X=28),
        [1400.0, 2800.0],
    )


@pytest.mark.parametrize(
    "dates",
    [
        ["2026-01-01", "2026-01-03"],  # Missing day
        ["2026-01-01", "2026-01-01"],  # Duplicate date
        ["2026-01-01", None],          # Missing date
    ],
)
def test_invalid_daily_dates_are_rejected(dates):
    data = pd.DataFrame({"Date": pd.to_datetime(dates)})

    with pytest.raises(ValueError):
        _check_contiguous(data, "Test Category")


def test_valid_daily_dates_are_accepted():
    data = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=3, freq="D")
    })

    _check_contiguous(data, "Test Category")


def test_target_window_overlap():
    train = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-15"])
    })
    validation = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-07", "2026-02-01"])
    })

    assert target_window_overlap_ratio(
        train, validation, X=7
    ) == 0.5


def test_purged_chronological_split():
    data = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=100, freq="D"),
        "Quantity": np.arange(100),
    })

    X, Y = 7, 7

    splits = purged_chronological_split(
        data,
        X=X,
        Y=Y,
        train_size=0.60,
        val_size=0.20,
    )

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    offset = pd.Timedelta(days=X + Y - 1)

    assert train["Date"].max() + offset < validation["Date"].min()
    assert validation["Date"].max() + offset < test["Date"].min()

    report = split_overlap_report(splits, X=X)
    assert (report["Overlapping rows (%)"] == 0).all()

    assert len(train) + len(validation) + len(test) < len(data)