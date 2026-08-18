"""Feature engineering for single-category forecasting.

This consolidates ``prepare_single_category_dataset`` (02 notebook) and
``preprocess`` (pipeline notebook), which had drifted apart. Where they
disagreed, the correct version was kept — see notes on weekly_mean below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_features(
    daily: pd.DataFrame,
    category: str,
    X: int = config.DEFAULT_X,
    Y: int = config.DEFAULT_Y,
    holiday_dates=config.HOLIDAY_DATES,
) -> pd.DataFrame:
    """Build the modelling frame for one category.

    Target is total quantity over the X days starting Y days after the
    current date. All features use only information available at the current
    date — rolling windows are shifted by 1 so today's quantity never enters
    a feature describing today.
    """
    df = (
        daily[daily["Category"] == category]
        .sort_values("Date")
        .reset_index(drop=True)
        .copy()
    )
    df["Date"] = pd.to_datetime(df["Date"])

    _check_contiguous(df, category)

    keep = [c for c in ["Date", "Category", "Quantity"] if c in df.columns]
    df = df[keep].copy()

    # --- Target ----------------------------------------------------------
    quantities = df["Quantity"].to_numpy()
    target_col = config.target_name(X, Y)
    df[target_col] = [
        quantities[i + Y : i + Y + X].sum() if i + Y + X <= len(df) else np.nan
        for i in range(len(df))
    ]

    # --- Lags ------------------------------------------------------------
    for lag in list(range(1, 8)) + [14, 28]:
        df[f"lag_{lag}"] = df["Quantity"].shift(lag)

    # --- Rolling statistics (shifted by 1: past observations only) -------
    shifted = df["Quantity"].shift(1)
    for window in (7, 14, 28):
        roll = shifted.rolling(window)
        df[f"rolling_mean_{window}"] = roll.mean()
        df[f"rolling_std_{window}"] = roll.std()
        df[f"rolling_min_{window}"] = roll.min()
        df[f"rolling_max_{window}"] = roll.max()
        df[f"rolling_sum_{window}"] = roll.sum()

    # --- Weekly means ----------------------------------------------------
    # Non-overlapping weekly blocks going back one month:
    #   rolling_mean_7 = last week, weekly_mean_2 = week before that, etc.
    #
    # pipeline.ipynb instead used shifted.rolling(14).mean(), which makes
    # weekly_mean_2 numerically IDENTICAL to rolling_mean_14 (and
    # weekly_mean_4 identical to rolling_mean_28) — duplicate columns fed
    # to the models and to mutual-information ranking. The 02 definition
    # below is the correct one.
    for i, back in enumerate((7, 14, 21), start=2):
        df[f"weekly_mean_{i}"] = shifted.shift(back).rolling(7).mean()

    # --- Trend -----------------------------------------------------------
    df["trend_7_28"] = df["rolling_mean_7"] - df["rolling_mean_28"]

    # --- Calendar --------------------------------------------------------
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    # --- Holidays --------------------------------------------------------
    df = _add_holiday_features(df, X, Y, holiday_dates)

    return df.dropna().reset_index(drop=True)


def _add_holiday_features(df, X, Y, holiday_dates):
    if not holiday_dates:
        df["is_holiday"] = 0
        df["number_of_holidays_in_target_window"] = 0
        df["holiday_in_target_window"] = 0
        return df

    holidays = {pd.Timestamp(d).date() for d in holiday_dates}
    df["is_holiday"] = df["Date"].dt.date.isin(holidays).astype(int)

    # Counted by calendar date rather than row offset. Equivalent while the
    # daily grid is complete, but robust if a date is ever missing.
    offsets = [pd.Timedelta(days=k) for k in range(Y, Y + X)]
    df["number_of_holidays_in_target_window"] = [
        sum((d + off).date() in holidays for off in offsets)
        for d in df["Date"]
    ]
    df["holiday_in_target_window"] = (
        df["number_of_holidays_in_target_window"] > 0
    ).astype(int)
    return df


def _check_contiguous(df: pd.DataFrame, category: str) -> None:
    """Warn if dates are not consecutive days.

    Both the target and the lag features assume row i+1 is the day after
    row i. If a date is missing, lag_7 silently stops meaning "one week ago"
    and every downstream number is quietly wrong.
    """
    if len(df) < 2:
        return
    gaps = df["Date"].diff().dropna()
    if not (gaps == pd.Timedelta(days=1)).all():
        n_gaps = int((gaps != pd.Timedelta(days=1)).sum())
        print(
            f"WARNING [{category}]: {n_gaps} gap(s) in the daily date index. "
            "Lag and target semantics assume consecutive days."
        )


def available_features(data: pd.DataFrame) -> list[str]:
    """The configured feature list, restricted to columns actually present."""
    return [f for f in config.FEATURES if f in data.columns]
