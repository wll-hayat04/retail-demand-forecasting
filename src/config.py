"""Project-wide constants and paths.

Everything that was previously duplicated across notebooks lives here.
Import from this module — never redefine these inline in a notebook.
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths — derived from the repo root, so the project runs on any machine.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

RAW_EXCEL = DATA_RAW / "Online Retail.xlsx"
CATEGORY_MAP = DATA_RAW / "products_to_categories.json"
DAILY_CLEAN = DATA_PROCESSED / "daily_category_sales_clean.csv"

for _d in (DATA_RAW, DATA_PROCESSED, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Forecasting horizons
#   X = length of the forecast window (days summed)
#   Y = lead time (days ahead the forecast is made)
# ---------------------------------------------------------------------------

DEFAULT_X = 7
DEFAULT_Y = 7


def target_name(X: int = DEFAULT_X, Y: int = DEFAULT_Y) -> str:
    """Canonical target column name. Use this instead of f-strings inline."""
    return f"target_X{X}_Y{Y}"


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

# Catch-all bucket, not a real category — excluded from all modelling.
EXCLUDED_CATEGORIES = ["Miscellaneous"]

SELECTED_CATEGORIES = [
    "Tealight Holders & Sets",
    "Cake Cases & Baking Accessories",
    "Jumbo Bags & Shoppers",
    "Christmas Decorations",
    "Jewellery - Earrings",
]


# ---------------------------------------------------------------------------
# UK (England & Wales) bank holidays covering the dataset period.
# Substitute days are included because the shop closes on those too.
#
# NOTE: 2011-04-29 (Royal Wedding) was present in 02 but MISSING from
# pipeline.ipynb. It is a genuine public holiday inside the data window,
# so it is kept here. This is now the single source of truth.
# ---------------------------------------------------------------------------

HOLIDAY_DATES = frozenset(
    pd.to_datetime(
        [
            # 2010
            "2010-12-25",  # Christmas Day
            "2010-12-26",  # Boxing Day
            "2010-12-27",  # Christmas Day (substitute)
            "2010-12-28",  # Boxing Day (substitute)
            # 2011
            "2011-01-01",  # New Year's Day
            "2011-01-03",  # New Year's Day (substitute)
            "2011-04-22",  # Good Friday
            "2011-04-25",  # Easter Monday
            "2011-04-29",  # Royal Wedding
            "2011-05-02",  # Early May bank holiday
            "2011-05-30",  # Spring bank holiday
            "2011-08-29",  # Summer bank holiday
            "2011-12-25",  # Christmas Day
            "2011-12-26",  # Boxing Day
            "2011-12-27",  # Christmas Day (substitute)
        ]
    ).date
)


# ---------------------------------------------------------------------------
# Feature list used by all models.
#
# rolling_sum_7 is deliberately NOT a model feature — it is the naive
# baseline's prediction. Including it would hand the models the baseline's
# answer directly.
# ---------------------------------------------------------------------------

LAG_FEATURES = [f"lag_{i}" for i in range(1, 8)] + ["lag_14", "lag_28"]

ROLLING_FEATURES = [
    f"rolling_{stat}_{w}"
    for stat in ("mean", "std", "min", "max")
    for w in (7, 14, 28)
]

WEEKLY_FEATURES = ["weekly_mean_2", "weekly_mean_3", "weekly_mean_4"]

TREND_FEATURES = ["trend_7_28"]

CALENDAR_FEATURES = ["day_of_week", "month"]

HOLIDAY_FEATURES = [
    "is_holiday",
    "number_of_holidays_in_target_window",
    "holiday_in_target_window",
]

FEATURES = (
    LAG_FEATURES
    + ROLLING_FEATURES
    + WEEKLY_FEATURES
    + TREND_FEATURES
    + CALENDAR_FEATURES
    + HOLIDAY_FEATURES
)

# Treated as categorical rather than ordered numeric by models that
# distinguish the two (day_of_week=6 is not "greater than" day_of_week=1).
CATEGORICAL_FEATURES = ["day_of_week", "month"]


# ---------------------------------------------------------------------------
# Split configuration
# ---------------------------------------------------------------------------

BLOCK_SIZE = 14
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
RANDOM_STATE = 42
