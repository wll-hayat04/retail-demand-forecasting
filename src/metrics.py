"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_forecast(y_true, y_pred) -> dict:
    """Evaluate a forecast.

    Returns a dict rather than a tuple: at the call site
    ``res["wape"]`` is unambiguous where ``res[2]`` is not, and adding a
    metric later does not silently shift positions in existing code.

    Negative predictions are clipped to zero — a category cannot sell a
    negative quantity, so this is a free improvement for any model that
    can produce them (Linear Regression, most often).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0.0)

    denominator = np.sum(np.abs(y_true))

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        # WAPE: scale-independent, so comparable across categories of very
        # different volume. Undefined when the actuals sum to zero.
        "wape": (
            float(np.sum(np.abs(y_true - y_pred)) / denominator * 100)
            if denominator > 0
            else np.nan
        ),
        "r2": r2_score(y_true, y_pred),
        # Positive = over-forecasting on average (overstock risk),
        # negative = under-forecasting (stockout risk).
        "bias": float(np.mean(y_pred - y_true)),
    }


def metrics_row(y_train, train_pred, y_val, val_pred, y_test, test_pred,
                **extra) -> dict:
    """Flatten train/val/test metrics into one row for a results table."""
    row = dict(extra)
    for split, (y_true, y_pred) in {
        "Train": (y_train, train_pred),
        "Validation": (y_val, val_pred),
        "Test": (y_test, test_pred),
    }.items():
        m = evaluate_forecast(y_true, y_pred)
        row[f"{split} WAPE (%)"] = round(m["wape"], 3)
        row[f"{split} R2"] = round(m["r2"], 3)
        if split == "Test":
            row["Test MAE"] = round(m["mae"], 3)
            row["Test RMSE"] = round(m["rmse"], 3)
            row["Test Bias"] = round(m["bias"], 3)
    return row


def summarise(results: list[dict]) -> pd.DataFrame:
    """Build a results DataFrame from accumulated metric rows."""
    return pd.DataFrame(results)
