"""End-to-end forecasting pipeline for one category."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from . import config, features as feat, models as mdl, splits as spl
from .metrics import metrics_row


def select_features(train: pd.DataFrame, features: list[str], target: str,
                    n_features: int = 10) -> list[str]:
    """Rank features by mutual information with the target, keep the top n.

    Computed on the training split only — ranking on the full dataset would
    leak validation and test information into the feature choice.
    """
    scores = mutual_info_regression(
        train[features], train[target], random_state=config.RANDOM_STATE
    )
    return [features[i] for i in np.argsort(scores)[::-1][:n_features]]


def train_all_models(splits: dict, features: list[str], target: str,
                     only: list[str] | None = None) -> tuple:
    """Run every registered model on one set of splits."""
    train, val, test = splits["train"], splits["validation"], splits["test"]
    names = only if only is not None else mdl.model_names()

    rows, predictions = [], {}
    for name in names:
        fit_predict = mdl.MODELS[name]
        try:
            train_pred, val_pred, test_pred = fit_predict(
                train, val, test, features, target
            )
        except Exception as exc:
            # One model failing must not abort a 54-category run.
            print(f"    [{name}] failed: {type(exc).__name__}: {exc}")
            continue

        rows.append(
            metrics_row(
                train[target], train_pred,
                val[target], val_pred,
                test[target], test_pred,
                Model=name,
            )
        )
        predictions[name] = {"validation": val_pred, "test": test_pred}

    return pd.DataFrame(rows), predictions


def optimize_horizons(daily, category, X_range=range(1, 29),
                      Y_range=range(1, 15), Y_fixed=config.DEFAULT_Y):
    """Greedy search: best X at fixed Y, then best Y at that X.

    IMPORTANT CAVEAT for interpretation. WAPE is
    ``sum|error| / sum|actual|``. Widening X grows the denominator roughly
    linearly in X while daily errors partly cancel, so WAPE falls with X
    largely by construction. Comparing WAPE across different X is comparing
    different prediction problems, not different model qualities.

    In practice X should be set by the business replenishment cycle and this
    search used only to report sensitivity. Y is the honest one to optimize:
    it is a genuine difficulty axis at fixed X.
    """
    def _val_wape(X, Y):
        data = feat.build_features(daily, category, X=X, Y=Y)
        if len(data) < 50:
            return np.nan
        s = spl.block_shuffled_split(data)
        target = config.target_name(X, Y)
        from .metrics import evaluate_forecast
        return evaluate_forecast(
            s["validation"][target], s["validation"]["rolling_sum_7"]
        )["wape"]

    x_scores = {X: _val_wape(X, Y_fixed) for X in X_range}
    best_X = min((k for k, v in x_scores.items() if not np.isnan(v)),
                 key=lambda k: x_scores[k], default=config.DEFAULT_X)

    y_scores = {Y: _val_wape(best_X, Y) for Y in Y_range}
    best_Y = min((k for k, v in y_scores.items() if not np.isnan(v)),
                 key=lambda k: y_scores[k], default=config.DEFAULT_Y)

    return best_X, best_Y, pd.DataFrame(
        {"X": list(x_scores), "val_wape": list(x_scores.values())}
    ), pd.DataFrame(
        {"Y": list(y_scores), "val_wape": list(y_scores.values())}
    )


def run_pipeline(daily, category, X=config.DEFAULT_X, Y=config.DEFAULT_Y,
                 n_features=None, verbose=True) -> dict | None:
    """Full run for one category at fixed X and Y."""
    if verbose:
        print(f"\n{'=' * 60}\n{category}  (X={X}, Y={Y})")

    data = feat.build_features(daily, category, X=X, Y=Y)
    if len(data) < 50:
        if verbose:
            print(f"  skipped — only {len(data)} usable rows")
        return None

    target = config.target_name(X, Y)
    features = feat.available_features(data)
    splits = spl.block_shuffled_split(data)

    if n_features is not None:
        features = select_features(splits["train"], features, target,
                                   n_features)

    results, predictions = train_all_models(splits, features, target)
    results.insert(0, "Category", category)
    results["X"] = X
    results["Y"] = Y

    # Selected on validation; test is reported but never selected on.
    best = results.loc[results["Validation WAPE (%)"].idxmin()]

    if verbose:
        print(f"  best: {best['Model']}  "
              f"val WAPE {best['Validation WAPE (%)']:.1f}%  "
              f"test WAPE {best['Test WAPE (%)']:.1f}%")

    return {
        "category": category,
        "X": X,
        "Y": Y,
        "features": features,
        "splits": splits,
        "results": results,
        "predictions": predictions,
        "best_model": best["Model"],
        "leakage_ratio": spl.leakage_ratio(splits, X),
    }
