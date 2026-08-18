"""Repeated-split evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features as feat, models as mdl, splits as spl
from .metrics import evaluate_forecast


def repeated_evaluation(daily, category, X=config.DEFAULT_X, Y=config.DEFAULT_Y,
                        seeds=range(30), split_name="validation"):
    """Run every registered model across many block-shuffled draws."""
    data = feat.build_features(daily, category, X=X, Y=Y)
    if len(data) < 50:
        return pd.DataFrame()

    target = config.target_name(X, Y)
    features = feat.available_features(data)

    rows = []
    for seed in seeds:
        s = spl.block_shuffled_split(data, random_state=seed)
        train, held = s["train"], s[split_name]

        for name, fit_predict in mdl.MODELS.items():
            try:
                _, val_pred, test_pred = fit_predict(
                    train, s["validation"], s["test"], features, target
                )
            except Exception:
                continue
            pred = val_pred if split_name == "validation" else test_pred
            m = evaluate_forecast(held[target], pred)
            rows.append({"Category": category, "Model": name, "seed": seed,
                         "wape": m["wape"], "r2": m["r2"]})

    return pd.DataFrame(rows)


def summarise_repeats(repeats):
    """Mean, spread, and win-rate per model across seeds."""
    if repeats.empty:
        return repeats

    winners = repeats.loc[repeats.groupby(["Category", "seed"])["wape"].idxmin()]
    n_seeds = repeats["seed"].nunique()
    win_rate = (winners.groupby(["Category", "Model"]).size()
                .div(n_seeds).rename("win_rate"))

    out = (repeats.groupby(["Category", "Model"])
           .agg(wape_mean=("wape", "mean"),
                wape_std=("wape", "std"),
                wape_min=("wape", "min"),
                wape_max=("wape", "max"),
                r2_mean=("r2", "mean"))
           .join(win_rate)
           .fillna({"win_rate": 0.0})
           .reset_index()
           .sort_values(["Category", "wape_mean"]))
    return out.round(3)


def is_difference_meaningful(repeats, model_a, model_b):
    """Compare two models on the SAME draws (paired, not independent)."""
    a = repeats[repeats["Model"] == model_a].set_index("seed")["wape"]
    b = repeats[repeats["Model"] == model_b].set_index("seed")["wape"]
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).to_numpy()

    if len(diff) < 2:
        return {"n": len(diff)}

    mean, sd = float(np.mean(diff)), float(np.std(diff, ddof=1))
    se = sd / np.sqrt(len(diff))
    return {"n": len(diff),
            "mean_diff": round(mean, 3),
            "std_diff": round(sd, 3),
            "ci_low": round(mean - 1.96 * se, 3),
            "ci_high": round(mean + 1.96 * se, 3),
            "a_better_pct": round(float(np.mean(diff < 0)) * 100, 1)}
