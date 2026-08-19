import numpy as np, pandas as pd
from src import config, features as feat, splits as spl, models as mdl
from src.metrics import evaluate_forecast

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
daily = daily[~daily["Category"].isin(config.EXCLUDED_CATEGORIES)]

rows = []
for cat in config.SELECTED_CATEGORIES:
    data = feat.build_features(daily, cat, X=7, Y=7)
    target = config.target_name(7, 7)
    features = feat.available_features(data)

    for label in ["original", "stratified"]:
        w, dens = [], []
        for seed in range(30):
            s = (spl.block_shuffled_split(data, random_state=seed) if label == "original"
                 else spl.stratified_block_split(data, target, random_state=seed))
            _, vp, _ = mdl.MODELS["Random Forest"](s["train"], s["validation"], s["test"], features, target)
            w.append(evaluate_forecast(s["validation"][target], vp)["wape"])
            dens.append(s["validation"][target].sum())
        w, dens = np.array(w), np.array(dens)
        rows.append({"Category": cat, "split": label,
                     "wape_mean": round(w.mean(), 1), "wape_std": round(w.std(), 1),
                     "wape_max": round(w.max(), 1),
                     "denom_cv": round(dens.std() / dens.mean(), 3)})

df = pd.DataFrame(rows)
df.to_csv(config.RESULTS / "split_strategy_comparison.csv", index=False)
print(df.to_string(index=False))
