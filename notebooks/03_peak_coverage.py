import numpy as np, pandas as pd
from src import config, features as feat, splits as spl, models as mdl
from src.metrics import evaluate_forecast

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
cat = "Christmas Decorations"
data = feat.build_features(daily, cat, X=7, Y=7)
target = config.target_name(7, 7)
features = feat.available_features(data)

peak = data["Date"].dt.month.isin([11, 12])
rows = []
for seed in range(30):
    s = spl.block_shuffled_split(data, random_state=seed)
    tr_peak = s["train"]["Date"].dt.month.isin([11, 12]).mean()
    _, vp, _ = mdl.MODELS["Random Forest"](s["train"], s["validation"], s["test"], features, target)
    rows.append({"seed": seed,
                 "pct_peak_in_train": round(tr_peak * 100, 1),
                 "val_wape": round(evaluate_forecast(s["validation"][target], vp)["wape"], 1)})

df = pd.DataFrame(rows).sort_values("pct_peak_in_train")
print(df.to_string(index=False))
print("\ncorrelation:", round(df["pct_peak_in_train"].corr(df["val_wape"]), 3))
