import pandas as pd
from src import config, features as feat, splits as spl, models as mdl
from src.metrics import evaluate_forecast

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
cat = "Christmas Decorations"
data = feat.build_features(daily, cat, X=7, Y=7)
target = config.target_name(7, 7)
features = feat.available_features(data)

rows = []
for seed in range(30):
    s = spl.block_shuffled_split(data, random_state=seed)
    _, vp, _ = mdl.MODELS["Random Forest"](s["train"], s["validation"], s["test"], features, target)
    m = evaluate_forecast(s["validation"][target], vp)
    rows.append({"seed": seed,
                 "val_actual_sum": int(s["validation"][target].sum()),
                 "val_mae": round(m["mae"], 1),
                 "val_wape": round(m["wape"], 1)})

df = pd.DataFrame(rows).sort_values("val_actual_sum")
print(df.to_string(index=False))
print("\nWAPE vs denominator:", round(df["val_actual_sum"].corr(df["val_wape"]), 3))
print("MAE  vs denominator:", round(df["val_actual_sum"].corr(df["val_mae"]), 3))
