import pandas as pd
from src import config, features as feat, splits as spl, models as mdl
from src.metrics import evaluate_forecast

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
daily = daily[~daily["Category"].isin(config.EXCLUDED_CATEGORIES)]
cats = sorted(daily["Category"].unique())

rows = []
for cat in cats:
    data = feat.build_features(daily, cat, X=7, Y=7)
    if len(data) < 50:
        continue
    target = config.target_name(7, 7)
    features = feat.available_features(data)
    s = spl.stratified_block_split(data, target, random_state=42)
    rec = {"Category": cat}
    for name in mdl.model_names():
        _, vp, tp = mdl.MODELS[name](s["train"], s["validation"], s["test"],
                                     features, target)
        rec[f"val_{name}"] = evaluate_forecast(s["validation"][target], vp)["wape"]
        rec[f"test_{name}"] = evaluate_forecast(s["test"][target], tp)["wape"]
        rec[f"r2_{name}"] = evaluate_forecast(s["test"][target], tp)["r2"]
    rows.append(rec)

df = pd.DataFrame(rows)
names = mdl.model_names()

# exclude degenerate categories, same rule as before
best_r2 = df[[f"r2_{n}" for n in names]].max(axis=1)
df = df[best_r2 > -5].reset_index(drop=True)
print(f"n = {len(df)} categories\n")

test_cols = [f"test_{n}" for n in names]
val_cols  = [f"val_{n}" for n in names]

oracle = df[test_cols].min(axis=1)
chosen = df.apply(lambda r: r[f"test_{names[[r[c] for c in val_cols].index(min(r[c] for c in val_cols))]}"], axis=1)

print("strategy                      median test WAPE   mean   >10pts worse than oracle")
print(f"{'select per category (val)':30s} {chosen.median():8.1f} {chosen.mean():10.1f} "
      f"{((chosen-oracle)>10).sum():10d}/{len(df)}")
for n in names:
    col = df[f"test_{n}"]
    print(f"{'always ' + n:30s} {col.median():8.1f} {col.mean():10.1f} "
          f"{((col-oracle)>10).sum():10d}/{len(df)}")
print(f"{'oracle (best on test)':30s} {oracle.median():8.1f} {oracle.mean():10.1f}")

df.to_csv(config.RESULTS / "strategy_comparison.csv", index=False)
