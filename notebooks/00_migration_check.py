import pandas as pd
from src import config, features as feat, splits as spl, pipeline as pipe

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
daily = daily[~daily["Category"].isin(config.EXCLUDED_CATEGORIES)]

print("shape:", daily.shape)
print("categories:", daily["Category"].nunique())
print("range:", daily["Date"].min(), "->", daily["Date"].max())

res = pipe.run_pipeline(daily, "Tealight Holders & Sets", X=7, Y=7)
print(res["results"][["Model", "Validation WAPE (%)", "Test WAPE (%)", "Test R2"]].to_string(index=False))
print("leakage ratio:", round(res["leakage_ratio"], 3))
