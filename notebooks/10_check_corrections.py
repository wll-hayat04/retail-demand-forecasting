import pandas as pd
from src import config, validation as val

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
daily = daily[~daily["Category"].isin(config.EXCLUDED_CATEGORIES)]

checks = [
    ("Jewellery - Earrings", 28, 7, "Random Forest", "XGBoost"),
    ("Tealight Holders & Sets", 7, 3, "XGBoost", "7-Day Rolling Sum Baseline"),
]

for cat, X, Y, a, b in checks:
    rep = val.repeated_evaluation(daily, cat, X=X, Y=Y, seeds=range(30), stratified=True)
    d = val.is_difference_meaningful(rep, a, b)
    verdict = "DISTINGUABLE" if (d["ci_low"] < 0) == (d["ci_high"] < 0) else "indistinguable"
    print(f"{cat} (X={X},Y={Y}) | {a} vs {b}: {d} -> {verdict}")
