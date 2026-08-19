import numpy as np, pandas as pd
from src import config, validation as val

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
daily = daily[~daily["Category"].isin(config.EXCLUDED_CATEGORIES)]

# X et Y du tableau original de project_context.md
ORIGINAL = {
    "Cake Cases & Baking Accessories": (24, 14, "Random Forest", 6.9),
    "Christmas Decorations":           (7,  6,  "7-Day Rolling Sum Baseline", 19.6),
    "Jewellery - Earrings":            (28, 7,  "XGBoost", 9.7),
    "Jumbo Bags & Shoppers":           (16, 8,  "Random Forest", 11.3),
    "Tealight Holders & Sets":         (7,  3,  "7-Day Rolling Sum Baseline", 12.5),
}

rows = []
for cat, (X, Y, claimed_model, claimed_wape) in ORIGINAL.items():
    print(f"running {cat} (X={X}, Y={Y})")
    rep = val.repeated_evaluation(daily, cat, X=X, Y=Y, seeds=range(30), stratified=True)
    if rep.empty:
        continue
    s = val.summarise_repeats(rep)
    best = s.iloc[0]
    claimed = s[s["Model"] == claimed_model]
    rows.append({
        "Category": cat, "X": X, "Y": Y,
        "claimed_model": claimed_model,
        "claimed_wape": claimed_wape,
        "measured_wape_mean": round(float(claimed["wape_mean"].iloc[0]), 1) if len(claimed) else np.nan,
        "measured_wape_std": round(float(claimed["wape_std"].iloc[0]), 1) if len(claimed) else np.nan,
        "claimed_model_winrate": round(float(claimed["win_rate"].iloc[0]), 2) if len(claimed) else np.nan,
        "actual_best_model": best["Model"],
        "actual_best_wape": round(float(best["wape_mean"]), 1),
    })

out = pd.DataFrame(rows)
out.to_csv(config.RESULTS / "final_table_revised.csv", index=False)
print()
print(out.to_string(index=False))
