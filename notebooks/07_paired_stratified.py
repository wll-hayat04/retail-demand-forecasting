import pandas as pd
from src import config, validation as val

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
daily = daily[~daily["Category"].isin(config.EXCLUDED_CATEGORIES)]

frames = []
for cat in config.SELECTED_CATEGORIES:
    print("running", cat)
    frames.append(val.repeated_evaluation(daily, cat, X=7, Y=7, seeds=range(30),
                                          stratified=True))

repeats = pd.concat(frames, ignore_index=True)
repeats.to_csv(config.RESULTS / "repeated_stratified_raw.csv", index=False)

summary = val.summarise_repeats(repeats)
summary.to_csv(config.RESULTS / "repeated_stratified_summary.csv", index=False)
print(summary.to_string(index=False))

print()
for cat in config.SELECTED_CATEGORIES:
    sub = repeats[repeats["Category"] == cat]
    print(cat, "| RF vs Baseline:",
          val.is_difference_meaningful(sub, "Random Forest", "7-Day Rolling Sum Baseline"))
