import time, pandas as pd
from src import config, models as mdl, validation as val

N_EST, N_SEEDS = 8, 5
CATS = ["Tealight Holders & Sets", "Jewellery - Earrings"]

print(f"n_estimators={N_EST}, seeds={N_SEEDS}, categories={len(CATS)}")
if not mdl.enable_tabfm(n_estimators=N_EST):
    raise SystemExit("TabFM unavailable")

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
daily = daily[~daily["Category"].isin(config.EXCLUDED_CATEGORIES)]

frames = []
for cat in CATS:
    t0 = time.time()
    frames.append(val.repeated_evaluation(daily, cat, X=7, Y=7,
                                          seeds=range(N_SEEDS), stratified=True))
    print(f"{cat}: {time.time()-t0:.0f}s")

rep = pd.concat(frames, ignore_index=True)
rep.to_csv(config.RESULTS / "tabfm_comparison_raw.csv", index=False)

summary = val.summarise_repeats(rep)
summary.to_csv(config.RESULTS / "tabfm_comparison_summary.csv", index=False)
print()
print(summary.to_string(index=False))

print()
for cat in CATS:
    sub = rep[rep["Category"] == cat]
    for other in ["Random Forest", "XGBoost", "7-Day Rolling Sum Baseline"]:
        d = val.is_difference_meaningful(sub, "TabFM", other)
        if d.get("n", 0) < 2:
            continue
        decided = (d["ci_low"] < 0) == (d["ci_high"] < 0)
        verdict = ("TabFM better" if decided and d["mean_diff"] < 0
                   else "TabFM worse" if decided else "within noise")
        print(f"{cat[:22]:22s} vs {other[:24]:24s} {d['mean_diff']:+7.2f} "
              f"CI[{d['ci_low']:+6.2f},{d['ci_high']:+6.2f}] {verdict}")
