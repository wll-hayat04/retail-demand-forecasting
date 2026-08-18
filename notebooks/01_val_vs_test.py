import pandas as pd
from src import config, pipeline as pipe

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
daily = daily[~daily["Category"].isin(config.EXCLUDED_CATEGORIES)]

rows = []
for cat in config.SELECTED_CATEGORIES:
    res = pipe.run_pipeline(daily, cat, X=7, Y=7, verbose=False)
    r = res["results"]
    r["Gap (test-val)"] = r["Test WAPE (%)"] - r["Validation WAPE (%)"]
    rows.append(r)

out = pd.concat(rows, ignore_index=True)
out.to_csv(config.RESULTS / "step9_val_vs_test.csv", index=False)
print(out[["Category","Model","Validation WAPE (%)","Test WAPE (%)","Gap (test-val)","Test R2"]].to_string(index=False))
