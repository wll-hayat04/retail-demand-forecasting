import time
import pandas as pd
from src import config, pipeline as pipe

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
daily = daily[~daily["Category"].isin(config.EXCLUDED_CATEGORIES)]
cats = sorted(daily["Category"].unique())
print(f"{len(cats)} categories")

rows, t_start = [], time.time()
for i, cat in enumerate(cats, 1):
    res = pipe.run_pipeline(daily, cat, X=7, Y=7, verbose=False, stratified=True)
    if res is None:
        print(f"[{i:2d}/{len(cats)}] {cat[:40]:40s} SKIPPED")
        continue
    r = res["results"]
    best_val = r.loc[r["Validation WAPE (%)"].idxmin()]
    best_test = r.loc[r["Test WAPE (%)"].idxmin()]
    rows.append({
        "Category": cat,
        "n_rows": len(res["splits"]["train"]) + len(res["splits"]["validation"])
                  + len(res["splits"]["test"]),
        "selected_model": best_val["Model"],
        "val_WAPE": best_val["Validation WAPE (%)"],
        "test_WAPE": best_val["Test WAPE (%)"],
        "test_R2": best_val["Test R2"],
        "gap": round(best_val["Test WAPE (%)"] - best_val["Validation WAPE (%)"], 2),
        "best_on_test": best_test["Model"],
        "best_test_WAPE": best_test["Test WAPE (%)"],
        "selection_loss": round(best_val["Test WAPE (%)"]
                                - best_test["Test WAPE (%)"], 2),
        "selection_correct": best_val["Model"] == best_test["Model"],
        "leakage": round(res["leakage_ratio"], 3),
    })
    print(f"[{i:2d}/{len(cats)}] {cat[:40]:40s} {best_val['Model'][:18]:18s} "
          f"test {best_val['Test WAPE (%)']:6.1f}%")

out = pd.DataFrame(rows)
out.to_csv(config.RESULTS / "pipeline_results_all_categories.csv", index=False)

print(f"\nelapsed: {time.time()-t_start:.0f}s | {len(out)} categories")
print(f"\nmedian test WAPE: {out['test_WAPE'].median():.1f}%")
print(f"positive test R2: {(out['test_R2'] > 0).sum()}/{len(out)}")
print(f"selection picked the test winner: {out['selection_correct'].sum()}/{len(out)}")
print("\nmodels selected on validation:")
print(out["selected_model"].value_counts().to_string())
print("\nmodels that actually won on test:")
print(out["best_on_test"].value_counts().to_string())
