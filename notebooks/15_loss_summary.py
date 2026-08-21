import pandas as pd
from src import config

df = pd.read_csv(config.RESULTS / "pipeline_results_all_categories.csv")
clean = df[df["test_R2"] > -5]

loss = clean["selection_loss"]
print(f"n = {len(clean)} categories (4 degenerate excluded)")
print(f"selection accuracy: {clean['selection_correct'].sum()}/{len(clean)}")
print()
print(f"selection loss (test WAPE points lost by choosing on validation):")
print(f"  median   {loss.median():6.2f}")
print(f"  mean     {loss.mean():6.2f}")
print(f"  75th pct {loss.quantile(0.75):6.2f}")
print(f"  90th pct {loss.quantile(0.90):6.2f}")
print(f"  max      {loss.max():6.2f}")
print()
print(f"loss < 2 pts:  {(loss < 2).sum()}/{len(clean)}")
print(f"loss > 10 pts: {(loss > 10).sum()}/{len(clean)}")
print()
print("worst 5:")
print(clean.nlargest(5, "selection_loss")[
    ["Category","selected_model","test_WAPE","best_on_test","best_test_WAPE","selection_loss"]
].to_string(index=False))
