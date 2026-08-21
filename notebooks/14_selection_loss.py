import pandas as pd
from src import config

df = pd.read_csv(config.RESULTS / "pipeline_results_all_categories.csv")

clean = df[df["test_R2"] > -5]
excluded = df[df["test_R2"] <= -5]
print(f"degenerate (test_R2 < -5): {len(excluded)}")
for c in excluded["Category"]:
    print("   ", c)

print(f"\nselection accuracy: {clean['selection_correct'].sum()}/{len(clean)} "
      f"({clean['selection_correct'].mean()*100:.0f}%)")
print(f"median test WAPE:   {clean['test_WAPE'].median():.1f}%")
print(f"positive test R2:   {(clean['test_R2']>0).sum()}/{len(clean)}")
print(f"median val->test gap: {clean['gap'].median():.1f} pts")
