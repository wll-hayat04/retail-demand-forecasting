import pandas as pd
from src import config

df = pd.read_csv(config.RESULTS / "pipeline_results_all_categories.csv")
print(f"selection accuracy: {df['selection_correct'].sum()}/{len(df)} "
      f"({df['selection_correct'].mean()*100:.0f}%)")
print(f"median test WAPE: {df['test_WAPE'].median():.1f}%")
print(f"positive test R2: {(df['test_R2']>0).sum()}/{len(df)}")
print(f"\nval->test gap: median {df['gap'].median():.1f} pts, "
      f"min {df['gap'].min():.1f}, max {df['gap'].max():.1f}")
print("\nworst 5 by test WAPE:")
print(df.nlargest(5, "test_WAPE")[["Category","selected_model","test_WAPE","test_R2"]].to_string(index=False))
print("\nbest 5:")
print(df.nsmallest(5, "test_WAPE")[["Category","selected_model","test_WAPE","test_R2"]].to_string(index=False))
