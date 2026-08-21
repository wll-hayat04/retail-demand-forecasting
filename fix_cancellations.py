import pandas as pd

df = pd.read_excel("data/raw/Online Retail.xlsx")
print("raw rows:", len(df))

cancels = df[df["InvoiceNo"].astype(str).str.startswith("C")]
print("cancellation rows:", len(cancels))

# Match each cancellation to its originating order: same customer, same
# product, opposite quantity, and the order must precede the cancellation.
drop_idx = set(cancels.index)
matched = 0
for _, c in cancels.iterrows():
    cand = df[(df["CustomerID"] == c["CustomerID"]) &
              (df["StockCode"] == c["StockCode"]) &
              (df["Quantity"] == -c["Quantity"]) &
              (df["InvoiceDate"] <= c["InvoiceDate"]) &
              (~df.index.isin(drop_idx))]
    if len(cand):
        drop_idx.add(cand.index[-1])
        matched += 1

print(f"cancellations matched to an original order: {matched}/{len(cancels)}")
clean = df.drop(index=list(drop_idx))
print("rows after removing both sides:", len(clean))
print("max quantity after:", clean["Quantity"].max())
