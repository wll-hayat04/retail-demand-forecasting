import numpy as np, pandas as pd
from src import config, features as feat, splits as spl, models as mdl
from src.metrics import evaluate_forecast

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
cat = "Christmas Decorations"
data = feat.build_features(daily, cat, X=7, Y=7)
target = config.target_name(7, 7)
features = feat.available_features(data)

def stratified_split(data, block_size=14, train=0.70, val=0.15, seed=42):
    d = data.sort_values("Date").reset_index(drop=True).copy()
    nb = int(np.ceil(len(d) / block_size))
    d["block_id"] = np.repeat(np.arange(nb), block_size)[:len(d)]
    # rank blocks by mean target, then deal them round-robin into the
    # three splits so each gets a similar mix of high and low weeks
    order = d.groupby("block_id")[target].mean().sort_values().index.to_numpy()
    rng = np.random.default_rng(seed)
    buckets = {"train": [], "validation": [], "test": []}
    pattern = ["train"]*7 + ["validation"]*2 + ["test"]*2
    for i in range(0, len(order), len(pattern)):
        chunk = order[i:i+len(pattern)]
        pat = pattern[:len(chunk)]
        rng.shuffle(chunk)
        for b, k in zip(chunk, pat):
            buckets[k].append(b)
    return {k: d[d["block_id"].isin(v)].drop(columns="block_id").reset_index(drop=True)
            for k, v in buckets.items()}

for label, fn in [("original", spl.block_shuffled_split), ("stratified", stratified_split)]:
    w = []
    for seed in range(30):
        s = fn(data, seed=seed) if label == "stratified" else fn(data, random_state=seed)
        _, vp, _ = mdl.MODELS["Random Forest"](s["train"], s["validation"], s["test"], features, target)
        w.append(evaluate_forecast(s["validation"][target], vp)["wape"])
    w = np.array(w)
    print(f"{label:11s} mean {w.mean():6.1f}  std {w.std():6.1f}  min {w.min():5.1f}  max {w.max():6.1f}")
