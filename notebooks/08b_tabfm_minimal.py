import os, time
import torch
torch.set_num_threads(os.cpu_count())
print("threads:", torch.get_num_threads())

import pandas as pd
from tabfm import TabFMRegressor
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1
from src import config, features as feat, splits as spl
from src.metrics import evaluate_forecast

print("loading float32...")
t0 = time.time()
model = tabfm_v1.load(model_type="regression", dtype=None)
print(f"loaded in {time.time()-t0:.1f}s")

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
data = feat.build_features(daily, "Tealight Holders & Sets", X=7, Y=7)
target = config.target_name(7, 7)
features = feat.available_features(data)
s = spl.stratified_block_split(data, target, random_state=42)

def prep(df):
    Xd = df[features].copy()
    for c in ["day_of_week", "month"]:
        Xd[c] = Xd[c].astype(str)
    return Xd

reg = TabFMRegressor(model=model, n_estimators=1, random_state=42)
t0 = time.time()
reg.fit(prep(s["train"]), s["train"][target])
vp = reg.predict(prep(s["validation"]))
el = time.time() - t0

print(f"n_estimators=1, validation only: {el:.1f}s")
m = evaluate_forecast(s["validation"][target], vp)
print(f"WAPE {m['wape']:.2f}%   R2 {m['r2']:.3f}")
print(f"-> at n_estimators=16, 30 seeds: ~{el*16*30/60:.0f} min")
