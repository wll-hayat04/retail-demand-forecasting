import time, pandas as pd
from src import config, features as feat, splits as spl, models as mdl
from src.metrics import evaluate_forecast

N_EST = 4
print(f"loading TabFM (n_estimators={N_EST})...")
if not mdl.enable_tabfm(n_estimators=N_EST):
    raise SystemExit("TabFM not available")

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
cat = "Tealight Holders & Sets"
data = feat.build_features(daily, cat, X=7, Y=7)
target = config.target_name(7, 7)
features = feat.available_features(data)
s = spl.stratified_block_split(data, target, random_state=42)
print("train rows:", len(s["train"]))

t0 = time.time()
_, vp, tp = mdl.MODELS["TabFM"](s["train"], s["validation"], s["test"], features, target)
el = time.time() - t0
print(f"elapsed: {el:.1f}s  -> 30 seeds would take {el*30/60:.1f} min")

for name, pred, df in [("val", vp, s["validation"]), ("test", tp, s["test"])]:
    m = evaluate_forecast(df[target], pred)
    print(f"{name}: WAPE {m['wape']:.2f}%  R2 {m['r2']:.3f}")
