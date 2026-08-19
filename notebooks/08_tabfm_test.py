import time, pandas as pd
from src import config, features as feat, splits as spl, models as mdl
from src.metrics import evaluate_forecast

print("loading TabFM...")
if not mdl.enable_tabfm():
    raise SystemExit("TabFM not available")
print("models:", mdl.model_names())

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
cat = "Tealight Holders & Sets"
data = feat.build_features(daily, cat, X=7, Y=7)
target = config.target_name(7, 7)
features = feat.available_features(data)
s = spl.stratified_block_split(data, target, random_state=42)
print("train rows:", len(s["train"]), "| features:", len(features))

t0 = time.time()
_, vp, tp = mdl.MODELS["TabFM"](s["train"], s["validation"], s["test"], features, target)
print(f"elapsed: {time.time()-t0:.1f}s")

for name, pred, df in [("val", vp, s["validation"]), ("test", tp, s["test"])]:
    m = evaluate_forecast(df[target], pred)
    print(f"{name}: WAPE {m['wape']:.2f}%  R2 {m['r2']:.3f}  MAE {m['mae']:.1f}")
