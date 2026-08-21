# Retail Demand Forecasting

Forecasting units sold per product category over the next **X** days, computed
**Y** days in advance. UCI Online Retail dataset: 54 categories, 374 days
(Dec 2010 – Dec 2011), ~230 usable training rows per category.

The project began as a modelling exercise. Its main results turned out to be
methodological.

---

## Findings

**Per-category model selection is no better than chance.** Choosing the model
with the lowest validation WAPE picks the actual test winner in **13 of 54
categories (24%)**. Random choice among four models gives 25%.

**A single fixed model beats selection.** Applying Random Forest to every
category gives a mean test WAPE of **37.7** against **41.9** for per-category
selection, with **10 catastrophic errors instead of 17** (>10 WAPE points off
the best available model). Selection adds variance without adding information.

**WAPE was measuring the evaluation window, not forecast quality.** Across 30
splits, WAPE correlated **−0.587** with the evaluation set's denominator while
MAE correlated **+0.863** — the two metrics disagreed on identical
predictions. Validation sets containing the Christmas peak have a large
denominator and therefore a small WAPE almost regardless of accuracy.

**Stratifying splits by demand cut the variance 5–12×.** Ordering 14-day
blocks by mean demand before assigning them to train/validation/test reduced
WAPE standard deviation from 42.8 to 8.8 on Christmas Decorations and from
53.5 to 4.6 on Jewellery.

**An implausible R² exposed a data cleaning error.** One category returned a
test R² of −2211. Tracing it back found two orders of 74,215 and 80,995 units,
each cancelled within 30 minutes, whose cancellation rows had been dropped
while the originals were kept. Matching cancellations to their originating
orders brought that category from **440.5% to 15.6%** test WAPE.

**TabFM wins on accuracy and loses on cost.** Google's zero-shot tabular
foundation model wins **15 of 20 paired comparisons** across five categories,
with no tuning or feature engineering. It is also **426× slower than Random
Forest on CPU**, ships 6.6 GB of weights, and its pretrained weights are
non-commercial only.

Overall: median test WAPE **36.5%**, positive test R² in **20 of 54**
categories. Two thirds of categories are not forecast better than their own
mean at a 7-day horizon with 7 days of lead time — a limit of one year of
data, not of the models.

Full write-up: [`RESULTS_REVISED.md`](RESULTS_REVISED.md) and
[`RESULTS_54_AND_TABFM.md`](RESULTS_54_AND_TABFM.md).

---

## Reproducing

```bash
git clone https://github.com/wll-hayat04/retail-demand-forecasting.git
cd retail-demand-forecasting
pip install -e . --no-deps
pip install -r requirements.txt
```

Place `Online Retail.xlsx` and `products_to_categories.json` in `data/raw/`,
then:

```python
from src.data import build_daily_dataset
build_daily_dataset()          # writes data/processed/daily_category_sales_clean.csv
```

Single category, all models:

```python
import pandas as pd
from src import config, pipeline as pipe

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
daily = daily[~daily["Category"].isin(config.EXCLUDED_CATEGORIES)]

res = pipe.run_pipeline(daily, "Tealight Holders & Sets", X=7, Y=7)
res["results"]
```

With uncertainty, which is the point:

```python
from src import validation as val

rep = val.repeated_evaluation(daily, "Tealight Holders & Sets",
                              seeds=range(30), stratified=True)
val.summarise_repeats(rep)
val.is_difference_meaningful(rep, "Random Forest", "7-Day Rolling Sum Baseline")
```

`is_difference_meaningful` compares two models on the *same* splits. If the
confidence interval crosses zero, the models are indistinguishable on this
dataset and no claim should be made either way.

---

## Structure

```
src/
  config.py      paths, holidays, feature list, split parameters
  data.py        Excel -> cleaned daily sales (incl. cancellation matching)
  features.py    target, lags, rolling stats, calendar, holiday features
  splits.py      block-shuffled and stratified block splits, leakage measure
  metrics.py     MAE, RMSE, WAPE, R2, bias
  models.py      model registry; TabFM registered on demand
  pipeline.py    feature selection, training loop, horizon search
  validation.py  repeated evaluation, paired comparison with CIs

notebooks/
  01_data_preparation.ipynb   cleaning and aggregation
  02_modeling.ipynb           single-category analysis
  03_split_strategy.ipynb     comparison of four split strategies
  04_tabfm.ipynb              TabFM evaluation
  pipeline.ipynb              reusable pipeline demonstration

results/                      all CSVs backing the numbers above
```

Adding a model is one function plus a decorator in `src/models.py`; every
notebook that imports the registry picks it up.

---

## Method notes

**Splitting.** Four strategies were compared (random, chronological,
TimeSeriesCV, block-shuffled). Block-shuffled was selected as the only one
giving positive R² on most categories, with ~50% residual leakage documented
and measured. Stratified block splitting is the improvement described above.

**Leakage.** Two rows share leakage when their target windows overlap, i.e.
when they are fewer than X days apart. `splits.leakage_ratio` measures this
rather than asserting it.

**Metrics.** WAPE is scale-independent and comparable across categories, which
is why it was chosen — but it is not comparable across different values of X,
since the denominator grows with the window. Conclusions rest on paired
comparisons over repeated splits, not on absolute WAPE levels.

**Limitation.** One year of data contains exactly one Christmas. No split
strategy or model creates seasonal signal that is not there, and this is the
binding constraint on every result.

---

*Internship project, Miningful — Hayat Waldi, EMI (Université Mohammed V de
Rabat), June–August 2026. Supervisor: Nevio Dubbini.*
