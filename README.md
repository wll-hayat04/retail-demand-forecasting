# Retail Demand Forecasting

Forecasting units sold per product category over the next **X** days, computed **Y** days in advance. The project uses the UCI Online Retail dataset: 54 categories, 374 days (December 2010–December 2011), and approximately 230 usable training rows per category in the reference experiment.

The project began as a modelling exercise and evolved into an investigation of **evaluation reliability, horizon choice, forecast errors, feature selection, and predictive uncertainty**.

---

## Main findings

**Per-category model selection was unreliable in the reference experiment.** Selecting the model with the lowest validation WAPE identified the model with the lowest test WAPE in **13 of 54 categories (24%)**. For comparison, uniform random choice among four models has a 25% selection probability. This descriptive comparison does not establish that model selection performs at chance level in general.

**A fixed Random Forest performed better than validation-based selection in that experiment.** Applying Random Forest to every category yielded mean test WAPE of **37.7%**, compared with **41.9%** for per-category selection. It also produced **10**, rather than **17**, instances where the selected model's WAPE exceeded the lowest observed test WAPE by more than 10 percentage points. These figures describe the evaluated data and splits, not a guarantee about future performance.

**WAPE is sensitive to the evaluation-set denominator.** Across 30 splits, WAPE correlated **−0.587** with the evaluation set's demand denominator, whereas MAE correlated **+0.863** with it. Changes in the amount and timing of demand included in an evaluation set can therefore affect WAPE rankings. WAPE should be interpreted alongside MAE and the amount of actual demand, especially across different forecast-window lengths.

**Demand-stratified splitting reduced measured variability, but did not eliminate leakage.** Ordering 14-day blocks by mean demand before assigning them to train, validation, and test reduced the standard deviation of WAPE from **42.8 to 8.8** for Christmas Decorations and from **53.5 to 4.6** for Jewellery – Earrings. Because the forecasting targets overlap in time, these splits are not equivalent to independent prospective evaluation.

**An extreme R² helped identify a cleaning error.** One category initially had test R² of **−2211**. Investigation found two orders of **74,215** and **80,995** units, each cancelled within 30 minutes: cancellation rows had been dropped while the original purchases remained. Matching cancellations to their originating orders changed that category's measured test WAPE from **440.5% to 15.6%**.

**TabFM improved accuracy in the tested comparisons at substantial computational cost.** Google's zero-shot tabular foundation model had lower error in **15 of 20 paired comparisons** across five categories, without task-specific tuning. In this setup, it was **426× slower than Random Forest on CPU**, required approximately **6.6 GB** of weights, and its pretrained weights were restricted to non-commercial use.

**Performance remains category-dependent.** In the reference **X = 7, Y = 7** experiment, median test WAPE was **36.5%**, and test R² was positive in **20 of 54 categories**. For the remaining 34 categories, test R² was non-positive under this evaluation protocol. The dataset contains only one annual seasonal cycle, which limits the information available for learning rare seasonal events.

---

## Further analyses

### Forecasting windows and lead times

The target at reference date *t* is the **total quantity sold from day t + Y through day t + Y + X − 1**. A joint analysis considered **X ∈ {7, 14, 28}** and **Y ∈ {1, 7, 14}** across five representative categories, using **10 repeated splits** per configuration and common reference dates within each category.

Increasing the lead time often increased forecasting error at a fixed window length, but the pattern depended on the category and X. For example, at **X = 14**, Jewellery – Earrings validation WAPE increased from **60.89% (Y = 1)** to **130.49% (Y = 14)**, whereas Cake Cases & Baking Accessories remained comparatively stable across lead times at **X = 28** (**11.14–11.66%**).

Longer windows often showed lower WAPE, partly because X increases the quantity accumulated in the WAPE denominator. **Different X values define different forecasting tasks**: a lower WAPE at X = 28 is not, on its own, evidence of a better forecast than at X = 7. The operational decision should determine X and Y before comparing models for that target.

### Feature selection

For Jewellery – Earrings and Christmas Decorations, an experiment compared **30 features** with the **top 15** and **top 10** ranked by mutual information **using training data only** in each of 10 repeated splits. The prediction target was held fixed at **X = 7, Y = 7**.

With 10 features, mean validation WAPE changed from **75.23% to 73.61%** for Jewellery – Earrings and from **33.82% to 33.36%** for Christmas Decorations. However, across-split WAPE standard deviations increased from **6.60 to 11.81** and from **10.18 to 12.93**, respectively. A smaller feature set may simplify the model while maintaining comparable average accuracy, but **did not provide a consistent improvement in stability**. The 30-feature configuration remains the reference setup.

### Error analysis and probabilistic forecasts

Detailed error analysis of Jewellery – Earrings highlighted systematic underprediction during abrupt increases in future demand. Added short- versus long-term demand-regime features improved one test split, but **did not consistently improve performance across 30 paired splits**. An exploratory attempt to detect future demand jumps from the features available at prediction time was also unstable; the observed jump rows represented only a small number of distinct episodes because target windows overlap.

Probabilistic forecasting estimates **conditional quantiles of the same future cumulative-demand target**, not quantiles of historical sales or residuals. At **X = 7, Y = 7**, the Q10 and Q90 predictions define a nominal **80% prediction interval**. For Jewellery – Earrings, raw empirical test coverage was **26.19%**; validation-based conformal adjustment increased it to **52.38%**, still below the nominal target. None of the seven high-demand test observations fell inside the calibrated intervals.

Across the five categories, calibration increased empirical coverage, but results varied substantially. Some categories attained high coverage with wide intervals, while abrupt demand spikes remained poorly covered in others. **An interval is useful only when its observed coverage and width are reported together.** The conformal results are exploratory: temporal dependence, overlapping target windows, and the evaluation protocol limit the applicability of standard exchangeability-based coverage guarantees.

### Comparison of prediction interval strategies

Three approaches were compared for estimating uncertainty around future
cumulative demand:

- Absolute fixed margins around the point forecast.
- Percentage-based fixed margins around the point forecast.
- Conformally calibrated Q10–Q90 prediction intervals.

For the fixed-margin approaches, interval sizes were selected on validation
data to target at least 80% empirical coverage, then evaluated unchanged
on the test set.

The experiments revealed a category-dependent trade-off between interval
coverage and width.

For Christmas Decorations, both the percentage-based and conformal
intervals achieved 97.62% test coverage. However, their mean interval
widths represented 162.16% and 68.87% of mean observed test demand,
respectively.

The percentage-based interval used a ±75% margin around each point forecast.
The value 162.16% represents the resulting mean interval width relative
to mean observed demand, not the percentage margin applied to the forecast.

For Jewellery - Earrings, none of the tested approaches achieved 80%
test coverage, and none covered the high-demand observations.

These findings demonstrate that prediction intervals should be evaluated
jointly in terms of empirical coverage, interval width, and their ability
to represent high-demand events. No single approach performed uniformly
across all five categories.

---

## Reproducing

```bash
git clone https://github.com/wll-hayat04/retail-demand-forecasting.git
cd retail-demand-forecasting
pip install -e . --no-deps
pip install -r requirements.txt
```

Place `Online Retail.xlsx` and `products_to_categories.json` in `data/raw/`, then:

```python
from src.data import build_daily_dataset

build_daily_dataset()  # writes data/processed/daily_category_sales_clean.csv
```

Run all registered models for one category:

```python
import pandas as pd
from src import config, pipeline as pipe

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
daily = daily[~daily["Category"].isin(config.EXCLUDED_CATEGORIES)]

res = pipe.run_pipeline(daily, "Tealight Holders & Sets", X=7, Y=7)
res["results"]
```

Evaluate model differences over repeated splits:

```python
from src import validation as val

rep = val.repeated_evaluation(
    daily,
    "Tealight Holders & Sets",
    seeds=range(30),
    stratified=True,
)
val.summarise_repeats(rep)
val.is_difference_meaningful(
    rep,
    "Random Forest",
    "7-Day Rolling Sum Baseline",
)
```

`is_difference_meaningful` compares models on the **same splits**. If the reported confidence interval for their difference includes zero, this experiment does not establish a clear difference at the specified confidence level; it does not prove that their performance is identical.

---

## Project structure

```text
src/
  config.py       paths, holidays, feature list, split parameters
  data.py         Excel → cleaned daily sales, including cancellation matching
  features.py     targets, lags, rolling statistics, calendar and holiday features
  splits.py       block-shuffled and stratified block splits, leakage measure
  metrics.py      MAE, RMSE, WAPE, R², bias
  models.py       model registry; TabFM registered on demand
  pipeline.py     feature selection, training loop, horizon search
  validation.py   repeated evaluation, paired comparisons with confidence intervals

notebooks/
  01_data_preparation.ipynb   cleaning and aggregation
  02_modeling.ipynb           model comparison, X/Y horizon analysis,
                            error diagnostics, probabilistic forecasting,
                            and feature selection
  03_split_strategy.ipynb    comparison of four split strategies
  04_tabfm.ipynb             TabFM evaluation
  pipeline.ipynb             reusable pipeline demonstration
  RESULTS_54_AND_TABFM.md    category-level and TabFM results

results/                       CSV files backing the reported results
```

The model registry in `src/models.py` allows a new model to be added using one function and a decorator.

For additional results, see [`RESULTS_REVISED.md`](RESULTS_REVISED.md) and [`RESULTS_54_AND_TABFM.md`](notebooks/RESULTS_54_AND_TABFM.md).

---

## Methodological notes and limitations

**Splitting and leakage.** Random, chronological, time-series cross-validation, and block-shuffled splitting were explored. Block-shuffled and demand-stratified block splits made the observed evaluation results more stable, but target windows can overlap across train, validation, and test. `splits.leakage_ratio` measures this overlap. Stable results under these splits must **not** be interpreted as unbiased prospective forecast performance.

**Metrics and horizons.** WAPE normalizes absolute error by actual demand and is useful for comparing categories under an appropriate shared evaluation protocol. Its denominator changes with the evaluation sample and with X. Compare models using paired splits and the same (X, Y) target; report MAE, WAPE, and demand scale together.

**Limited history and rare events.** Approximately one year of data contains only one Christmas season and few distinct abrupt-demand episodes. Neither added model complexity nor a split strategy can recover predictive information that is absent from the historical and calendar features. External information available at prediction time, such as planned promotions, prices, and stock availability, could be investigated in future work.

**Scope of the conclusions.** The detailed X/Y, feature-selection, error, and probabilistic experiments concern selected categories and use exploratory validation protocols. Their conclusions should not be generalized automatically to all 54 categories or to future independent time periods.

---

*Internship project, Miningful — Hayat Waldi, EMI (Université Mohammed V de Rabat), June–August 2026. Supervisor: Nevio Dubbini.*
