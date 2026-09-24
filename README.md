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

**TabFM: predictive performance and computational constraints.**
TabFM, Google's zero-shot tabular foundation model, achieved lower
forecasting error in 15 of 20 paired comparisons across five categories,
without task-specific tuning.

Initial experiments were conducted locally on an Intel i5-1135G7 CPU.
A TabFM call required approximately 536.7 seconds, compared with
0.87 seconds for Random Forest under the measured configuration.
The full local evaluation could not be completed because loading the
approximately 6.6 GB model weights exhausted the available Windows
virtual memory.

The full TabFM evaluation was therefore performed on Kaggle using
an NVIDIA Tesla T4 GPU. GPU execution reduced the time per call to
approximately 10.5 seconds, around 51 times faster than the local
CPU measurement.

The GPU experiments covered five product categories and 30 repeated
splits, using 32 estimators. Their results were saved in
`results/tabfm_comparison_raw.csv`,
`results/tabfm_verdicts.csv`, and
`results/tabfm_comparison_summary.csv`.

These findings illustrate the trade-off between predictive accuracy,
computational cost, and hardware requirements. The pretrained weights
used in this project are also subject to non-commercial-use restrictions.

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

### Classical forecasting environment

The classical forecasting pipeline has been validated with Python 3.11.
From the repository root, create a virtual environment and install the
project dependencies:

```bash
git clone https://github.com/wll-hayat04/retail-demand-forecasting.git
cd retail-demand-forecasting

python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then install the dependencies and run the tests:

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

`requirements-lock.txt` records the dependency versions of the validated
Windows/Python 3.11 environment. It is an environment snapshot rather than
a guarantee of identical installations across operating systems.

### Data preparation

The raw UCI Online Retail workbook and the product-to-category mapping
are not included in the repository. Place `Online Retail.xlsx` and
`products_to_categories.json` in `data/raw/`, then run:

```python
from src.data import build_daily_dataset

build_daily_dataset()
```

This generates `data/processed/daily_category_sales_clean.csv`.

Cancellation matching uses a heuristic based on customer, product,
opposite quantity, transaction date, and the availability of an unmatched
original purchase. The matching decisions can be inspected in
`results/cancellation_matching_audit.csv` and reproduced with
`scripts/audit_cancellations.py`. The matching procedure does not establish
that every cancellation can be linked unambiguously to its originating
purchase.

### Running the forecasting pipeline

```python
import pandas as pd
from src import config, pipeline as pipe

daily = pd.read_csv(config.DAILY_CLEAN, parse_dates=["Date"])
daily = daily[~daily["Category"].isin(config.EXCLUDED_CATEGORIES)]

res = pipe.run_pipeline(
    daily,
    "Tealight Holders & Sets",
    X=7,
    Y=7,
    split_strategy="purged_chronological",
)

print(res["results"])
print(res["split_overlap_report"])
```

`purged_chronological` orders observations by reference date and removes
boundary observations whose target labels would not be fully observable
before the next partition begins. The reported overlap diagnostics concern
the target windows; zero target-window overlap does not eliminate all
temporal dependence or distribution shift.

For the historical exploratory block-shuffled and demand-stratified
experiments, use the corresponding split strategy explicitly. These
protocols must not be confused with purged chronological evaluation.

### Repeated model comparisons

```python
from src import validation as val

rep = val.repeated_evaluation(
    daily,
    "Tealight Holders & Sets",
    seeds=range(30),
    stratified=True,
)

summary = val.summarise_repeats(rep)

comparison = val.is_difference_meaningful(
    rep,
    "Random Forest",
    "7-Day Rolling Sum Baseline",
)
```

This reproduces an exploratory demand-stratified comparison rather than a
prospective evaluation. Model differences are computed on paired splits.
A confidence interval that contains zero does not establish a clear
difference at the specified confidence level; it does not prove that
the models have identical performance.

### TabFM environment and results

TabFM is optional and is not installed by the classical environment
requirements. Its experiments require a separate compatible PyTorch/TabFM
environment and access to the pretrained regression checkpoint.

The final five-category TabFM comparison was conducted separately on a
Tesla T4 GPU, using 30 paired demand-stratified splits and
`n_estimators=32`. Its saved results are available in:

- `results/tabfm_comparison_raw.csv`
- `results/tabfm_comparison_summary.csv`
- `results/tabfm_verdicts.csv`

`notebooks/04_tabfm.ipynb` contains exploratory local CPU experiments.
Its previously saved execution outputs were cleared because they came
from different runtime configurations. Running this notebook may require
substantial memory and computation time. Reproducing the classical
pipeline does not require running TabFM.

---

## Project structure

```text
src/
  config.py       paths, holidays, features, and experiment parameters
  data.py         transaction cleaning, cancellation matching, daily aggregation
  features.py     future-demand targets and past-only predictive features
  splits.py       block-shuffled, stratified, and purged chronological splits;
                  target-window overlap diagnostics
  metrics.py      forecasting error metrics
  models.py       baseline and machine-learning model registry; optional TabFM
  pipeline.py     model evaluation, feature selection, and horizon analysis
  validation.py   repeated evaluations and paired model comparisons

notebooks/
  01_data_preparation.ipynb   data preparation and diagnostics
  02_modeling.ipynb           model and forecasting-horizon experiments
  03_split_strategy.ipynb    historical split-strategy experiments
  04_tabfm.ipynb             exploratory local TabFM experiments
  pipeline.ipynb             pipeline demonstration
  RESULTS_54_AND_TABFM.md    category-level and TabFM results

scripts/
  audit_cancellations.py      cancellation-matching audit

tests/
  test_cancellations.py       cancellation-matching tests
  test_features.py            feature and target-construction tests
  test_temporal_validation.py temporal validation and baseline tests

results/                       saved experimental results and audit CSV
requirements.txt               classical environment and development dependencies
requirements-lock.txt          validated environment dependency snapshot
```

For additional experimental results, see
[`RESULTS_REVISED.md`](RESULTS_REVISED.md) and
[`RESULTS_54_AND_TABFM.md`](notebooks/RESULTS_54_AND_TABFM.md).

---

## Methodological notes and limitations

**Splitting and leakage.** Random, chronological, time-series, block-shuffled, and demand-stratified splitting strategies were explored. Block-shuffled splits can retain overlapping future-demand target windows across partitions. Demand-stratified splitting additionally uses observed target demand to construct the partitions, so it is not a prospective validation protocol. The pipeline now also supports a purged chronological split (`split_strategy="purged_chronological"`), which removes boundary observations until target labels in an earlier partition would be fully observable before the following partition begins. `split_overlap_report` checks target-window overlap between partitions. Historical results obtained with block-shuffled and demand-stratified splits remain exploratory and have not been retroactively replaced by purged chronological results.


**Metrics and horizons.** WAPE normalizes absolute error by observed
demand. Its denominator changes with the evaluation sample and with
the forecast-window length X. Model comparisons should use the same
target definition and paired evaluation samples, with MAE, WAPE, and
actual-demand scale reported together. The seven-day rolling-sum baseline
is the historical reference for X = 7; a horizon-scaled baseline is
available for other forecast-window lengths.

**Cancellation matching.** Matching cancellations to purchases is
heuristic. Some cancellation rows have no eligible original purchase,
while others have multiple possible matches. The audit records these
cases, but does not independently verify the underlying transaction
relationships. The two exceptional high-quantity cancellations were
matched to unique available candidates under the implemented matching
rule.

**Limited history and rare events.** Approximately one year of retail
data contains only one Christmas season and relatively few independent
episodes of abrupt demand. Neither increased model complexity nor
a different splitting strategy can recover information absent from
the available predictors. Future work could investigate additional
years of data and operational variables known at forecasting time,
such as planned promotions, prices, and stock availability.

**Scope of the conclusions.** The detailed horizon, feature-selection,
error-analysis, probabilistic-forecasting, and TabFM experiments concern
selected categories. Most historical comparisons used exploratory
evaluation protocols. Their observed performance should not be
generalized automatically to all categories or to independent future
periods.

---

*Internship project, Miningful — Hayat Waldi, EMI (Université Mohammed V de Rabat), June–August 2026. Supervisor: Nevio Dubbini.*
