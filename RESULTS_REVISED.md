# Revised Results

Replaces the "Final Results" section of `project_context.md`.

Every figure below is the mean over **30 stratified block-shuffled splits**,
reported with its standard deviation. The previous table gave single-seed
point estimates measured on the same split used for model selection.

---

## Why the numbers changed

The original table reported one number per category, obtained with
`random_state=42`. Re-running the identical code across 30 seeds showed those
numbers were not reproducible: for Tealight Holders & Sets, validation WAPE
ranged from 11.9% to 52.1% depending only on which blocks landed in which
split. The published 15.8% sat near the best draw of thirty.

Three separate issues compounded:

1. **Selection on the evaluation set.** X, Y and the model were all chosen on
   validation, then validation was reported as the result.
2. **A denominator artifact in WAPE.** WAPE is `sum|error| / sum|actual|`.
   Across 30 seeds, WAPE correlated **−0.587** with the evaluation set's
   denominator while MAE correlated **+0.863** — the two metrics disagreed on
   the same predictions. Validation WAPE was substantially measuring which
   weeks landed in validation, not forecast quality.
3. **Seasonal imbalance between splits.** With ~24 blocks in a one-year
   series, some draws put the entire Christmas ramp in one split and none of
   it in another.

Issue 3 was addressed by stratifying blocks by demand before assignment
(see below). Issues 1 and 2 are addressed by reporting test metrics with
uncertainty, and by treating paired comparisons rather than absolute WAPE
levels as the basis for conclusions.

---

## Final results table

At the X and Y values selected by the original optimisation, stratified
split, 30 seeds:

| Category | X | Y | Best model | WAPE (mean ± sd) | Previously claimed |
|---|---|---|---|---|---|
| Cake Cases & Baking Accessories | 24 | 14 | Random Forest | **10.0 ± 3.0** | 6.9 |
| Christmas Decorations | 7 | 6 | 7-Day Rolling Sum Baseline | **21.9 ± 4.4** | 19.6 |
| Jumbo Bags & Shoppers | 16 | 8 | Random Forest | **24.9 ± 5.4** | 11.3 |
| Tealight Holders & Sets | 7 | 3 | XGBoost | **27.4 ± 10.4** | 12.5 |
| Jewellery - Earrings | 28 | 7 | Random Forest | **51.2 ± 11.3** | 9.7 |

**Model selection: 3 of 5 original choices confirmed, 2 corrected.**
Cake Cases, Christmas and Jumbo Bags keep the model originally identified.
Jewellery moves from XGBoost to Random Forest, and Tealight from the Baseline
to XGBoost.

**Both corrections have narrow margins** and should be stated as such:

| Category | Comparison | Mean diff | 95% CI | Wins |
|---|---|---|---|---|
| Jewellery - Earrings | RF vs XGBoost | −2.86 | [−5.45, −0.28] | 73% |
| Tealight Holders & Sets | XGBoost vs Baseline | −4.31 | [−8.08, −0.54] | 60% |

Tealight in particular is a 4.3-point mean difference against a 10.5-point
standard deviation. The effect is real but small relative to the spread.

---

## How far off were the original figures

| Category | Claimed | Measured | Ratio |
|---|---|---|---|
| Christmas Decorations | 19.6 | 21.9 | 1.1× |
| Cake Cases & Baking | 6.9 | 10.0 | 1.4× |
| Jumbo Bags & Shoppers | 11.3 | 24.9 | 2.2× |
| Tealight Holders & Sets | 12.5 | 31.7 | 2.5× |
| Jewellery - Earrings | 9.7 | 54.1 | 5.6× |

Christmas was essentially correct — the claimed value falls inside the
measured interval. Jewellery is the serious case: 9.7% sits roughly four
standard deviations below the mean. It is the sparsest category (~60%
zero-demand days) and therefore the most unstable, so a single seed was most
likely to mislead there.

The `claimed_model_winrate` column of `results/final_table_revised.csv` is
worth reporting directly: no model exceeded **0.70**, and three were near
**0.30**. Even where the original model choice was correct on average, it won
only about one draw in three.

---

## Model comparison at fixed X=7, Y=7

Stratified split, 30 seeds, all five categories. This is the like-for-like
comparison across categories, since WAPE is not comparable across different X.

| Category | Best model | WAPE (mean ± sd) | Baseline win rate |
|---|---|---|---|
| Cake Cases & Baking Accessories | Random Forest | 32.6 ± 3.9 | 0.000 |
| Christmas Decorations | 7-Day Baseline | 22.5 ± 4.6 | 0.700 |
| Jumbo Bags & Shoppers | Linear Regression | 30.5 ± 5.6 | 0.200 |
| Tealight Holders & Sets | 7-Day Baseline | 21.9 ± 5.4 | 0.367 |
| Jewellery - Earrings | XGBoost | 77.1 ± 5.7 | 0.000 |

**The naive baseline never wins a single draw out of 30** in Cake Cases,
Jumbo Bags or Jewellery. That the ML models add value in those categories is
no longer a matter of interpretation.

Paired comparison, Random Forest against the baseline:

| Category | Mean diff | 95% CI | RF wins | Verdict |
|---|---|---|---|---|
| Cake Cases & Baking | −13.79 | [−15.32, −12.25] | 100% | RF clearly better |
| Jewellery - Earrings | −32.90 | [−38.81, −26.99] | 93% | RF clearly better |
| Jumbo Bags & Shoppers | −3.73 | [−6.06, −1.40] | 70% | RF better |
| Christmas Decorations | +6.99 | [+3.34, +10.65] | 23% | Baseline better |
| Tealight Holders & Sets | +0.39 | [−1.43, +2.20] | 57% | **Indistinguishable** |

Note that at X=7/Y=7 Jumbo Bags is best served by Linear Regression (30.5)
rather than Random Forest (33.1) — a linear model beating two tree ensembles
suggests little exploitable non-linearity at this sample size.

---

## Split strategy

`split_comparison.ipynb` selected block-shuffled over random, chronological
and TimeSeriesCV. That conclusion stands. What it could not show is that the
chosen split was itself unstable.

Stratifying blocks by mean demand before assignment, 30 seeds, Random Forest:

| Category | WAPE sd (original → stratified) | Worst case | Denominator CV |
|---|---|---|---|
| Tealight Holders & Sets | 8.2 → 6.2 | 45 → 34 | 0.181 → 0.066 |
| Cake Cases & Baking | 9.7 → **3.9** | 51 → 39 | 0.147 → 0.053 |
| Jumbo Bags & Shoppers | 11.0 → **4.6** | 73 → 45 | 0.235 → 0.053 |
| Christmas Decorations | 42.8 → **8.8** | 209 → 55 | 0.747 → 0.107 |
| Jewellery - Earrings | 53.5 → **4.6** | 360 → 84 | 0.600 → 0.090 |

The ranking of categories by original denominator CV (Christmas 0.747,
Jewellery 0.600) is exactly the ranking by instability. After stratification
all five fall between 0.053 and 0.107: the evaluation sets are finally
comparable across seeds.

Mean WAPE *rises* for Cake Cases (29.9 → 32.6). This is not a degradation —
it is the disappearance of favourable draws that were pulling the average
down.

---

## What this does not fix

Stratification reduces the variance of the estimate. It does not create
signal. The dataset contains 374 days, one Christmas, ~230 training rows per
category and 24 blocks. No split strategy changes that, and it is the binding
constraint on every result above.

Practical reading of the numbers: **Cake Cases at 10% WAPE is usable.
Jumbo Bags and Tealight at 25–27% are weak for replenishment planning.
Jewellery at 51% is not usable** — the error is roughly half the quantity
being predicted. And for Christmas, the most seasonal category, the winner is
a 7-day rolling sum, meaning the models add nothing there.

---

## Conclusions that no longer hold

- **"Longer X dramatically improves WAPE (Jewellery 73% → 9.7%)."** Widening
  X grows the WAPE denominator roughly linearly while daily errors partly
  cancel, so WAPE falls with X substantially by construction. Comparing WAPE
  across different X compares different prediction problems. X should be set
  by the business replenishment cycle and performance reported at that X.

- **"Correlated category features degrade 4/5 categories."** Decided on a
  single 42-row validation set, which cannot resolve differences of that
  size. Currently unsupported in either direction — not disproven, untested.

- **"Hyperparameter tuning didn't help."** Very likely true, but for a
  sharper reason than originally given: the seed-to-seed standard deviation
  (4–10 WAPE points) exceeds any plausible tuning effect on ~230 training
  rows. The tuning signal was below the noise floor.

---

## Note on the optimisation procedure

`optimize_XY` searches X and Y using the **baseline only**, then the best
model is selected afterwards. The horizons are therefore tuned for the
rolling-sum baseline and inherited by Random Forest and XGBoost. Worth
stating explicitly as a limitation.
