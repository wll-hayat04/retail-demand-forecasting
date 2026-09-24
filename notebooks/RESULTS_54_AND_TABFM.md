# Results — 54 categories and TabFM

All figures use the corrected dataset described in section 1. No categories
are excluded: an earlier draft applied an exclusion rule to four categories
with implausible R², which turned out to be treating a symptom rather than
its cause.

---

## 1. A data cleaning error found through an implausible result

Running the pipeline over all 54 categories produced four categories with
absurd test R², the worst being **Children's Toys & Playsets at −2211**.

The first reaction was to treat these as degenerate evaluation windows and
exclude them by rule. That was wrong, and the check that disproved it was
simple: the test window was perfectly healthy — standard deviation 467, max
2191. The problem was in training.

The full series for that category had a maximum of **83,181 units against a
median of 1,211**. Tracing it to the raw transactions:

| InvoiceNo | Description | Quantity | Timestamp |
|---|---|---|---|
| 541431 | MEDIUM CERAMIC TOP STORAGE JAR | 74,215 | 2011-01-18 10:01 |
| C541433 | MEDIUM CERAMIC TOP STORAGE JAR | −74,215 | 2011-01-18 10:17 |
| 581483 | PAPER CRAFT, LITTLE BIRDIE | 80,995 | 2011-12-09 09:15 |
| C581484 | PAPER CRAFT, LITTLE BIRDIE | −80,995 | 2011-12-09 09:27 |

Two data-entry errors, each cancelled within 30 minutes by the same customer
for the same product. The original cleaning step dropped invoices beginning
with "C" — the cancellations — but kept the orders they reversed. 155,210
units that were never sold were therefore counted as sales.

**Why this destroys the metric.** The spike sits in the *training* split. Lag
and rolling features around it reach tens of thousands, so the model predicts
tens of thousands on test rows whose features look vaguely similar, against
actuals near 1,400. SSE becomes enormous while SST stays normal, and R²
collapses. It is a corrupted model evaluated on clean data, not a corrupted
evaluation — which is why excluding the category would have hidden the bug
rather than fixed it.

**The fix** matches each cancellation to its originating order on customer,
product and exactly opposite quantity, with the order preceding the
cancellation, and removes both. Implemented in `src/data.py`.

| | Before | After |
|---|---|---|
| Transaction rows | 527,792 | 524,957 |
| Max single transaction | 83,181 | 12,540 |
| Max daily category total | 83,181 | 8,855 |
| Children's Toys test WAPE | 440.5% | **15.6%** |

Of 9,288 cancellation rows, **2,899 (31%) were matched** to an original order.
The remainder reference orders placed before the dataset begins, or are
partial cancellations whose quantities do not align exactly. These are dropped
without a counterpart, which is the conservative choice.

---

## 2. Per-category model selection is no better than chance

Setup: all 54 categories, X=7 and Y=7 fixed, stratified block split, four
models. X and Y are deliberately not optimised per category — WAPE is not
comparable across different X, since the denominator grows with the window,
so medians over categories with different X would be meaningless.

For each category, the model with the lowest **validation** WAPE is compared
against the model that actually achieved the lowest **test** WAPE.

**The validation-selected model is the test winner in 13 of 54 cases (24%).**
Random choice among four models gives 25%.

A single category shows the mechanism clearly. On Tealight Holders & Sets:

| Model | Validation WAPE | Test WAPE |
|---|---|---|
| Linear Regression | **15.389** (1st) | 31.024 |
| Random Forest | 15.596 (2nd) | **22.657** (1st) |
| 7-Day Rolling Sum Baseline | 15.813 (3rd) | 42.482 (last) |
| XGBoost | 17.032 (4th) | 24.904 |

The top three are separated by 0.4 WAPE points on validation — far below the
seed-to-seed standard deviation of 3.5 to 6.2 measured on this category — and
the ranking inverts entirely on test.

Across all categories, what is chosen and what wins diverge:

| Model | Selected on validation | Actual test winner |
|---|---|---|
| Random Forest | 18 | **21** |
| XGBoost | 16 | 11 |
| 7-Day Rolling Sum Baseline | 10 | 11 |
| Linear Regression | 10 | 11 |

Note the effect of the data fix: before it, selection scored 20/54 (37%).
Cleaning the data made selection look *worse*, not better. The corrupted
categories were ones where validation and test failed together, producing
agreements that were artefacts rather than successful selections.

### The cost of a wrong choice

Selection loss = test WAPE of the chosen model minus test WAPE of the best
available model. Zero when selection succeeds.

| Statistic | Value |
|---|---|
| Median | 6.13 |
| Mean | 10.32 |
| 90th percentile | 33.47 |
| Maximum | 50.00 |

**18 of 54 categories lose under 2 points; 18 of 54 lose more than 10.**

The distribution is strongly right-skewed — mean 10.32 against median 6.13.
Selection is harmless in a third of cases and severe in another third:

| Category | Selected | Test WAPE | Should have been | Its WAPE | Loss |
|---|---|---|---|---|---|
| Easter Decorations | Linear Regression | 73.6 | Baseline | 23.6 | 50.0 |
| Jigsaw Puzzles & Board Games | Linear Regression | 69.0 | XGBoost | 29.0 | 40.0 |
| Egg Cups & Holders | Baseline | 84.8 | XGBoost | 45.1 | 39.8 |
| Hair Accessories | Baseline | 88.0 | Random Forest | 51.3 | 36.7 |
| Mirrors | XGBoost | 70.9 | Linear Regression | 37.0 | 33.9 |

Easter Decorations is the clearest case: validation selected a model three
times worse than the naive baseline, on a strongly seasonal category where
the choice matters most.

### A single fixed model beats selection

| Strategy | Median test WAPE | Mean | Categories >10 pts off oracle |
|---|---|---|---|
| Select per category (validation) | 36.5 | 42.8 | **18/54** |
| **Always Random Forest** | 35.8 | **38.0** | **10/54** |
| Always XGBoost | 36.0 | 39.0 | 13/54 |
| Always Baseline | 43.2 | 49.3 | 32/54 |
| Always Linear Regression | 44.8 | 51.2 | 28/54 |
| Oracle (best on test) | 30.1 | 32.5 | 0 |

**Applying Random Forest everywhere beats per-category selection by 4.8 WAPE
points on average, with 10 catastrophic errors instead of 18.** Selection is
not merely unhelpful — it actively degrades results by adding variance
without adding information.

The oracle sits 5.5 points below Random Forest. Perfect selection would gain
little, which is precisely why a noisy selection procedure loses more than it
gains: the prize is small relative to the estimation error.

### Recommendation

At this data volume, apply **a single model across all categories** rather
than selecting per category. Random Forest is the safest default. This also
simplifies operations: one model to train, deploy and monitor instead of 54.

Per-category selection becomes worthwhile only with enough data to estimate
it reliably. A practical threshold could be established by repeating this
analysis on progressively longer subsamples of a multi-year series.

### Overall performance

Median test WAPE across 54 categories is **36.5%**, with positive test R² in
**20 of 54**. Roughly two thirds of categories are not predicted better than
their own mean at a 7-day horizon with 7 days of lead time.

---

## 3. TabFM

TabFM (Google Research, June 2026) is a zero-shot tabular foundation model.
It does not train: `.fit()` stores the training rows as context and
`.predict()` runs a single forward pass. No hyperparameters, no feature
selection.

Evaluated on the same protocol as everything else — 30 stratified splits,
five categories, X=7 and Y=7 — with `n_estimators=32`, the library default.

### Accuracy

| Category | TabFM WAPE | Best competitor | TabFM R² | Win rate |
|---|---|---|---|---|
| Tealight Holders & Sets | **17.8 ± 3.5** | 21.9 (Baseline) | 0.648 | **0.700** |
| Jewellery - Earrings | **73.2 ± 9.0** | 77.4 (XGBoost) | 0.370 | **0.633** |
| Jumbo Bags & Shoppers | 31.0 ± 5.4 | 30.5 (LinReg) | 0.411 | 0.367 |
| Christmas Decorations | 26.5 ± 7.7 | 22.5 (Baseline) | **0.912** | 0.333 |
| Cake Cases & Baking | 33.5 ± 3.4 | 32.6 (RF) | 0.164 | 0.200 |

Paired comparison across the same 30 splits gives a clearer picture than the
means, because pairing cancels the split-to-split variation:

**15 of 20 comparisons favour TabFM, 4 fall within noise, 1 is a loss.**

| Comparison | Result |
|---|---|
| vs Random Forest | Wins in 4 of 5 categories, never loses |
| vs XGBoost | Wins in 4 of 5, never loses |
| vs Linear Regression | Wins in 3 of 5, never loses |
| vs Baseline | Wins in 4 of 5, loses on Christmas Decorations |

The single loss is against the naive 7-day rolling sum on Christmas
Decorations: +3.97 WAPE points, CI [+0.68, +7.26].

Two results are worth isolating. On **Tealight**, TabFM reaches 17.8 WAPE
against 22.3 for Random Forest with a *lower* standard deviation, winning 21
of 30 draws — the highest win rate any model achieves anywhere in this study.
On **Jewellery**, the sparsest category, R² rises from 0.15 to 0.37, which is
where a model carrying strong priors over small tables would be expected to
help most.

Note also that on Christmas Decorations TabFM has the best R² in the table
(0.912) while losing on WAPE. The two metrics disagree on identical
predictions — the same denominator effect documented in `RESULTS_REVISED.md`.

### Cost

| | Time |
|---|---|
| TabFM, one fit+predict, CPU (i5-1135G7, bfloat16) | **536.7 s** |
| TabFM, one fit+predict, GPU (Tesla T4) | 10.5 s |
| TabFM, 30 seeds on one category, GPU | ~1,372 s |
| Random Forest, one fit+predict, CPU | 1.26 s |
| Linear Regression, one fit+predict, CPU | 0.21 s |

In one local CPU benchmark on an Intel i5-1135G7, TabFM required **536.7 s** for one fit-and-predict call, compared with **1.26 s** for Random Forest (about **426× slower**).
On a Tesla T4, a TabFM fit-and-predict call took approximately **10.5 s**, about **51× faster than the local CPU benchmark**.
These timings are hardware- and configuration-specific and are reported as empirical measurements rather than as a general speed ratio for TabFM.

Two further constraints. The model ships 6.6 GB of weights for the regression
head alone (13 GB for the full repository), and loading it in float32
exhausted the Windows paging file on a 16 GB machine — bfloat16 was required,
which is emulated and slow on CPUs without native support.

**Licensing:** the TabFM source code is Apache-2.0, but the pretrained weights
are distributed under `tabfm-non-commercial-v1.0` and are restricted to
non-commercial, non-production use. Adequate for this study; a blocker for any
deployment.

### Assessment

TabFM is the most accurate model in this study, and it achieves that with no
tuning and no feature engineering — the four classical models each required a
feature set built by hand.

But the margin is 2 to 5 WAPE points, against 6.6 GB of weights, a PyTorch
dependency, a GPU requirement, and a non-commercial licence. For this problem,
at this scale, that trade is not obviously worth making. The finding is more
useful as evidence about *where* foundation models help — sparse, small-sample
tabular problems — than as a deployment recommendation.

---

## 4. Limitations

**One year of data contains exactly one Christmas.** No split strategy or
model creates seasonal signal that is not there, and this is the binding
constraint on every result above.

**Residual leakage.** The block-shuffled split accepts overlapping target
windows between splits: measured at 57% of test rows having a training
neighbour within X−1 days on Tealight Holders & Sets. Results are therefore
optimistic relative to true deployment performance. Purged cross-validation
with an embargo period would remove this properly and is the natural next
step.

**Single seed for the 54-category run.** Each category is evaluated on one
split, so no individual row of `pipeline_results_all_categories.csv` should be
quoted on its own. The aggregate over 54 categories carries the conclusion;
the noise in each row averages out across them.

**TabFM evaluated on five categories, not all 54**, for the runtime reasons
given above. The five span the behavioural range — high-volume regular,
high-volume volatile, strongly seasonal, and sparse — but they were selected
before this analysis and are not a random sample.
