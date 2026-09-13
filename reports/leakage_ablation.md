# Leakage ablation: split strategy x feature set

One model (`rf_deep`: RandomForest, 400 trees, max_depth 24, min_samples_leaf 2,
class_weight balanced, seed 42) trained four times. Only the split and the
feature set change, so each delta is attributable.

## Results

| split | features | train | test | base rate | **PR-AUC** | lift | ROC-AUC | precision | recall | F1 | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| athlete | old | 215,844 | 53,887 | 0.1478 | 0.5744 | 3.89x | 0.8489 | 0.5838 | 0.4757 | 0.5242 | 0.8723 |
| athlete | new | 215,844 | 53,887 | 0.1478 | **0.7175** | 4.85x | 0.8927 | 0.7335 | 0.5650 | 0.6383 | 0.9053 |
| temporal | old | 238,232 | 31,499 | 0.1448 | 0.3893 | 2.69x | 0.7637 | 0.4773 | 0.2561 | 0.3333 | 0.8517 |
| **temporal** | **new** | 238,232 | 31,499 | 0.1448 | **0.4396** | **3.04x** | 0.7913 | 0.5395 | 0.2817 | 0.3702 | 0.8612 |

The bottom row is the number to report.

## What each change cost or bought

**Temporal split: -0.185 PR-AUC** (0.5744 -> 0.3893, holding features fixed).
The athlete-grouped split was inflating the headline by ~47%. It closes athlete
leakage but still trains on 2016 to predict 1924, and the medal rate itself
drifts hard across eras -- 37.6% in the 1890s down to ~14% from the 1960s on as
fields grew. Scoring on a random slice of all eras rewards interpolating a trend
that real use requires extrapolating.

**New features: +0.143 PR-AUC on the athlete split, +0.050 on the temporal one.**
Most of that is field/roster size, not the missingness flags (see below).

## Feature importances (temporal split, new features)

| feature | importance |
|---|---|
| noc_medal_rate | 0.2604 |
| **field_size** | **0.1501** |
| sport_medal_rate | 0.1212 |
| year | 0.1020 |
| **team_size** | **0.0988** |
| age | 0.0891 |
| weight | 0.0758 |
| height | 0.0718 |
| sex | 0.0110 |
| season | 0.0085 |
| height_missing | 0.0059 |
| weight_missing | 0.0054 |

`field_size` + `team_size` together account for 0.249 -- essentially as much as
country medal rate. That signal existed before; it was just being absorbed by
the athlete features, where it read as skill. A team gold produces one medal row
per athlete (up to 38 rows for a single 1908 gymnastics result), and events with
<=8 entrants medal at 55% versus 12% for fields of 50-100.

## Negative result: the missingness flags barely matter

The flags were added on a strong prior. Marginally, `height_missing` correlates
with the target at 0.0007 -- apparently useless. But that is Simpson's paradox:
missingness is driven by era (corr(year, height_missing) = -0.65), and *within*
a decade a missing measurement tracks a much lower medal rate:

| decade | medal rate, height present | height missing |
|---|---|---|
| 1920s | 30.6% | 18.3% |
| 1950s | 18.9% | 10.6% |
| 1980s | 15.4% | 2.6% |
| 2010s | 14.3% | 1.6% |

The within-era gap is real and large. The tree still ranks both flags last
(0.0059 / 0.0054), because it already has `year` and can reconstruct most of the
effect from the era-conditional structure of the other features. Keeping them:
they cost nothing, they make the assumption explicit rather than silently
median-filled, and they would matter more for a linear model.

## Reproducing

```
python -m src.train --config configs/config.yaml          # temporal (default)
```

Set `data.split_strategy: athlete` in `configs/config.yaml` to reproduce the old
numbers. The ablation itself is in `reports/leakage_ablation.csv`.

Note: these four runs were executed outside MLflow, so `mlflow.db` and
`models/model_bundle.joblib` are unchanged and still hold the pre-fix
athlete-split model. Re-run the training pipeline in your own venv to refresh
them.
