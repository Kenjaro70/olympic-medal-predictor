# 🥇 Olympic Medal Predictor

[![CI](https://github.com/Kenjaro70/olympic-medal-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Kenjaro70/olympic-medal-predictor/actions/workflows/ci.yml)

An end-to-end intelligent application that predicts an athlete's probability of
winning an Olympic medal — and lets you ask about it in plain English.

Ask: *"I'm a 24-year-old female swimmer from the USA, 175 cm and 63 kg — what
are my medal chances?"* An LLM parses the athlete's details out of your
question, a trained scikit-learn model estimates the medal probability, and the
LLM explains the result in context, including what was assumed and what the
model can't know.

**Who it's for / problem it solves:** a demo-quality analytics tool for fans,
journalists, or aspiring athletes who want a data-grounded, conversational
answer to "how likely is an athlete like X to medal?" — without knowing
anything about feature vectors or model APIs.

Built as the capstone for the TripleTen AI/ML Engineering Bootcamp.

## Dataset

[120 years of Olympic history: athletes and results](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)
— 271,116 athlete-event rows, 1896–2016. The data is downloaded from the
[TidyTuesday 2021-07-27 mirror](https://github.com/rfordatascience/tidytuesday/blob/master/data/2021/2021-07-27/readme.md)
of the same Kaggle dataset (public HTTPS, no Kaggle credentials needed).

**Task:** binary classification — did the athlete win a medal (Gold/Silver/
Bronze) in an event entry? Base rate ≈ 14.7%.

## Setup

```bash
git clone https://github.com/Kenjaro70/olympic-medal-predictor.git
cd olympic-medal-predictor

python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Get the data (~35 MB)
mkdir -p data/raw
curl -L -o data/raw/olympics.csv https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-07-27/olympics.csv

# Configure the LLM provider (Nebius AI Studio by default)
cp .env.example .env   # then paste your NEBIUS_API_KEY into .env
```

API keys are read from environment variables only — nothing is hardcoded, and
`.env` is gitignored. Any OpenAI-compatible provider works via `LLM_BASE_URL`
and `LLM_MODEL` overrides (see `.env.example`).

## Usage

```bash
# 1. Train all 6 configured experiments (logs everything to MLflow)
python -m src.train --config configs/config.yaml

# 2. Compare runs and identify the best model programmatically
#    (--export refreshes the committed run-history files in reports/)
python -m src.compare_runs --export reports

# 3. Launch the chat interface
streamlit run src/app.py
# or the CLI: python -m src.cli "30-year-old male judoka from Japan, will he medal?"

# Inspect experiments in the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Run the test suite
pytest tests/ -v
```

### Docker (bonus)

```bash
python -m src.train                       # produce models/model_bundle.joblib first
docker build -t olympic-medal-predictor .
docker run -p 8501:8501 --env-file .env olympic-medal-predictor
```

Then open http://localhost:8501.

## Architecture

```
user question (natural language)
        │
        ▼
┌───────────────────────────┐   extraction prompt returns strict JSON
│ LLM: feature extraction   │──── {sex, age, height, weight, noc, sport,
└───────────────────────────┘      season, year, out_of_scope}
        │ normalize + validate (src/llm_interface.py)
        ├── out of scope ──────────► polite scope explanation
        ├── missing required ──────► clarifying question (sex/age/country/sport)
        ▼
┌───────────────────────────┐   models/model_bundle.joblib = best MLflow run
│ trained sklearn model     │   + fitted medians/encoders/scaler so inference
└───────────────────────────┘   preprocessing matches training exactly
        │ medal probability
        ▼
┌───────────────────────────┐   response prompt gets prediction + parsed
│ LLM: response generation  │   features + imputed-field list + model card
└───────────────────────────┘
        │
        ▼
conversational answer with probability, context, and caveats
```

**Preprocessing** (`src/preprocess.py`): dedupe, binary target from `medal`,
median imputation for age/height/weight, binary encoding for sex/season, and
target (medal-rate) encoding for the high-cardinality NOC and sport columns.
All statistics are fit on the training split only and reapplied to test and
inference data, so there is no leakage. The train/test split is **grouped by
athlete id** so the same athlete never appears on both sides.

**Training** (`src/train.py`): reads all hyperparameters from
`configs/config.yaml` — six configurations across three algorithms (logistic
regression, random forest, histogram gradient boosting). Every run logs
params, data description, five metrics, and the model artifact to MLflow
(SQLite backend). The best run by ROC AUC is bundled with the preprocessing
state for the app.

**Comparison** (`src/compare_runs.py`): uses `mlflow.search_runs()` to rank
all runs and print the winner.

**Measurement** (`scripts/report_numbers.py`): re-measures the leakage
ablation and the threshold sweep and writes `reports/leakage_ablation.md`, so
the numbers quoted in this README are output rather than hand-copied. Run it
after any change to preprocessing or features.

## Results

> **Headline: PR-AUC 0.440 on a 2012-2016 holdout (3.04x the 14.5% base rate).**
> An earlier version of this README reported ROC-AUC 0.853 on an
> athlete-grouped random split. That split was too easy -- see
> [Honest evaluation](#honest-evaluation) below and
> [`reports/leakage_ablation.md`](reports/leakage_ablation.md). The table in
> this section is the old athlete-split result, kept for comparison.

Test-set metrics (20% held-out, grouped by athlete; base rate 14.7%):

| run | model | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|
| **rf_deep** (selected) | random forest | 0.845 | 0.481 | 0.613 | **0.539** | **0.853** |
| hist_gb_tuned | hist gradient boosting | 0.880 | 0.815 | 0.240 | 0.370 | 0.843 |
| hist_gb_default | hist gradient boosting | 0.873 | 0.814 | 0.186 | 0.302 | 0.823 |
| rf_shallow | random forest | 0.711 | 0.303 | 0.737 | 0.429 | 0.804 |
| logreg_strong_reg | logistic regression | 0.699 | 0.277 | 0.641 | 0.386 | 0.741 |
| logreg_baseline | logistic regression | 0.699 | 0.277 | 0.641 | 0.386 | 0.741 |

The boosted models post the highest raw accuracy but do it by rarely
predicting "medal" (recall ≈ 0.19–0.24) — accuracy is misleading at a 14.7%
base rate. **rf_deep** is selected: best ROC AUC (0.853) *and* best F1
(0.539), meaning its probabilities rank athletes best while keeping a sane
precision/recall balance.

**Model selection:** ROC AUC is the primary metric because the classes are
imbalanced (~15% positives) and the app surfaces a *probability*, so ranking
quality matters more than a 0.5-threshold accuracy. Class-weighted models keep
recall on the rare medal class honest.

**Findings:** the strongest signals are the country's and sport's historical
medal rates — powerhouse countries and small-field sports medal far more
often. Demographics (age/height/weight) add only modest lift, which makes
sense: within elite athletes, body metrics don't separate medalists well. The
model predicts *base rates for an athlete profile*, not individual talent —
the LLM layer states this caveat in every answer.

### Honest evaluation

The athlete-grouped split closes athlete leakage but still lets the model train
on 2016 to predict 1924. The medal rate itself drifts -- 37.6% in the 1890s down
to ~14% from the 1960s on -- so a random slice of all eras rewards interpolating
a trend that real use requires extrapolating. The default split is now
**temporal**: train on <= 2008, test on 2012-2016.

Same model (`rf_deep`), four runs, one variable at a time. These numbers are
**generated, not transcribed** -- regenerate them with
`python -m scripts.report_numbers`, which writes
[reports/leakage_ablation.md](reports/leakage_ablation.md) with mean ± sd over
three seeds:

| split | features | PR-AUC | lift vs base | ROC-AUC | F1 |
|---|---|---|---|---|---|
| athlete | original | 0.5744 | 3.89x | 0.8489 | 0.5242 |
| athlete | + size & missingness | 0.7175 | 4.85x | 0.8927 | 0.6383 |
| temporal | original | 0.3893 | 2.69x | 0.7637 | 0.3333 |
| **temporal** | **+ size & missingness** | **0.4396** | **3.04x** | 0.7913 | 0.3702 |

Moving to a temporal holdout costs 0.185 PR-AUC -- the old headline was inflated
by roughly 47%. Two features were added to stop the model absorbing opportunity
as skill:

- **`field_size` / `team_size`** -- a team gold produces one medal row per
  athlete (up to 38 for one 1908 gymnastics result), and events with <=8
  entrants medal at 55% vs 12% for fields of 50-100. Together these rank second
  and fifth in importance (0.150 and 0.099).
- **`height_missing` / `weight_missing`** -- whether a measurement was recorded
  is era-driven (corr with year -0.65), and within a decade a missing value
  tracks a much lower medal rate (1980s: 15.4% present vs 2.6% missing).
  Honest caveat: the tree ranks both flags last, because `year` already lets it
  reconstruct most of the effect. Kept for explicitness, not for lift.

**Metric change:** selection is now `pr_auc`, not `roc_auc`. At a 14.5% base
rate a model that never predicts "medal" scores 85.5% accuracy, and ROC-AUC
stays flattering under imbalance. `pr_auc_lift` (PR-AUC / base rate) is logged
alongside so a value near 1.0 reads immediately as "learned nothing".

Set `data.split_strategy: athlete` in `configs/config.yaml` to reproduce the
older numbers.

### Operating point

The app reports a probability, but any yes/no call needs a cut, and a bare 0.5
is an arbitrary one. `data.target_precision` in `configs/config.yaml` tunes it:
the pipeline picks the threshold that **maximizes recall subject to precision >=
the target**.

The threshold is chosen on a validation slice carved from the *end of the
training range* -- never on the test set, since tuning on the data you report is
leakage even when the model itself was fit honestly. The slice spans four
calendar years (`validation_years`) rather than the last Games alone: the most
recent Games inside the training range is 2010, which is Winter-only and just
4.4k rows, and a threshold tuned on Winter transfers badly to a Summer-dominated
holdout. A four-year window covers 2008 + 2010 -- 18,004 rows, both seasons.

`rf_deep` on the 2012-2016 holdout (also regenerated by
`python -m scripts.report_numbers`):

| target precision | threshold | precision | recall | F1 |
|---|---|---|---|---|
| none (0.5 cut) | 0.5000 | 0.5395 | 0.2817 | 0.3702 |
| 0.5 | 0.4369 | 0.4821 | 0.3635 | **0.4145** |
| **0.6** (default) | 0.5406 | **0.5699** | 0.2379 | 0.3357 |
| 0.7 | 0.6883 | 0.6448 | 0.1206 | 0.2032 |

**The target is approximate, not a guarantee.** Precision on the validation
slice hits the target almost exactly, but transfer to the holdout costs a few
points -- asking for 0.70 delivers 0.645. The gap is drift between Olympiads,
and it widens the harder you push. Treat the target as a dial, not a contract.

Note `pr_auc` and `roc_auc` do not appear above: they are threshold-free and
identical across every row. Moving the threshold cannot make the model rank
better, only trade precision against recall along the curve it already has.

### Experiment tracking evidence

The full history of all 6 logged runs is committed to the repository in three
forms, so it is inspectable without retraining:

- [`reports/experiments.md`](reports/experiments.md) — human-readable table of
  every run, generated with `mlflow.search_runs()` via
  `python -m src.compare_runs --export reports`
- [`reports/experiments.csv`](reports/experiments.csv) — every logged
  hyperparameter and metric for every run
- [`mlflow.db`](mlflow.db) — the SQLite MLflow tracking store itself; browse
  the runs with `mlflow ui --backend-store-uri sqlite:///mlflow.db` (the
  heavyweight per-run model artifacts in `mlruns/` stay untracked, per the
  no-model-files-in-Git rule)

## Testing

```bash
pytest tests/ -v
```

- `tests/test_preprocess.py` (6 tests): missing-value imputation, categorical
  encoding incl. unseen categories, scaling, target construction, grouped
  split, and input immutability.
- `tests/test_model.py` (2 tests): prediction type/shape/probability sanity and
  a minimum ROC AUC threshold, trained on the committed `data/sample.csv`
  (10,000-row sample) so CI needs no full dataset or API key.
- `tests/test_interface.py` (6 tests): feature extraction and normalization
  with an injected fake LLM, clarifying questions for incomplete input,
  out-of-scope refusal, and malformed-LLM-output recovery.

The same command runs on every push in GitHub Actions
([workflow](.github/workflows/ci.yml) — see the CI badge at the top of this
README for the latest result). Output of `pytest tests/ -v` on a clean run:

<details>
<summary>pytest tests/ -v — 14 passed</summary>

```text
============================= test session starts =============================
platform win32 -- Python 3.13.9, pytest-9.1.1, pluggy-1.6.0
rootdir: olympic-medal-predictor
configfile: pytest.ini
collected 14 items

tests/test_interface.py::test_parse_query_extracts_and_normalizes_features PASSED [  7%]
tests/test_interface.py::test_incomplete_input_triggers_clarifying_question PASSED [ 14%]
tests/test_interface.py::test_out_of_scope_query_is_declined PASSED      [ 21%]
tests/test_interface.py::test_garbage_llm_output_handled_gracefully PASSED [ 28%]
tests/test_interface.py::test_extract_json_tolerates_code_fences PASSED  [ 35%]
tests/test_interface.py::test_normalize_rejects_invalid_values PASSED    [ 42%]
tests/test_model.py::test_predictions_have_correct_type_and_shape PASSED [ 50%]
tests/test_model.py::test_model_meets_minimum_performance PASSED         [ 57%]
tests/test_preprocess.py::test_clean_builds_binary_target PASSED         [ 64%]
tests/test_preprocess.py::test_impute_fills_all_missing_values PASSED    [ 71%]
tests/test_preprocess.py::test_encode_categoricals_maps_and_handles_unseen PASSED [ 78%]
tests/test_preprocess.py::test_scale_features_standardizes PASSED        [ 85%]
tests/test_preprocess.py::test_functions_do_not_mutate_input PASSED      [ 92%]
tests/test_preprocess.py::test_prepare_datasets_keeps_athletes_separate PASSED [100%]

============================= 14 passed in 9.70s ==============================
```

</details>

## Demo script

1. **Happy path:** "I'm a 24-year-old female swimmer from the USA, 175 cm and
   63 kg — what are my medal chances at the 2016 games?"
2. **Incomplete input:** "Will my friend win a medal in judo?" → the app asks
   for the missing sex, age, and country instead of guessing.
3. **Out of scope:** "What's the capital of France?" → the app explains what
   it can and cannot answer.

## Reflection

**What I learned:** how to make an LLM a *disciplined* interface to a model —
strict JSON extraction with normalization and validation between the LLM and
the model is what keeps garbage out of `predict()`. Also, target-encoding
high-cardinality categoricals without leaking requires the same fit/transform
discipline as any scaler.

**What was challenging:** edge cases dominate the interface work (unit
conversions, unseen countries/sports, missing fields, off-topic questions),
and testing LLM-dependent code — solved by injecting the LLM callable so the
pipeline logic is testable offline.

**With more time:** per-event models (medal odds differ hugely between the
100m final and team handball), calibration curves for the reported
probabilities, entity resolution against the historical athlete list ("how did
Michael Phelps do?"), and deploying the container to a cloud service with
scheduled retraining.
