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

## Results

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
