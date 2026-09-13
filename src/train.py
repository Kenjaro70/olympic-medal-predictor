"""Train Olympic medal prediction models with MLflow experiment tracking.

Usage:
    python -m src.train --config configs/config.yaml
    python -m src.train --config configs/config.yaml --experiment rf_deep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluate import (
    DEFAULT_THRESHOLD,
    choose_threshold,
    evaluate_model,
    feature_importances,
)
from src.preprocess import FEATURE_COLUMNS, features_target, load_raw, prepare_datasets

MODEL_BUILDERS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}

# Selection metric: average precision, scored against the ~15% base rate.
# ROC-AUC stays high under this much imbalance even when precision is poor,
# so it is still logged but no longer decides which model wins.
SELECTION_METRIC = "pr_auc"


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def build_model(model_type: str, params: dict, random_state: int):
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model type: {model_type}")
    params = dict(params or {})
    if model_type in ("random_forest", "hist_gradient_boosting"):
        params.setdefault("random_state", random_state)
    return MODEL_BUILDERS[model_type](**params)


def run_experiments(config: dict, only: str | None = None) -> dict:
    data_cfg = config["data"]
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    raw = load_raw(data_cfg["raw_path"])
    split_strategy = data_cfg.get("split_strategy", "temporal")
    holdout_from = data_cfg.get("holdout_from", 2012)
    train, test, artifacts = prepare_datasets(
        raw,
        data_cfg["test_size"],
        data_cfg["random_state"],
        split_strategy=split_strategy,
        holdout_from=holdout_from,
    )
    # Carve the most recent Games in train as a threshold-tuning slice. The
    # threshold must not be chosen on the test set, and using the latest
    # in-train Games mimics the real gap between fitting and deploying.
    target_precision = data_cfg.get("target_precision")
    if target_precision is not None and split_strategy == "temporal":
        # Span a full Olympiad, not just the last Games. The most recent
        # Games inside the training range is 2010 -- Winter only, 4.4k rows
        # -- and a threshold tuned on Winter alone transfers badly to a
        # Summer-dominated holdout. Four years captures one of each.
        window = int(data_cfg.get("validation_years", 4))
        # year_raw, not year -- the latter is standardized by this point.
        val_from = int(train["year_raw"].max()) - window + 1
        fit_mask = train["year_raw"] < val_from
        fit_part, val_part = train[fit_mask], train[~fit_mask]
        seasons = sorted(val_part["season"].unique())
        print(
            f"Threshold tuning: fit on {len(fit_part)} rows (< {val_from}), "
            f"validate on {len(val_part)} rows ({val_from}-"
            f"{int(train['year_raw'].max())}, "
            f"{len(sorted(val_part['year_raw'].unique()))} Games, "
            f"{len(seasons)} season(s))"
        )
        if val_part.empty:
            raise ValueError(
                f"validation_years={window} leaves no rows to tune the "
                "threshold on"
            )
    else:
        fit_part = val_part = None
        val_from = None

    X_train, y_train = features_target(train)
    X_test, y_test = features_target(test)
    split_desc = (
        f"temporal, holdout from {holdout_from}"
        if split_strategy == "temporal"
        else "grouped by athlete id"
    )
    print(
        f"Data: {len(raw)} raw rows -> {len(train)} train / {len(test)} test "
        f"({split_desc}), train medal rate {y_train.mean():.3f}, "
        f"test medal rate {y_test.mean():.3f}"
    )

    experiments = config["experiments"]
    if only:
        experiments = [e for e in experiments if e["name"] == only]
        if not experiments:
            raise ValueError(f"No experiment named '{only}' in config")

    best = None
    for exp in experiments:
        with mlflow.start_run(run_name=exp["name"]) as run:
            model = build_model(
                exp["model"], exp.get("params"), data_cfg["random_state"]
            )

            threshold, threshold_info = DEFAULT_THRESHOLD, {}
            if val_part is not None:
                # Fit on the earlier slice only, so the validation Games are
                # genuinely unseen when the threshold is picked.
                probe = build_model(
                    exp["model"], exp.get("params"), data_cfg["random_state"]
                )
                Xf, yf = features_target(fit_part)
                Xv, yv = features_target(val_part)
                probe.fit(Xf, yf)
                threshold, threshold_info = choose_threshold(
                    yv, probe.predict_proba(Xv)[:, 1], target_precision
                )

            # Final model sees all training rows, including the validation
            # Games; only the threshold came from the probe.
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, threshold=threshold)

            mlflow.log_param("model_type", exp["model"])
            mlflow.log_params(exp.get("params") or {})
            mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
            mlflow.log_param("data_description", data_cfg["description"])
            mlflow.log_param("test_size", data_cfg["test_size"])
            mlflow.log_param("random_state", data_cfg["random_state"])
            mlflow.log_param("n_train_rows", len(X_train))
            mlflow.log_param("n_test_rows", len(X_test))
            mlflow.log_param("split_strategy", split_strategy)
            mlflow.log_param("threshold_tuned_on", val_from)
            for key, value in threshold_info.items():
                mlflow.log_param(f"threshold_{key}", value)
            mlflow.log_param("holdout_from", holdout_from if split_strategy == "temporal" else None)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="model")

            print(f"{exp['name']}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            candidate = {
                "name": exp["name"],
                "model_type": exp["model"],
                "params": exp.get("params") or {},
                "metrics": metrics,
                "threshold": threshold,
                "threshold_info": threshold_info,
                "run_id": run.info.run_id,
                "model": model,
            }
            if best is None or metrics[SELECTION_METRIC] > best["metrics"][SELECTION_METRIC]:
                best = candidate

    save_best(best, artifacts, config)
    return best


def save_best(best: dict, artifacts: dict, config: dict) -> None:
    """Persist the winning model plus preprocessing state for the app."""
    model_dir = Path(config["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": best["model"],
        "preprocessing": artifacts,
        "feature_columns": FEATURE_COLUMNS,
        "metadata": {
            "experiment_name": best["name"],
            "model_type": best["model_type"],
            "params": best["params"],
            "metrics": best["metrics"],
            "run_id": best["run_id"],
            "selection_metric": SELECTION_METRIC,
            "decision_threshold": best.get("threshold", 0.5),
            "threshold_info": best.get("threshold_info", {}),
            "feature_importances": feature_importances(
                best["model"], FEATURE_COLUMNS
            ),
        },
    }
    joblib.dump(bundle, model_dir / "model_bundle.joblib")
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(bundle["metadata"], fh, indent=2)
    print(
        f"\nBest model: {best['name']} ({best['model_type']}) "
        f"{SELECTION_METRIC}={best['metrics'][SELECTION_METRIC]:.4f} "
        f"-> {model_dir / 'model_bundle.joblib'}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--experiment", default=None, help="Run a single named experiment"
    )
    args = parser.parse_args()
    run_experiments(load_config(args.config), args.experiment)


if __name__ == "__main__":
    main()
