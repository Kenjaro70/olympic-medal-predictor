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

from src.evaluate import evaluate_model, feature_importances
from src.preprocess import FEATURE_COLUMNS, features_target, load_raw, prepare_datasets

MODEL_BUILDERS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}

# Selection metric: robust to the ~15% medal-rate class imbalance.
SELECTION_METRIC = "roc_auc"


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
    train, test, artifacts = prepare_datasets(
        raw, data_cfg["test_size"], data_cfg["random_state"]
    )
    X_train, y_train = features_target(train)
    X_test, y_test = features_target(test)
    print(
        f"Data: {len(raw)} raw rows -> {len(train)} train / {len(test)} test "
        f"(grouped by athlete id), medal rate {y_train.mean():.3f}"
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
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)

            mlflow.log_param("model_type", exp["model"])
            mlflow.log_params(exp.get("params") or {})
            mlflow.log_param("features", ",".join(FEATURE_COLUMNS))
            mlflow.log_param("data_description", data_cfg["description"])
            mlflow.log_param("test_size", data_cfg["test_size"])
            mlflow.log_param("random_state", data_cfg["random_state"])
            mlflow.log_param("n_train_rows", len(X_train))
            mlflow.log_param("n_test_rows", len(X_test))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, name="model")

            print(f"{exp['name']}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            candidate = {
                "name": exp["name"],
                "model_type": exp["model"],
                "params": exp.get("params") or {},
                "metrics": metrics,
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
