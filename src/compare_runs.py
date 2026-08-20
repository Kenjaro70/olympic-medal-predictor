"""Compare MLflow runs programmatically and identify the best one.

Usage:
    python -m src.compare_runs --config configs/config.yaml
"""

from __future__ import annotations

import argparse

import mlflow
import yaml

DISPLAY_COLUMNS = [
    "tags.mlflow.runName",
    "params.model_type",
    "metrics.accuracy",
    "metrics.precision",
    "metrics.recall",
    "metrics.f1",
    "metrics.roc_auc",
]


def compare(config_path: str, metric: str = "roc_auc") -> None:
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    runs = mlflow.search_runs(
        experiment_names=[config["mlflow"]["experiment_name"]],
        order_by=[f"metrics.{metric} DESC"],
    )
    if runs.empty:
        print("No runs logged yet — run `python -m src.train` first.")
        return

    cols = [c for c in DISPLAY_COLUMNS if c in runs.columns]
    table = runs[cols].rename(columns=lambda c: c.split(".", 1)[1])
    print(f"All runs, ranked by {metric}:\n")
    print(table.to_string(index=False))

    best = runs.iloc[0]
    print(
        f"\nBest run: {best['tags.mlflow.runName']} "
        f"(run_id={best['run_id']}) with {metric}="
        f"{best[f'metrics.{metric}']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--metric", default="roc_auc")
    args = parser.parse_args()
    compare(args.config, args.metric)


if __name__ == "__main__":
    main()
