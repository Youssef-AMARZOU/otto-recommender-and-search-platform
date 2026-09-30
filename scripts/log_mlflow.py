"""Log the trained ranker and offline results to the MLflow tracking server.

M4 registry hook: mirrors what CI/CD would do after a successful pipeline
run - params and metrics come from ``reports/offline_eval.json`` and
``reports/ab_replay.json``, the model flavor comes from
``mlflow.lightgbm.log_model``.

Usage::

    PYTHONPATH=src python scripts/log_mlflow.py            # log a run
    PYTHONPATH=src python scripts/log_mlflow.py --register # also push to registry

The tracking URI defaults to ``./mlruns`` unless ``MLFLOW_TRACKING_URI`` is
set; the registered model name defaults to ``otto-ranker``. Exits with a
clear message when mlflow is not installed (skeleton environments).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from otto_rec.ranking.train import FEATURE_COLUMNS


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_metrics(eval_report: dict, ab_report: dict) -> dict[str, float]:
    """Flatten the offline/online reports into MLflow scalar metrics."""
    metrics: dict[str, float] = {}
    for name, system in eval_report.get("systems", {}).items():
        macro = system.get("macro", {})
        for key in ("recall@20", "ndcg@20", "mrr@20"):
            if key in macro:
                metrics[f"{name}_{key.replace('@', '_at_')}"] = macro[key]
        if "otto_weighted_recall@20" in system:
            metrics[f"{name}_otto_weighted_recall_at_20"] = system["otto_weighted_recall@20"]
    latency = eval_report.get("latency_ms", {})
    for key in ("p50", "p95", "p99"):
        if key in latency:
            metrics[f"offline_latency_ms_{key}"] = latency[key]
    if ab_report:
        decision = ab_report.get("decision", ab_report)
        for src_key, dst_key in (
            ("relative_lift", "ab_relative_lift"),
            ("p_value", "ab_p_value"),
            ("power", "ab_power"),
        ):
            if src_key in decision and isinstance(decision[src_key], (int, float)):
                metrics[dst_key] = float(decision[src_key])
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Log ranker + eval results to MLflow")
    parser.add_argument("--eval-report", default="reports/offline_eval.json")
    parser.add_argument("--ab-report", default="reports/ab_replay.json")
    parser.add_argument("--model", default="models/ranker.txt")
    parser.add_argument("--run-name", default="otto-m3-offline")
    parser.add_argument("--register", action="store_true", help="push the logged model to the registry")
    parser.add_argument("--registry-name", default="otto-ranker")
    args = parser.parse_args()

    try:
        import mlflow
        import mlflow.lightgbm
    except ImportError:
        print("mlflow is not installed; run: pip install mlflow", file=sys.stderr)
        return 2

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 2

    eval_report = _load_json(Path(args.eval_report))
    ab_report = _load_json(Path(args.ab_report))

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "otto-recommender"))
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "model": str(model_path),
                "model_type": "lightgbm_lambdarank",
                "n_features": len(FEATURE_COLUMNS),
                "features": ",".join(FEATURE_COLUMNS),
                "n_eval_sessions": eval_report.get("config", {}).get("n_eval_sessions", 0),
                "history_rule": eval_report.get("config", {}).get("history_rule", ""),
            }
        )
        metrics = build_metrics(eval_report, ab_report)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path))
        mlflow.lightgbm.log_model(
            model_path,
            artifact_path="model",
            # local booster file references pathlib path objects
            skops_trusted_types=[
                "pathlib.WindowsPath",
                "pathlib.PosixPath",
                "pathlib.PureWindowsPath",
                "pathlib.PurePosixPath",
            ],
        )
        if eval_report:
            mlflow.log_artifact(args.eval_report)
        if ab_report:
            mlflow.log_artifact(args.ab_report)
        run_id = run.info.run_id
        print(f"[mlflow] run {run_id} logged to {mlflow.get_tracking_uri()}")
        if args.register:
            model_uri = f"runs:/{run_id}/model"
            mlflow.register_model(model_uri, args.registry_name)
            print(f"[mlflow] registered {model_uri} as {args.registry_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
