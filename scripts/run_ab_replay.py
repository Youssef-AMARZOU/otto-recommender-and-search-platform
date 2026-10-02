"""Simulated A/B test on per-session offline outcomes (M4 experimentation).

Sessions are deterministically split into control/treatment with the
assignment module; control sees the champion system (co-visitation) and
treatment the challenger (two-stage). The script reports:

- power analysis: sessions per arm required to detect ``--mde`` at alpha 0.05
- observed effect on the primary metric (default: clicks hit-rate@20),
  raw and CUPED-adjusted with session length as covariate
- assumption-checked hypothesis test (Shapiro-Wilk normality -> independent
  t-test / Mann-Whitney U, Levene homogeneity for the t-test variant), the
  decision follows its p-value when scipy is available
- guardrail: two-stage single-row scoring latency p99 vs. threshold
- a ship / hold / inconclusive recommendation from those inputs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from otto_rec.experimentation.assignment import assign_arm
from otto_rec.experimentation.cuped import cuped
from otto_rec.experimentation.power import required_sample_size

NORMAL = NormalDist()


def two_proportion_ztest(control: np.ndarray, treatment: np.ndarray) -> tuple[float, float]:
    """Two-sided pooled z-test for two binary samples; returns (z, p)."""
    n_c, n_t = len(control), len(treatment)
    p_c, p_t = float(control.mean()), float(treatment.mean())
    p_pool = (control.sum() + treatment.sum()) / (n_c + n_t)
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_c + 1.0 / n_t))
    if se == 0.0:
        return 0.0, 1.0
    z = (p_t - p_c) / se
    return float(z), float(2.0 * (1.0 - NORMAL.cdf(abs(z))))


def welch_mean_test(control: np.ndarray, treatment: np.ndarray) -> tuple[float, float]:
    """Two-sided Welch z-test on means for continuous samples (CUPED-adjusted)."""
    n_c, n_t = len(control), len(treatment)
    se = np.sqrt(float(control.var(ddof=1)) / n_c + float(treatment.var(ddof=1)) / n_t)
    if se == 0.0:
        return 0.0, 1.0
    z = (float(treatment.mean()) - float(control.mean())) / se
    return float(z), float(2.0 * (1.0 - NORMAL.cdf(abs(z))))


def run_ab_replay(
    outcomes_path: Path,
    eval_report_path: Path,
    report_path: Path,
    control: str = "covisitation",
    treatment: str = "two_stage",
    metric_type: str = "clicks",
    mde_relative: float = 0.05,
    treatment_fraction: float = 0.5,
    salt: str = "otto-ab-v1",
    latency_threshold_ms: float = 50.0,
) -> dict:
    started = time.perf_counter()
    outcomes = pd.read_parquet(outcomes_path)
    column = f"hit20_{control}_{metric_type}"
    treatment_column = f"hit20_{treatment}_{metric_type}"
    if column not in outcomes.columns or treatment_column not in outcomes.columns:
        raise KeyError(f"expected columns {column} and {treatment_column} in {outcomes_path}")

    outcomes["arm"] = [
        assign_arm(str(int(sid)), treatment_fraction=treatment_fraction, salt=salt)
        for sid in outcomes["session_code"]
    ]

    control_mask = outcomes["arm"] == "control"
    treatment_mask = outcomes["arm"] == "treatment"
    y_control = outcomes.loc[control_mask, column].to_numpy(dtype=float)
    y_treatment = outcomes.loc[treatment_mask, treatment_column].to_numpy(dtype=float)
    baseline = float(y_control.mean())

    power = {"baseline": round(baseline, 5), "mde_relative": mde_relative}
    if 0.0 < baseline < 1.0:
        required = required_sample_size(baseline, mde_relative)
        power["required_per_arm"] = required
        power["observed_per_arm"] = int(treatment_mask.sum())
        power["adequately_powered"] = bool(int(treatment_mask.sum()) >= required)
    else:
        power["required_per_arm"] = None
        power["adequately_powered"] = False

    observed_c = float(y_control.mean())
    observed_t = float(y_treatment.mean())
    z_raw, p_raw = two_proportion_ztest(y_control, y_treatment)

    y_all = np.concatenate([y_control, y_treatment])
    covariate = pd.concat([outcomes.loc[control_mask, "n_events"], outcomes.loc[treatment_mask, "n_events"]]).to_numpy(dtype=float)
    y_cuped = np.asarray(cuped(list(y_all), list(covariate)))
    cuped_c = y_cuped[: len(y_control)]
    cuped_t = y_cuped[len(y_control):]
    z_cuped, p_cuped = welch_mean_test(cuped_c, cuped_t)

    try:
        from otto_rec.experimentation.hypothesis import ab_test

        hypothesis = {
            "raw": ab_test(y_control, y_treatment),
            "cuped": ab_test(cuped_c, cuped_t),
        }
        decision_p = hypothesis["cuped"]["p_value"]
    except ImportError:
        hypothesis = {
            "skipped": "scipy not installed; falling back to the analytic z-tests"
        }
        decision_p = p_cuped

    eval_report = json.loads(eval_report_path.read_text())
    latency_p99 = eval_report.get("latency_ms", {}).get("p99")
    guardrail_pass = latency_p99 is not None and latency_p99 <= latency_threshold_ms

    lift = (observed_t - observed_c) / observed_c if observed_c > 0 else None
    cuped_lift = (cuped_t.mean() - cuped_c.mean()) / cuped_c.mean() if cuped_c.mean() > 0 else None

    if not guardrail_pass:
        recommendation = "hold"
    elif decision_p < 0.05 and cuped_lift is not None and cuped_lift > 0:
        recommendation = "ship"
    elif decision_p < 0.05 and cuped_lift is not None and cuped_lift <= 0:
        recommendation = "hold"
    else:
        recommendation = "inconclusive"

    report = {
        "config": {
            "control": control,
            "treatment": treatment,
            "primary_metric": f"hit20 (ground-truth type: {metric_type})",
            "treatment_fraction": treatment_fraction,
            "salt": salt,
            "n_sessions": int(len(outcomes)),
        },
        "power": power,
        "effect": {
            "control_rate": round(observed_c, 5),
            "treatment_rate": round(observed_t, 5),
            "relative_lift": round(lift, 5) if lift is not None else None,
            "z": round(z_raw, 4),
            "p_value": round(p_raw, 5),
            "cuped": {
                "control_rate": round(float(cuped_c.mean()), 5),
                "treatment_rate": round(float(cuped_t.mean()), 5),
                "relative_lift": round(cuped_lift, 5) if cuped_lift is not None else None,
                "z": round(z_cuped, 4),
                "p_value": round(p_cuped, 5),
            },
            "hypothesis": hypothesis,
        },
        "guardrails": {
            "scoring_latency_p99_ms": latency_p99,
            "threshold_ms": latency_threshold_ms,
            "pass": bool(guardrail_pass),
        },
        "recommendation": recommendation,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B replay simulation on offline session outcomes")
    parser.add_argument("--outcomes", default="reports/session_outcomes.parquet")
    parser.add_argument("--eval-report", default="reports/offline_eval.json")
    parser.add_argument("--report", default="reports/ab_replay.json")
    parser.add_argument("--control", default="covisitation")
    parser.add_argument("--treatment", default="two_stage")
    parser.add_argument("--metric-type", default="clicks", choices=["clicks", "carts", "orders"])
    parser.add_argument("--mde", type=float, default=0.05)
    args = parser.parse_args()
    run_ab_replay(
        Path(args.outcomes),
        Path(args.eval_report),
        Path(args.report),
        control=args.control,
        treatment=args.treatment,
        metric_type=args.metric_type,
        mde_relative=args.mde,
    )
