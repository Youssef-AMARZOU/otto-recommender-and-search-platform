"""Monitor offline metrics for drift between outcome windows (M6).

Splits ``reports/session_outcomes.parquet`` into a reference window (first
half, sorted by session id) and a current window (second half), renders the
drift report and applies the drift gate. Exits non-zero when the gate fails
so it can run as a scheduled check.

Usage (from the project root, venv active)::

    PYTHONPATH=src python scripts/monitor_drift.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from otto_rec.monitoring.drift import drift_gate, generate_drift_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Drift gate over session outcomes")
    parser.add_argument("--outcomes", default="reports/session_outcomes.parquet")
    parser.add_argument("--report", default="reports/drift.html")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    frame = pd.read_parquet(args.outcomes).sort_values("session_code")
    # identifiers are sorted on, so they would "drift" by construction
    metrics = frame.drop(columns=["session_code"])
    midpoint = len(metrics) // 2
    if midpoint < 20:
        raise SystemExit(f"not enough outcome rows to split: {len(frame)}")
    reference = metrics.iloc[:midpoint].reset_index(drop=True)
    current = metrics.iloc[midpoint:].reset_index(drop=True)

    report = generate_drift_report(reference, current, args.report)
    summary_path = Path(args.report).with_suffix(".json")
    passed = drift_gate(summary_path, threshold=args.threshold)
    print(f"[drift] report -> {report}")
    print(f"[drift] gate {'PASS' if passed else 'FAIL'} (threshold={args.threshold})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
