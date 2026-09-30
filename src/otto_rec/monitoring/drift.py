"""Drift monitoring for offline metrics and serving features (M6).

``generate_drift_report`` compares a reference window against the current
window and writes a self-contained HTML report plus a machine-readable JSON
summary next to it (``drift.json`` when the report is ``drift.html``). The
HTML is rendered by evidently (``DataDriftPreset``) when installed, with a
plain built-in table as fallback.

Column-level ``drift_detected`` (what the gate consumes) always uses our own
measurement: a K-S test must be significant (p < 0.05) **and** the effect
size must matter (PSI >= 0.1), so large evaluation windows do not flag
sub-percentage noise as drift.

``drift_gate`` decides pass/fail on the JSON summary: the share of drifted
columns must stay at or below ``threshold`` (0.3 = at most 30% of monitored
columns may drift). Fail-closed on a missing/unreadable summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_THRESHOLD = 0.05
DEFAULT_PSI = 0.1

# Module hooks so tests (and degraded deployments) can disable the backend.
_report_cls = None
_preset_cls = None


def _evidently_classes():
    global _report_cls, _preset_cls
    if _report_cls is None or _preset_cls is None:
        from evidently import Report  # noqa: PLC0415 - optional dependency
        from evidently.presets import DataDriftPreset  # noqa: PLC0415

        _report_cls, _preset_cls = Report, DataDriftPreset
    return _report_cls, _preset_cls


def _as_frame(data: Any, label: str) -> pd.DataFrame:
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)
    if isinstance(data, pd.DataFrame):
        return data
    raise TypeError(f"{label} must be a DataFrame or path to parquet/csv, got {type(data)}")


def _numeric_columns(reference: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    columns = []
    for name in current.columns:
        if name not in reference.columns:
            continue
        if pd.api.types.is_numeric_dtype(current[name]) and pd.api.types.is_numeric_dtype(
            reference[name]
        ):
            columns.append(str(name))
    return columns


def _ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        return 0.0
    merged = np.sort(np.concatenate([reference, current]))
    cdf_ref = np.searchsorted(np.sort(reference), merged, side="right") / reference.size
    cdf_cur = np.searchsorted(np.sort(current), merged, side="right") / current.size
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index over reference quantile bins."""
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if reference.size == 0 or current.size == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts = np.histogram(reference, bins=edges)[0].astype(np.float64)
    cur_counts = np.histogram(current, bins=edges)[0].astype(np.float64)
    ref_share = np.clip(ref_counts / reference.size, 1e-6, None)
    cur_share = np.clip(cur_counts / current.size, 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def _column_summary(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """K-S + PSI per numeric column with a practical drift decision."""
    try:
        from scipy import stats  # noqa: PLC0415 - optional dependency

        scipy_stats = stats
    except ImportError:  # pragma: no cover - scipy ships with the training env
        scipy_stats = None
    columns: dict[str, dict] = {}
    drifted = 0
    for name in _numeric_columns(reference, current):
        ref = reference[name].to_numpy(dtype=np.float64)
        cur = current[name].to_numpy(dtype=np.float64)
        statistic = _ks_statistic(ref, cur)
        psi = _psi(ref, cur)
        if scipy_stats is not None:
            ref_clean, cur_clean = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
            p_value: float | None = float(scipy_stats.ks_2samp(ref_clean, cur_clean).pvalue)
            significant = p_value < DEFAULT_THRESHOLD
        else:
            p_value = None
            significant = True  # no p-value: decide on effect size alone
        detected = significant and psi >= DEFAULT_PSI
        columns[name] = {
            "ks": round(statistic, 6),
            "psi": round(psi, 6),
            "p_value": p_value,
            "drift_detected": detected,
        }
        drifted += int(detected)
    total = len(columns)
    return {
        "n_columns": total,
        "n_drifted": drifted,
        "drift_share": round(drifted / total, 6) if total else 0.0,
        "columns": columns,
    }


def _fallback_html(summary: dict) -> str:
    rows = "".join(
        f"<tr><td>{name}</td><td>{values.get('ks')}</td><td>{values.get('psi')}</td>"
        f"<td>{values.get('p_value')}</td><td>{values.get('drift_detected')}</td></tr>"
        for name, values in summary["columns"].items()
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Drift report</title></head><body>"
        f"<h1>Drift report ({summary['backend']})</h1>"
        f"<p>drifted {summary['n_drifted']}/{summary['n_columns']} "
        f"(share {summary['drift_share']})</p>"
        "<table border='1'><tr><th>column</th><th>ks</th><th>psi</th>"
        "<th>p_value</th><th>drift</th></tr>"
        f"{rows}</table></body></html>"
    )


def generate_drift_report(
    reference: Any, current: Any, report_path: str = "reports/drift.html"
) -> str:
    """Compare windows, write HTML + JSON summary, return the HTML path."""
    reference_frame = _as_frame(reference, "reference")
    current_frame = _as_frame(current, "current")
    columns = _numeric_columns(reference_frame, current_frame)
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    backend = "kstest"
    try:
        report_cls, preset_cls = _evidently_classes()
        snapshot = report_cls(metrics=[preset_cls()]).run(
            current_data=current_frame[columns],
            reference_data=reference_frame[columns],
        )
        snapshot.save_html(str(path))
        backend = "evidently"
    except Exception as exc:  # noqa: BLE001 - fall back rather than lose the gate
        print(f"[drift] evidently HTML unavailable ({exc}); using built-in report", flush=True)

    summary = _column_summary(reference_frame[columns], current_frame[columns])
    summary["backend"] = backend
    if backend == "kstest":
        path.write_text(_fallback_html(summary), encoding="utf-8")

    summary["report"] = str(path)
    summary["n_reference_rows"] = int(len(reference_frame))
    summary["n_current_rows"] = int(len(current_frame))
    json_path = path.with_suffix(".json") if path.suffix == ".html" else Path(str(path) + ".json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(path)


def drift_gate(drift_summary: Any, threshold: float = 0.3) -> bool:
    """True when the share of drifted columns is within ``threshold``."""
    if isinstance(drift_summary, (str, Path)):
        try:
            drift_summary = json.loads(Path(drift_summary).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
    if not isinstance(drift_summary, dict) or "drift_share" not in drift_summary:
        return False
    try:
        return float(drift_summary["drift_share"]) <= threshold
    except (TypeError, ValueError):
        return False
