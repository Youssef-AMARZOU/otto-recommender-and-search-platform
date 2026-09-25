"""Feature and prediction drift monitoring with Evidently (M6).

Compares a reference window (training period) against the current serving
window and gates the pipeline when drift exceeds thresholds.
"""

from __future__ import annotations


def generate_drift_report(reference, current, report_path: str = "reports/drift.html") -> str:
    """Build an Evidently drift report over shared feature columns."""
    raise NotImplementedError("M6: Evidently drift report")


def drift_gate(drift_summary: dict[str, float], threshold: float = 0.3) -> bool:
    """Return True when every drifted-feature share stays under threshold."""
    raise NotImplementedError("M6: drift gating for the metrics loop")
