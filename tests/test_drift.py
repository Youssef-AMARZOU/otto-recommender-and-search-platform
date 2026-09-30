"""Tests for the M6 drift module (evidently backend + K-S fallback)."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

_HAS_PANDAS = importlib.util.find_spec("pandas") is not None
_HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(_HAS_PANDAS and _HAS_NUMPY, "pandas/numpy not installed")
class TestDrift(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np
        import pandas as pd

        self.tmp = Path(tempfile.mkdtemp(prefix="drift_test_"))
        rng = np.random.default_rng(0)
        self.reference = pd.DataFrame(
            {"a": rng.normal(0, 1, 300), "b": rng.normal(5, 1, 300), "tag": ["x"] * 300}
        )
        self.current = pd.DataFrame(
            {"a": rng.normal(0.8, 1, 300), "b": rng.normal(5, 1, 300), "tag": ["y"] * 300}
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_report_and_gate_evidently(self) -> None:
        from otto_rec.monitoring.drift import drift_gate, generate_drift_report

        report = generate_drift_report(
            self.reference, self.current, str(self.tmp / "drift.html")
        )
        self.assertTrue(Path(report).exists())
        summary = json.loads((self.tmp / "drift.json").read_text())
        self.assertIn("drift_share", summary)
        self.assertEqual(summary["backend"], "evidently")
        # column a shifted, column b identical -> 50% drifted
        self.assertAlmostEqual(summary["drift_share"], 0.5)
        self.assertTrue(summary["columns"]["a"]["drift_detected"])
        self.assertFalse(summary["columns"]["b"]["drift_detected"])
        self.assertTrue(drift_gate(summary, threshold=0.5))
        self.assertFalse(drift_gate(summary, threshold=0.3))
        self.assertTrue(drift_gate(str(self.tmp / "drift.json"), threshold=0.5))

    def test_fallback_backend(self) -> None:
        import otto_rec.monitoring.drift as drift_module
        from otto_rec.monitoring.drift import generate_drift_report

        class Boom:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("evidently disabled in test")

        original_report, original_preset = (
            getattr(drift_module, "_report_cls", None),
            getattr(drift_module, "_preset_cls", None),
        )
        drift_module._report_cls = Boom
        drift_module._preset_cls = Boom
        try:
            report = generate_drift_report(
                self.reference, self.current, str(self.tmp / "fb.html")
            )
            self.assertTrue(Path(report).exists())
            summary = json.loads((self.tmp / "fb.json").read_text())
            self.assertEqual(summary["backend"], "kstest")
            self.assertTrue(summary["columns"]["a"]["drift_detected"])
            self.assertFalse(summary["columns"]["b"]["drift_detected"])
        finally:
            drift_module._report_cls = original_report
            drift_module._preset_cls = original_preset

    def test_gate_fail_closed(self) -> None:
        from otto_rec.monitoring.drift import drift_gate

        self.assertFalse(drift_gate({}))
        self.assertFalse(drift_gate({"drift_share": "not-a-number"}))
        self.assertFalse(drift_gate(str(self.tmp / "does-not-exist.json")))
        self.assertTrue(drift_gate({"drift_share": 0.1}, threshold=0.3))

    def test_rejects_bad_input_types(self) -> None:
        from otto_rec.monitoring.drift import generate_drift_report

        with self.assertRaises(TypeError):
            generate_drift_report(42, self.current, str(self.tmp / "x.html"))


if __name__ == "__main__":
    unittest.main()
