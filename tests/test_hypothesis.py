"""Tests for the normality-gated A/B hypothesis test (scipy-gated)."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

_HAS_SCIPY = importlib.util.find_spec("scipy") is not None


@unittest.skipUnless(_HAS_SCIPY, "scipy required")
class AbTestTest(unittest.TestCase):
    def test_normal_shift_selects_t_test_and_rejects(self) -> None:
        from otto_rec.experimentation.hypothesis import ab_test

        rng = np.random.default_rng(0)
        report = ab_test(rng.normal(0.0, 1.0, 400), rng.normal(0.3, 1.0, 400))
        self.assertTrue(report["normality"]["both_normal"])
        self.assertIn(report["test"], {"independent_t_test", "welch_t_test"})
        self.assertTrue(report["reject_null"])
        self.assertLess(report["p_value"], 0.05)

    def test_skewed_data_selects_mann_whitney(self) -> None:
        from otto_rec.experimentation.hypothesis import ab_test

        rng = np.random.default_rng(1)
        report = ab_test(
            rng.lognormal(0.0, 1.2, 500), rng.lognormal(0.3, 1.2, 500)
        )
        self.assertFalse(report["normality"]["both_normal"])
        self.assertEqual(report["test"], "mann_whitney_u")
        self.assertTrue(report["reject_null"])

    def test_no_effect_not_rejected(self) -> None:
        from otto_rec.experimentation.hypothesis import ab_test

        rng = np.random.default_rng(2)
        report = ab_test(rng.normal(0.0, 1.0, 300), rng.normal(0.0, 1.0, 300))
        self.assertFalse(report["reject_null"])

    def test_binary_metric_takes_rank_test_path(self) -> None:
        from otto_rec.experimentation.hypothesis import ab_test

        rng = np.random.default_rng(3)
        report = ab_test(
            rng.binomial(1, 0.21, 5000), rng.binomial(1, 0.21, 5000)
        )
        self.assertFalse(report["normality"]["both_normal"])
        self.assertEqual(report["test"], "mann_whitney_u")
        self.assertFalse(report["reject_null"])

    def test_constant_groups_guard(self) -> None:
        from otto_rec.experimentation.hypothesis import ab_test

        same = ab_test(np.zeros(50), np.zeros(50))
        self.assertEqual(same["test"], "constant_groups")
        self.assertFalse(same["reject_null"])
        diff = ab_test(np.zeros(50), np.ones(50))
        self.assertTrue(diff["reject_null"])

    def test_rejects_small_samples_and_missing_scipy_contract(self) -> None:
        from otto_rec.experimentation.hypothesis import ab_test

        with self.assertRaises(ValueError):
            ab_test(np.zeros(1), np.zeros(1))


if __name__ == "__main__":
    unittest.main()
