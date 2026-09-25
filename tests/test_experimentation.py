import unittest
from statistics import pvariance

from otto_rec.experimentation.assignment import assign_arm, hash_bucket
from otto_rec.experimentation.cuped import cuped
from otto_rec.experimentation.power import required_sample_size


class TestAssignment(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(hash_bucket("session-42", "exp-1"), hash_bucket("session-42", "exp-1"))

    def test_salt_changes_bucket(self):
        buckets = {hash_bucket(f"s{i}", "exp-1") for i in range(500)}
        other = {hash_bucket(f"s{i}", "exp-2") for i in range(500)}
        self.assertNotEqual(buckets, other)

    def test_bucket_range(self):
        for i in range(100):
            bucket = hash_bucket(f"s{i}")
            self.assertTrue(0 <= bucket < 1000)

    def test_fraction_boundaries(self):
        ids = [f"session-{i}" for i in range(2000)]
        for entity_id in ids:
            self.assertEqual(assign_arm(entity_id, treatment_fraction=0.0), "control")
            self.assertEqual(assign_arm(entity_id, treatment_fraction=1.0), "treatment")

    def test_approximate_split(self):
        ids = [f"session-{i}" for i in range(5000)]
        treatment_share = sum(assign_arm(i, treatment_fraction=0.5) == "treatment" for i in ids) / len(ids)
        self.assertLess(abs(treatment_share - 0.5), 0.05)

    def test_rejects_bad_fraction(self):
        with self.assertRaises(ValueError):
            assign_arm("s1", treatment_fraction=1.5)


class TestPower(unittest.TestCase):
    def test_positive_sample_size(self):
        self.assertGreater(required_sample_size(baseline=0.10, mde_relative=0.05), 0)

    def test_larger_effect_needs_fewer_sessions(self):
        big_effect = required_sample_size(baseline=0.10, mde_relative=0.20)
        small_effect = required_sample_size(baseline=0.10, mde_relative=0.02)
        self.assertLess(big_effect, small_effect)

    def test_higher_power_needs_more_sessions(self):
        self.assertGreater(
            required_sample_size(baseline=0.10, mde_relative=0.05, power=0.9),
            required_sample_size(baseline=0.10, mde_relative=0.05, power=0.8),
        )

    def test_stricter_alpha_needs_more_sessions(self):
        self.assertGreater(
            required_sample_size(baseline=0.10, mde_relative=0.05, alpha=0.01),
            required_sample_size(baseline=0.10, mde_relative=0.05, alpha=0.05),
        )

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            required_sample_size(baseline=1.5, mde_relative=0.05)
        with self.assertRaises(ValueError):
            required_sample_size(baseline=0.10, mde_relative=0.0)
        with self.assertRaises(ValueError):
            required_sample_size(baseline=0.95, mde_relative=1.0)


class TestCuped(unittest.TestCase):
    def test_preserves_mean(self):
        covariates = [1.0, 2.0, 3.0, 4.0, 5.0]
        values = [10.0, 12.0, 13.0, 15.0, 17.0]
        adjusted = cuped(values, covariates)
        self.assertAlmostEqual(sum(adjusted) / len(adjusted), sum(values) / len(values), places=9)

    def test_reduces_variance_when_correlated(self):
        covariates = [float(i) for i in range(100)]
        values = [2.0 * x + ((i % 7) - 3) for i, x in enumerate(covariates)]
        adjusted = cuped(values, covariates)
        self.assertLess(pvariance(adjusted), pvariance(values))

    def test_constant_covariate_returns_values(self):
        values = [1.0, 2.0, 3.0]
        self.assertEqual(cuped(values, [5.0, 5.0, 5.0]), values)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            cuped([1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()
