"""Hypothesis testing for A/B analysis.

Implements the decision tree from the "A/B Testing in Practice" workflow
(Kaggle: babyoda/a-b-testing-in-practice):

1. Test the normality assumption with the Shapiro-Wilk test on both arms.
   If both arms look normal (p > alpha), continue with a t-test; otherwise
   fall back to the distribution-free Mann-Whitney U test.
2. For the t-test, test homogeneity of variances with Levene's test:
   homoscedastic arms use the equal-variance (independent) t-test, the rest
   use Welch's t-test.

SciPy is imported lazily so the rest of the package stays importable in the
lean runtime environment; ``ab_test`` itself requires ``requirements-train.txt``.
"""

from __future__ import annotations

import numpy as np

SHAPIRO_MAX_N = 5000


def _summary(values: np.ndarray) -> dict:
    return {
        "n": int(values.size),
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std(ddof=1)), 6) if values.size > 1 else 0.0,
        "median": round(float(np.median(values)), 6),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
    }


def _shapiro_p(values: np.ndarray, alpha: float) -> float:
    """Shapiro-Wilk p-value; deterministic subsample above ``SHAPIRO_MAX_N``."""
    if np.unique(values).size == 1:
        return 1.0  # constant arm: degenerate, treated as normal
    if values.size > SHAPIRO_MAX_N:
        rng = np.random.default_rng(0)
        values = rng.choice(values, size=SHAPIRO_MAX_N, replace=False)
    from scipy import stats  # noqa: PLC0415 - lazy: lean envs have no scipy

    return float(stats.shapiro(values).pvalue)


def ab_test(control, treatment, alpha: float = 0.05) -> dict:
    """Normality-gated A/B test on two per-session metric samples.

    Returns a report with summary stats, the Shapiro/Levene assumption
    results, the selected test, its statistic and p-value, and a boolean
    ``reject_null`` for H0: "no statistically significant difference between
    the control and treatment groups".
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    if control.size < 2 or treatment.size < 2:
        raise ValueError("ab_test needs at least 2 observations per arm")
    try:
        from scipy import stats  # noqa: PLC0415 - lazy import
    except ImportError as exc:
        raise ImportError(
            "ab_test requires scipy; install requirements-train.txt"
        ) from exc

    report: dict = {
        "alpha": alpha,
        "h0": "no statistically significant difference between control and treatment",
        "control": _summary(control),
        "treatment": _summary(treatment),
    }

    if np.unique(np.concatenate([control, treatment])).size == 1:
        report["normality"] = {"test": "shapiro_wilk", "both_normal": None}
        report["homogeneity"] = {"test": None}
        report.update(
            {"test": "constant_groups", "statistic": 0.0, "p_value": 1.0, "reject_null": False}
        )
        return report

    p_c = _shapiro_p(control, alpha)
    p_t = _shapiro_p(treatment, alpha)
    both_normal = bool(p_c > alpha and p_t > alpha)
    report["normality"] = {
        "test": "shapiro_wilk",
        "p_control": round(p_c, 6),
        "p_treatment": round(p_t, 6),
        "both_normal": both_normal,
    }

    if both_normal:
        levene_p = float(stats.levene(control, treatment).pvalue)
        equal_var = bool(levene_p > alpha)
        report["homogeneity"] = {
            "test": "levene",
            "p_value": round(levene_p, 6),
            "equal_var": equal_var,
        }
        statistic, p_value = stats.ttest_ind(control, treatment, equal_var=equal_var)
        method = "independent_t_test" if equal_var else "welch_t_test"
    else:
        report["homogeneity"] = {
            "test": None,
            "reason": "non-normal arms; t-test assumptions do not hold",
        }
        statistic, p_value = stats.mannwhitneyu(control, treatment, alternative="two-sided")
        method = "mann_whitney_u"

    report["test"] = method
    report["statistic"] = round(float(statistic), 6)
    report["p_value"] = round(float(p_value), 8)
    report["reject_null"] = bool(p_value < alpha)
    return report
