"""CUPED: variance reduction with a pre-experiment covariate.

Adjusted metric: Y_cuped = Y - theta * (X - mean(X)), with
theta = Cov(X, Y) / Var(X). Using pre-period clicks or revenue as X shrinks
the noise in the experiment metric, so the same detectable effect needs fewer
sessions.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean, pvariance


def cuped(values: Sequence[float], covariates: Sequence[float]) -> list:
    """Return CUPED-adjusted metric values (mean preserved, variance reduced)."""
    ys = list(values)
    xs = list(covariates)
    if len(ys) != len(xs):
        raise ValueError("values and covariates must have the same length")
    if len(ys) < 2:
        raise ValueError("need at least two observations")

    var_x = pvariance(xs)
    if var_x == 0.0:
        return ys[:]

    mean_x = fmean(xs)
    mean_y = fmean(ys)
    covariance = fmean((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    theta = covariance / var_x
    return [y - theta * (x - mean_x) for x, y in zip(xs, ys)]
