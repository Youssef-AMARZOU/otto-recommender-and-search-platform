"""Sample-size calculation for a two-proportion A/B test.

Standard two-sided z-test with pooled variance: run it before the experiment
to confirm the planned session count can detect the minimum detectable effect
at the chosen alpha and power.
"""

from __future__ import annotations

from math import ceil, sqrt
from statistics import NormalDist


def required_sample_size(
    baseline: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Sessions needed per arm.

    ``baseline`` is the control conversion rate (e.g. CTR),
    ``mde_relative`` is the minimum detectable relative lift (e.g. 0.05 for +5%).
    """
    if not 0.0 < baseline < 1.0:
        raise ValueError("baseline must be a rate in (0, 1)")
    if mde_relative <= 0.0:
        raise ValueError("mde_relative must be positive")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")

    p1 = baseline
    p2 = baseline * (1.0 + mde_relative)
    if p2 >= 1.0:
        raise ValueError("baseline * (1 + mde_relative) must stay below 1")

    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(power)
    p_bar = (p1 + p2) / 2.0

    numerator = (
        z_alpha * sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_power * sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    denominator = (p2 - p1) ** 2
    return ceil(numerator / denominator)
