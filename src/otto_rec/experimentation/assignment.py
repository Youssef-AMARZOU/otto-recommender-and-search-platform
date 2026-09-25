"""Deterministic A/B arm assignment for the traffic-replay harness.

Assignment hashes the session id with an experiment salt so the same session
always lands in the same arm across runs, which makes simulated traffic splits
reproducible without storing an assignment table.
"""

from __future__ import annotations

import hashlib


def hash_bucket(entity_id: str, salt: str = "", buckets: int = 1000) -> int:
    """Map an entity id to a stable bucket in [0, buckets)."""
    if buckets <= 0:
        raise ValueError("buckets must be a positive integer")
    digest = hashlib.sha256(f"{salt}:{entity_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def assign_arm(
    entity_id: str,
    treatment_fraction: float = 0.5,
    salt: str = "",
    control: str = "control",
    treatment: str = "treatment",
    buckets: int = 1000,
) -> str:
    """Route an entity to ``treatment`` with probability ``treatment_fraction``."""
    if not 0.0 <= treatment_fraction <= 1.0:
        raise ValueError("treatment_fraction must be in [0, 1]")
    threshold = treatment_fraction * buckets
    return treatment if hash_bucket(entity_id, salt, buckets) < threshold else control
