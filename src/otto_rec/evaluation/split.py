"""Temporal train/test splitting — predict tomorrow from today.

Never split randomly: recommendation evaluation must respect session time so
that training data always precedes the evaluated window.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def temporal_split(
    sessions: Sequence[dict[str, Any]],
    train_ratio: float = 0.8,
    time_key: str = "timestamp",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sort sessions by ``time_key`` and cut chronologically.

    Returns ``(train_sessions, test_sessions)`` where every training session
    starts no later than every test session.
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1")
    if sessions and time_key not in sessions[0]:
        raise KeyError(f"sessions are missing the {time_key!r} key")

    ordered = sorted(sessions, key=lambda session: session[time_key])
    cut = int(len(ordered) * train_ratio)
    return ordered[:cut], ordered[cut:]
