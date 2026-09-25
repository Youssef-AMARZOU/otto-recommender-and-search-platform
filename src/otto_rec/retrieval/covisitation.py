"""Co-visitation candidate generator (M2 baseline).

Counts item-to-item co-occurrences within sessions over a time window and
returns the top-N neighbours per item: the fast, non-neural retrieval
baseline the two-tower retriever must beat.
"""

from __future__ import annotations

from collections.abc import Sequence


def build_co_visitation(
    train_sessions: Sequence[Sequence[str]],
    min_co_occurrence: int = 2,
    top_n: int = 100,
) -> dict[str, list[str]]:
    """Item id -> ranked neighbour ids from session co-occurrence counts."""
    raise NotImplementedError("M2: co-visitation matrix build")


def retrieve(
    session_items: Sequence[str],
    neighbours: dict[str, list[str]],
    top_n: int = 100,
) -> list[str]:
    """Candidate items for a session by aggregating neighbour lists."""
    raise NotImplementedError("M2: co-visitation candidate lookup")
