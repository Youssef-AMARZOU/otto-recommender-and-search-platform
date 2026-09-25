"""Feast feature definitions (M1).

Once M1 lands, entities and feature views are declared at module level so the
Feast CLI discovers them from ``configs/feature_store.yaml``. Intended views:

- session: last-N event counts, session recency, in-session item sequence stats
- item: popularity windows (1/7/30 day clicks, carts, orders), price/category
- user: long-term interaction aggregates and preferred categories

Training and serving both read through this module so feature definitions
never diverge between the two paths.
"""

from __future__ import annotations


def session_feature_views():
    """Feast feature views for session-scoped features."""
    raise NotImplementedError("M1: Feast session feature views")


def item_feature_views():
    """Feast feature views for item-scoped popularity windows."""
    raise NotImplementedError("M1: Feast item feature views")


def user_feature_views():
    """Feast feature views for user-scoped aggregates."""
    raise NotImplementedError("M1: Feast user feature views")
