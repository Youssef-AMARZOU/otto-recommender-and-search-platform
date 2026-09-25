"""Multi-objective GBDT ranker (M3): LightGBM LambdaMART over retrieved candidates.

Trained on the temporal split with group boundaries per session, optimizing a
weighted blend of click / cart / order relevance instead of CTR alone, so the
ranker aligns with the OTTO metric the offline evaluation reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class RankerConfig:
    params: dict = field(
        default_factory=lambda: {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.05,
            "num_leaves": 255,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.8,
        }
    )
    event_weights: dict[str, float] = field(
        default_factory=lambda: {"clicks": 0.10, "carts": 0.25, "orders": 0.65}
    )


def train_ranker(train_path: str, config: RankerConfig | None = None, output_path: str = "models/ranker.txt") -> str:
    """Fit LambdaMART on grouped session candidates; return the model path."""
    raise NotImplementedError("M3: LightGBM LambdaMART training")


def score_candidates(model_path: str, feature_rows: Sequence[Sequence[float]], candidate_ids: Sequence[str]) -> list:
    """Score retrieved candidates with the trained ranker, best first."""
    raise NotImplementedError("M3: batch candidate scoring")
