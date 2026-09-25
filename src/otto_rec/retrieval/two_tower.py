"""Two-tower neural retriever (M2).

The user tower embeds session history plus context features, the item tower
embeds item content/popularity features; both are trained with a contrastive
loss so that a user's embedding is close to their positive items in a shared
space. Serving queries the item tower offline and looks candidates up through
FAISS. torch is imported lazily inside training so the package stays
importable without ML dependencies installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class TwoTowerConfig:
    embedding_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 128)
    epochs: int = 5
    batch_size: int = 4096
    lr: float = 0.001
    loss: str = "sampled_softmax"


def train_two_tower(train_path: str, config: TwoTowerConfig | None = None, output_dir: str = "models/two_tower") -> str:
    """Train both towers and return the export directory."""
    raise NotImplementedError("M2: two-tower training loop")


def export_embeddings(output_dir: str, item_ids: Sequence[str]) -> str:
    """Write item embeddings for FAISS index construction."""
    raise NotImplementedError("M2: embedding export")
