"""Two-tower neural retriever (M2).

The item tower is an embedding table over the OTTO item vocabulary; the
session tower is an MLP over the mean-pooled history embedding. Both are
trained jointly with a sampled-softmax loss (uniform noise items) so a
session's query vector lands close to its future items in the shared space.

``train_two_tower`` reads the ETL events parquet, samples training sessions
temporally (same ``hash(session_code)`` sampling as the ranker), and exports:

- ``item_embeddings.npy`` + ``item_ids.parquet`` - row-aligned item vectors
- ``session_tower.npz`` - the session MLP weights for a numpy-only forward
  pass at serving time (no torch needed in the API process)
- ``config.json`` / ``metrics.json``

torch is imported lazily inside training so the package stays importable
without ML dependencies installed.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TwoTowerConfig:
    embedding_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 128)
    epochs: int = 5
    batch_size: int = 4096
    lr: float = 0.01
    loss: str = "sampled_softmax"
    n_sessions: int = 400_000
    examples_per_session: int = 2
    max_history: int = 20
    n_noise: int = 128
    temperature: float = 0.07
    seed: int = 42
    valid_fraction: float = 0.05


def _load_training_data(
    train_path: str, config: TwoTowerConfig, vocab_path: Path | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (flat_idx, offsets, targets, is_valid, item_ids).

    Examples are next-item pairs: history = up to ``max_history`` events
    before a random position, target = the event at that position.
    """
    split_path = Path(train_path).parent / "split.json"
    threshold = None
    if split_path.exists():
        threshold = json.loads(split_path.read_text())["threshold_min_ts"]

    started = time.perf_counter()
    from otto_rec.features.etl import connect  # noqa: PLC0415 - duckdb is train-only

    con = connect(memory_limit="4GB", temp_dir="data/tmp")
    con.execute(f"CREATE VIEW events AS SELECT * FROM '{Path(train_path).resolve().as_posix()}'")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE sample_sessions AS
        SELECT session_code FROM events
        {'WHERE ts < ' + str(int(threshold)) if threshold is not None else ''}
        GROUP BY session_code
        HAVING count(*) >= 2
        ORDER BY hash(session_code)
        LIMIT {config.n_sessions}
        """
    )
    sessions = con.execute(
        """
        SELECT e.session_code, e.pos, e.item FROM events e
        JOIN sample_sessions USING (session_code)
        ORDER BY e.session_code, e.pos
        """
    ).fetch_df()
    vocab_df = con.execute("SELECT DISTINCT item FROM events ORDER BY item").fetch_df()
    con.close()
    item_ids = vocab_df["item"].astype(str).to_numpy()
    if vocab_path is not None:
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(vocab_path, item_ids)
    code_of = {item: code for code, item in enumerate(item_ids)}

    rng = np.random.default_rng(config.seed)
    flat: list[int] = []
    offsets: list[int] = [0]
    targets: list[int] = []
    valid_flags: list[int] = []
    for _, group in sessions.groupby("session_code", sort=False):
        codes = [code_of[str(item)] for item in group["item"].tolist()]
        if len(codes) < 2:
            continue
        positions = rng.choice(
            np.arange(1, len(codes)),
            size=min(config.examples_per_session, len(codes) - 1),
            replace=False,
        )
        for position in sorted(int(p) for p in positions):
            history = codes[max(0, position - config.max_history) : position]
            if not history:
                continue
            flat.extend(history)
            offsets.append(len(flat))
            targets.append(codes[position])
            valid_flags.append(1 if rng.random() < config.valid_fraction else 0)
    print(
        f"[two_tower] data: {len(targets)} examples over "
        f"{sessions['session_code'].nunique()} sessions, vocab {len(item_ids)} "
        f"({time.perf_counter() - started:.1f}s)",
        flush=True,
    )
    return (
        np.asarray(flat, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        np.asarray(valid_flags, dtype=bool),
        item_ids,
    )


def train_two_tower(
    train_path: str, config: TwoTowerConfig | None = None, output_dir: str = "models/two_tower"
) -> str:
    """Train both towers and return the export directory."""
    import torch
    import torch.nn as nn

    config = config or TwoTowerConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    flat, offsets, targets, is_valid, item_ids = _load_training_data(
        train_path, config, out / "vocab.npy"
    )
    n_items = len(item_ids)
    n_examples = len(targets)
    if n_examples == 0:
        raise ValueError("no training examples built from events")

    class TwoTowerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.item_emb = nn.Embedding(n_items, config.embedding_dim)
            dims = [config.embedding_dim, *config.hidden_dims, config.embedding_dim]
            layers: list[nn.Module] = []
            for in_dim, out_dim in zip(dims[:-1], dims[1:]):
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())
            layers.pop()  # linear output, normalization happens in the loss
            self.session_mlp = nn.Sequential(*layers)
            nn.init.normal_(self.item_emb.weight, std=0.05)

        def embed_items(self, indices: torch.Tensor) -> torch.Tensor:
            return nn.functional.normalize(self.item_emb(indices), dim=-1)

        def forward(self, mean_history: torch.Tensor) -> torch.Tensor:
            return nn.functional.normalize(self.session_mlp(mean_history), dim=-1)

    model = TwoTowerModel()
    # Adagrad keeps one accumulator instead of Adam's two moments, which the
    # 1.86M x 64 embedding table cannot afford alongside the serving pipeline.
    # Default zero accumulator makes the first updates lr-normalised per row;
    # the embedding table starts at std 0.05 so those steps stay stable.
    optimizer = torch.optim.Adagrad(model.parameters(), lr=config.lr)
    rng = np.random.default_rng(config.seed)

    def batch_tensors(example_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        starts = offsets[example_ids]
        ends = offsets[example_ids + 1]
        lengths = (ends - starts).astype(np.int64)
        seg = np.repeat(np.arange(len(example_ids)), lengths)
        cursor = np.arange(lengths.sum()) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        indices = flat[np.repeat(starts, lengths) + cursor]
        return (
            torch.from_numpy(indices),
            torch.from_numpy(seg.astype(np.int64)),
            torch.from_numpy(targets[example_ids]),
        )

    def mean_pool(indices: torch.Tensor, seg: torch.Tensor, n_batches: int) -> torch.Tensor:
        emb = model.item_emb(indices)
        sums = torch.zeros(n_batches, config.embedding_dim).index_add_(0, seg, emb)
        counts = torch.bincount(seg, minlength=n_batches).unsqueeze(1).clamp(min=1)
        return sums / counts

    n_valid = int(is_valid.sum())
    valid_ids = np.flatnonzero(is_valid)
    order = np.arange(n_examples)
    started = time.perf_counter()
    metrics: dict[str, float] = {}
    for epoch in range(1, config.epochs + 1):
        model.train()
        rng.shuffle(order)
        epoch_losses = []
        for start in range(0, n_examples, config.batch_size):
            batch = order[start : start + config.batch_size]
            if batch.size == 0:
                continue
            indices, seg, pos = batch_tensors(batch)
            query = model(mean_pool(indices, seg, len(batch)))
            positive = model.embed_items(pos)
            noise = torch.randint(0, n_items, (len(batch), config.n_noise))
            # In-batch softmax (other sessions' targets as negatives) gives a
            # strong, cheap signal; uniform noise keeps rare items visible.
            logits = torch.cat(
                [
                    query @ positive.T,
                    torch.bmm(model.embed_items(noise), query.unsqueeze(-1)).squeeze(-1),
                ],
                dim=1,
            ) / config.temperature
            loss = nn.functional.cross_entropy(logits, torch.arange(len(batch)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        record: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": round(float(np.mean(epoch_losses)), 5),
        }
        if n_valid and epoch == config.epochs:
            model.eval()
            with torch.no_grad():
                hits = 0
                for start in range(0, n_valid, config.batch_size):
                    batch = valid_ids[start : start + config.batch_size]
                    indices, seg, pos = batch_tensors(batch)
                    query = model(mean_pool(indices, seg, len(batch)))
                    positive = model.embed_items(pos)
                    noise = torch.randint(0, n_items, (len(batch), config.n_noise))
                    scores = torch.cat(
                        [
                            query @ positive.T,
                            torch.bmm(model.embed_items(noise), query.unsqueeze(-1)).squeeze(-1),
                        ],
                        dim=1,
                    )
                    hits += int((scores.argmax(dim=1) == torch.arange(len(batch))).sum())
                record["valid_hit@1"] = round(hits / n_valid, 5)
        metrics.update(record)
        print(f"[two_tower] {record}", flush=True)

    with torch.no_grad():
        item_embeddings = model.item_emb.weight.detach().cpu().numpy().astype(np.float32)
    np.save(out / "item_embeddings.npy", item_embeddings)
    pd.DataFrame({"item": item_ids}).to_parquet(out / "item_ids.parquet", index=False)
    session_weights = {
        name: value.detach().cpu().numpy()
        for name, value in model.session_mlp.state_dict().items()
    }
    np.savez(out / "session_tower.npz", **session_weights)
    (out / "config.json").write_text(json.dumps(asdict(config), indent=2, default=list))
    metrics.update(
        {
            "n_examples": n_examples,
            "n_items": n_items,
            "elapsed_s": round(time.perf_counter() - started, 1),
        }
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[two_tower] exported to {out}: {metrics}", flush=True)
    return str(out)


def export_embeddings(output_dir: str, item_ids: Sequence[str]) -> str:
    """Write item embeddings for FAISS index construction."""
    out = Path(output_dir)
    embeddings = np.load(out / "item_embeddings.npy", mmap_mode="r")
    known = pd.read_parquet(out / "item_ids.parquet")["item"].astype(str)
    position = {item: index for index, item in enumerate(known.tolist())}
    rows = [position[item] for item in item_ids if item in position]
    selected = np.asarray(embeddings[rows], dtype=np.float32)
    kept = [item for item in item_ids if item in position]
    vectors_path = out / "faiss_vectors.npy"
    ids_path = out / "faiss_ids.parquet"
    np.save(vectors_path, selected)
    pd.DataFrame({"item": kept}).to_parquet(ids_path, index=False)
    return str(vectors_path)


_TOWER_CACHE: dict[str, dict[str, np.ndarray]] = {}


def _load_tower(output_dir: str) -> dict[str, np.ndarray]:
    """Load session-tower weights once per directory (npz parse is not free)."""
    weights = _TOWER_CACHE.get(output_dir)
    if weights is None:
        loaded = np.load(Path(output_dir) / "session_tower.npz")
        weights = {name: loaded[name] for name in loaded.files}
        _TOWER_CACHE[output_dir] = weights
    return weights


def session_query_vector(history_vectors: np.ndarray, output_dir: str) -> np.ndarray:
    """Numpy-only session-tower forward pass over mean-pooled history."""
    weights = _load_tower(output_dir)
    vector = history_vectors.mean(axis=0)
    linears = sorted(
        (name for name in weights if name.endswith(".weight")),
        key=lambda name: int(name.split(".")[0]),
    )
    for position, weight_name in enumerate(linears):
        vector = vector @ weights[weight_name].T + weights[weight_name.replace("weight", "bias")]
        if position < len(linears) - 1:
            vector = np.maximum(vector, 0.0)
    norm = np.linalg.norm(vector)
    return (vector / norm).astype(np.float32) if norm > 0 else vector.astype(np.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the two-tower retriever")
    parser.add_argument("--events", default="data/processed/events.parquet")
    parser.add_argument("--out", default="models/two_tower")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--n-sessions", type=int, default=None)
    args = parser.parse_args()
    cfg = TwoTowerConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.n_sessions is not None:
        cfg.n_sessions = args.n_sessions
    train_two_tower(args.events, cfg, args.out)
