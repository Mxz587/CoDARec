"""Fast, dependency-light smoke test for CoDARec (CPU only).

It runs on a tiny synthetic dataset, so no downloaded data or GPU is required:

    python tests/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codarec import metrics  # noqa: E402
from codarec.checkpoint import build_model, load_model, save_checkpoint  # noqa: E402
from codarec.config import Config  # noqa: E402
from codarec.data.dataset import DataBundle, InteractionDataset  # noqa: E402
from codarec.engine import (  # noqa: E402
    build_data_loaders,
    build_diffusion,
    evaluate_ranking,
    fit,
    train_one_epoch,
)
from codarec.models.denoising_mlp import timestep_embedding  # noqa: E402
from codarec.models.rl import PPOAgent, RecommendationEnv, rollout  # noqa: E402
from codarec.utils import Logger, set_seed  # noqa: E402


def synthetic_bundle(n_users: int = 40, n_items: int = 50, seed: int = 0) -> DataBundle:
    """A tiny implicit-feedback dataset with dense item embeddings."""
    rng = np.random.default_rng(seed)
    matrices = {}
    for split, per_user in (("train", 6), ("valid", 2), ("test", 2)):
        rows, cols = [], []
        for user in range(n_users):
            items = rng.choice(n_items, size=per_user, replace=False)
            rows.extend([user] * per_user)
            cols.extend(items.tolist())
        matrices[split] = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n_users, n_items)
        )
    return DataBundle(
        train=matrices["train"],
        valid=matrices["valid"],
        test=matrices["test"],
        n_users=n_users,
        n_items=n_items,
        item_embeddings=rng.standard_normal((n_items, 8)).astype(np.float32),
    )


def small_config(tmpdir: str) -> Config:
    return Config(
        dims=[16],
        emb_size=8,
        dropout=0.0,
        steps=5,
        noise_scale=0.1,
        batch_size=16,
        epochs=2,
        eval_every=1,
        patience=5,
        topn=[5, 10],
        select_metric="recall@5",
        save_dir=tmpdir,
        log_dir=tmpdir,
        seed=7,
        device="cpu",
    ).validate()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  ok  {message}")


def test_metrics() -> None:
    """Hand-computed ranking metrics.

    user 0: relevant {1, 2}, ranked [1, 5, 2, 4, 6] -> hits at rank 1 and 3
    user 1: relevant {3},   ranked [9, 8, 7, 6, 5] -> no hit
    """
    ground_truth = [[1, 2], [3]]
    predicted = [[1, 5, 2, 4, 6], [9, 8, 7, 6, 5]]
    precision, recall, ndcg, mrr = metrics.compute_topn_accuracy(ground_truth, predicted, [2, 5])

    # @2: user 0 hits once (1/2), user 1 never -> mean 0.25
    # @5: user 0 hits twice (2/5), user 1 never -> mean 0.2
    check(precision == [0.25, 0.2], f"precision {precision}")
    # @2: recall 1/2 for user 0; @5: recall 2/2 for user 0 -> mean 0.5
    check(recall == [0.25, 0.5], f"recall {recall}")
    # MRR only depends on the first hit -> 1.0 for user 0 at both cut-offs
    check(mrr == [0.5, 0.5], f"mrr {mrr}")
    # NDCG@2 = (1/log2(2)) / (1 + 1/log2(3)) / 2
    check(ndcg[0] == 0.3066, f"ndcg@2 {ndcg[0]}")
    # NDCG@5 = (1 + 1/log2(4)) / (1 + 1/log2(3)) / 2
    check(ndcg[1] == 0.4599, f"ndcg@5 {ndcg[1]}")

    flat = metrics.as_dict((precision, recall, ndcg, mrr), [2, 5])
    check(metrics.select_value((precision, recall, ndcg, mrr), [2, 5], "recall@5") == 0.5,
          "select_value reads recall@5")
    check(flat["precision@2"] == 0.25, "as_dict keys")


def test_dataset(bundle: DataBundle) -> None:
    dataset = InteractionDataset(bundle.train)
    check(len(dataset) == bundle.n_users, "dataset length")
    row = dataset[0]
    check(row.shape == (bundle.n_items,) and row.dtype == torch.float32,
          "dataset row is a dense float32 vector")
    loader, _ = build_data_loaders(small_config(tempfile.gettempdir()), bundle, torch.device("cpu"))
    batch = next(iter(loader))
    check(batch.shape == (min(len(dataset), 16), bundle.n_items), "loader batch shape")


def test_diffusion_and_model(cfg: Config, bundle: DataBundle) -> None:
    device = torch.device("cpu")
    diffusion = build_diffusion(cfg, device)
    model = build_model(cfg, bundle.n_items, device)

    timesteps = torch.arange(4)
    embedding = timestep_embedding(timesteps, cfg.emb_size)
    check(embedding.shape == (4, cfg.emb_size), "timestep embedding shape")

    x_start = torch.rand(4, bundle.n_items)
    losses = diffusion.training_losses(model, x_start, reweight=True)
    check(losses["loss"].shape == (4,), "diffusion loss is per-sample")
    check(float(losses["loss"].mean()) > 0, "diffusion loss is positive")

    sampled = diffusion.p_sample(model, x_start, steps=0)
    check(sampled.shape == x_start.shape, "p_sample keeps the shape")
    check(not torch.equal(sampled, x_start), "p_sample actually denoises")


def test_training_loop(cfg: Config, bundle: DataBundle) -> None:
    device = torch.device("cpu")
    diffusion = build_diffusion(cfg, device)
    model = build_model(cfg, bundle.n_items, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader, eval_loader = build_data_loaders(cfg, bundle, device)

    loss = train_one_epoch(model, diffusion, loader, optimizer, cfg, device)
    check(np.isfinite(loss), f"one epoch trains without NaN ({loss:.4f})")

    results = evaluate_ranking(model, diffusion, eval_loader, bundle.test, bundle.train, cfg, device)
    check(len(results) == 4 and len(results[0]) == len(cfg.topn), "evaluation returns 4 x |topn|")

    # The PPO scoring path must also produce a usable ranking.
    env = RecommendationEnv(
        bundle.train, model, max_steps=3, device=device,
        item_embeddings=bundle.item_embeddings,
    )
    agent = PPOAgent(state_dim=bundle.n_items, hidden_dims=[16], device=device)
    rl_results = evaluate_ranking(
        model, diffusion, eval_loader, bundle.test, bundle.train, cfg, device,
        rl_env=env, rl_agent=agent,
    )
    check(len(rl_results) == 4 and len(rl_results[0]) == len(cfg.topn),
          "PPO-driven evaluation returns the same metric layout")


def test_rl_branch(cfg: Config, bundle: DataBundle) -> None:
    device = torch.device("cpu")
    model = build_model(cfg, bundle.n_items, device)
    env = RecommendationEnv(
        bundle.train, model, max_steps=3, device=device,
        item_embeddings=bundle.item_embeddings,
    )
    agent = PPOAgent(
        state_dim=bundle.n_items, hidden_dims=[16], device=device, lr=1e-3
    )
    users = torch.arange(4)
    scores, transitions = rollout(env, agent, users, store_transitions=True)
    check(scores.shape == (4, bundle.n_items), "rollout returns item scores")
    check(len(transitions) == 3, "rollout records one transition per step")

    losses = agent.update(transitions)
    check(set(losses) == {"policy_loss", "value_loss"}, "PPO update returns losses")
    check(all(np.isfinite(value) for value in losses.values()), f"PPO losses finite {losses}")


def test_checkpoint_roundtrip(cfg: Config, bundle: DataBundle) -> None:
    device = torch.device("cpu")
    model = build_model(cfg, bundle.n_items, device)
    path = os.path.join(cfg.save_dir, "unit_test.pth")
    save_checkpoint(path, model, cfg, bundle.n_items, epoch=3, metrics={"recall@5": 0.5})
    check(os.path.isfile(path), "checkpoint file written")
    check(os.path.isfile(os.path.splitext(path)[0] + "_config.json"), "config json written")

    restored, saved_cfg, payload = load_model(path, device=device)
    check(payload["epoch"] == 3, "epoch stored in checkpoint")
    check(saved_cfg.dims == cfg.dims and saved_cfg.steps == cfg.steps, "config round-trips")
    for (name, lhs), (_, rhs) in zip(
        model.state_dict().items(), restored.state_dict().items()
    ):
        check(torch.equal(lhs, rhs), f"weights round-trip ({name})")
        break


def test_full_fit(cfg: Config, bundle: DataBundle) -> None:
    with Logger(None) as log:
        result = fit(cfg, bundle, torch.device("cpu"), log)
    check(result["best_epoch"] >= 1, f"fit selects an epoch ({result['best_epoch']})")
    check(os.path.isfile(result["checkpoint"]), "fit writes a checkpoint")
    check(np.isfinite(result["best_value"]), "selection metric is finite")


def main() -> int:
    set_seed(0)
    bundle = synthetic_bundle()

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = small_config(tmpdir)
        print("[metrics]")
        test_metrics()
        print("[dataset]")
        test_dataset(bundle)
        print("[diffusion + model]")
        test_diffusion_and_model(cfg, bundle)
        print("[training loop]")
        test_training_loop(cfg, bundle)
        print("[rl branch]")
        test_rl_branch(cfg, bundle)
        print("[checkpoint]")
        test_checkpoint_roundtrip(cfg, bundle)
        print("[fit]")
        test_full_fit(cfg, bundle)

    print("\nall smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
