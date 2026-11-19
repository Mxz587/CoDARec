"""Training and evaluation loops for CoDARec."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from codarec import metrics
from codarec.checkpoint import build_model, save_checkpoint
from codarec.config import Config
from codarec.data.dataset import DataBundle, InteractionDataset
from codarec.models.diffusion import GaussianDiffusion, ModelMeanType
from codarec.models.rl import PPOAgent, RecommendationEnv, rollout
from codarec.utils import count_parameters, format_seconds, make_worker_init_fn

MEAN_TYPES = {"x0": ModelMeanType.START_X, "eps": ModelMeanType.EPSILON}


# --------------------------------------------------------------------------- #
# Factory helpers
# --------------------------------------------------------------------------- #
def build_diffusion(cfg: Config, device: torch.device) -> GaussianDiffusion:
    """Instantiate the Gaussian diffusion process described by ``cfg``."""
    return GaussianDiffusion(
        MEAN_TYPES[cfg.mean_type],
        cfg.noise_schedule,
        cfg.noise_scale,
        cfg.noise_min,
        cfg.noise_max,
        cfg.steps,
        device,
        history_num_per_term=cfg.history_num_per_term,
    ).to(device)


def build_data_loaders(
    cfg: Config,
    bundle: DataBundle,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    """Return the shuffled training loader and the ordered evaluation loader."""
    dataset = InteractionDataset(bundle.train)
    common = dict(batch_size=cfg.batch_size, pin_memory=device.type == "cuda")
    train_loader = DataLoader(
        dataset,
        shuffle=True,
        worker_init_fn=make_worker_init_fn(cfg.seed),
        num_workers=cfg.num_workers,
        **common,
    )
    eval_loader = DataLoader(dataset, shuffle=False, num_workers=cfg.num_workers, **common)
    return train_loader, eval_loader


# --------------------------------------------------------------------------- #
# Single epoch
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    cfg: Config,
    device: torch.device,
) -> float:
    """Run one pass over the training data and return the mean loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        losses = diffusion.training_losses(model, batch, cfg.reweight)
        loss = losses["loss"].mean()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach())
    return total_loss / max(len(loader), 1)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_ranking(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    eval_loader: DataLoader,
    ground_truth,
    history_mask,
    cfg: Config,
    device: torch.device,
    rl_env: Optional[RecommendationEnv] = None,
    rl_agent: Optional[PPOAgent] = None,
) -> metrics.MetricSet:
    """Full-ranking evaluation on one split.

    Parameters
    ----------
    ground_truth:
        Sparse matrix holding the interactions that should be recommended.
    history_mask:
        Sparse matrix whose non-zero entries are removed from the predictions
        (items the user has already interacted with).
    rl_env, rl_agent:
        When both are given the PPO policy drives the denoising process instead
        of ancestral sampling.
    """
    model.eval()
    n_users = history_mask.shape[0]
    top_n = cfg.topn[-1]

    target_items: List[List[int]] = [
        ground_truth[user, :].nonzero()[1].tolist() for user in range(n_users)
    ]
    predict_items: List[List[int]] = []

    use_rl = rl_env is not None and rl_agent is not None

    with torch.no_grad():
        for batch_index, batch in enumerate(eval_loader):
            start = batch_index * cfg.batch_size
            stop = start + len(batch)
            history = history_mask[start:stop]
            batch = batch.to(device)

            if use_rl:
                users = torch.arange(start, stop, dtype=torch.long, device=device)
                prediction, _ = rollout(rl_env, rl_agent, users)
            else:
                prediction = diffusion.p_sample(
                    model, batch, cfg.sampling_steps, cfg.sampling_noise
                )

            # Items already seen by the user must never be recommended.
            prediction[history.nonzero()] = -np.inf

            _, indices = torch.topk(prediction, top_n)
            predict_items.extend(indices.cpu().numpy().tolist())

    return metrics.compute_topn_accuracy(target_items, predict_items, cfg.topn)


def test_mask(bundle: DataBundle, cfg: Config):
    """History mask used when evaluating the test split."""
    if cfg.test_with_valid:
        return bundle.train + bundle.valid
    return bundle.train


# --------------------------------------------------------------------------- #
# Full training run
# --------------------------------------------------------------------------- #
def fit(cfg: Config, bundle: DataBundle, device: torch.device, log) -> Dict[str, Any]:
    """Train CoDARec, keeping the best checkpoint on the selection metric."""
    checkpoint_path = os.path.join(cfg.save_dir, cfg.checkpoint_name)

    model = build_model(cfg, bundle.n_items, device)
    diffusion = build_diffusion(cfg, device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader, eval_loader = build_data_loaders(cfg, bundle, device)
    mask_test = test_mask(bundle, cfg)

    rl_env: Optional[RecommendationEnv] = None
    rl_agent: Optional[PPOAgent] = None
    if cfg.use_rl:
        rl_env = RecommendationEnv(
            bundle.train,
            model,
            max_steps=cfg.steps,
            device=device,
            item_embeddings=bundle.item_embeddings,
        )
        rl_agent = PPOAgent(
            state_dim=bundle.n_items,
            action_dim=2,
            hidden_dims=cfg.rl_hidden_dims,
            lr=cfg.rl_lr,
            gamma=cfg.rl_gamma,
            clip_epsilon=cfg.rl_clip_epsilon,
            device=device,
        )

    log.info(f"model parameters: {count_parameters(model):,}")
    log.info(f"checkpoint: {checkpoint_path}")
    log.info(f"diffusion: mean_type={cfg.mean_type} steps={cfg.steps} "
             f"schedule={cfg.noise_schedule} reweight={cfg.reweight}")

    best_value = -np.inf
    best_epoch = 0
    best_valid = None
    best_test = None
    best_metrics: Dict[str, float] = {}

    for epoch in range(1, cfg.epochs + 1):
        if epoch - best_epoch >= cfg.patience:
            log.info(f"early stopping at epoch {epoch} "
                     f"(no improvement for {cfg.patience} epochs)")
            break

        start_time = time.time()
        train_loss = train_one_epoch(model, diffusion, train_loader, optimizer, cfg, device)
        log.info(
            f"epoch {epoch:03d}  loss {train_loss:.4f}  "
            f"({format_seconds(time.time() - start_time)})"
        )

        if epoch % cfg.eval_every != 0:
            continue

        valid_results = evaluate_ranking(
            model, diffusion, eval_loader, bundle.valid, bundle.train, cfg, device,
            rl_env=rl_env, rl_agent=rl_agent,
        )
        test_results = evaluate_ranking(
            model, diffusion, eval_loader, bundle.test, mask_test, cfg, device,
            rl_env=rl_env, rl_agent=rl_agent,
        )
        metrics.report(log, valid_results, test_results, title=f"[epoch {epoch}]")

        value = metrics.select_value(valid_results, cfg.topn, cfg.select_metric)
        if value > best_value:
            best_value, best_epoch = value, epoch
            best_valid, best_test = valid_results, test_results
            best_metrics = {
                **{f"valid_{k}": v for k, v in metrics.as_dict(valid_results, cfg.topn).items()},
                **{f"test_{k}": v for k, v in metrics.as_dict(test_results, cfg.topn).items()},
            }
            save_checkpoint(
                checkpoint_path,
                model,
                cfg,
                bundle.n_items,
                epoch=epoch,
                metrics=best_metrics,
                optimizer=optimizer,
            )
            log.info(f"saved new best checkpoint ({cfg.select_metric}={value:.4f}) "
                     f"to {checkpoint_path}")

    if best_valid is None:
        raise RuntimeError(
            "training finished without any evaluation; "
            "increase --epochs or decrease --eval-every"
        )

    log.rule("=")
    log.info(f"best epoch: {best_epoch:03d}  {cfg.select_metric}={best_value:.4f}")
    metrics.report(log, best_valid, best_test)

    return {
        "best_epoch": best_epoch,
        "best_value": float(best_value),
        "select_metric": cfg.select_metric,
        "valid": best_valid,
        "test": best_test,
        "metrics": best_metrics,
        "checkpoint": checkpoint_path,
    }


__all__ = [
    "build_data_loaders",
    "build_diffusion",
    "evaluate_ranking",
    "fit",
    "test_mask",
    "train_one_epoch",
]
