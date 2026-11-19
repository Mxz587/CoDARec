#!/usr/bin/env python
"""Evaluate a trained CoDARec checkpoint on the validation/test splits.

Example
-------
    python evaluate.py --checkpoint checkpoints/sample_dataset_....pth --device cuda
"""

from __future__ import annotations

import os
import sys

from codarec import metrics
from codarec.checkpoint import load_model
from codarec.config import build_arg_parser, config_from_args
from codarec.data.dataset import load_data
from codarec.engine import build_data_loaders, build_diffusion, evaluate_ranking, test_mask
from codarec.utils import Logger, resolve_device, set_seed


def main(argv=None) -> int:
    parser = build_arg_parser("Evaluate a CoDARec checkpoint")
    parser.add_argument("--checkpoint", default=None, help="path to a .pth checkpoint")
    parser.add_argument("--split", default="test", choices=["valid", "test"],
                        help="split to report")
    parser.add_argument("--use-rl-rollout", dest="use_rl_rollout", action="store_true",
                        help="score with the PPO denoising policy checkpoint")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)

    if not args.checkpoint:
        parser.error("--checkpoint is required")

    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    with Logger(os.path.join(cfg.log_dir, f"{cfg.log_name}_eval.log")) as log:
        model, saved_cfg, payload = load_model(args.checkpoint, device=device)
        log.info(f"loaded checkpoint: {args.checkpoint} (epoch {payload.get('epoch')})")
        log.info(f"device: {device}")

        # Data and diffusion settings come from the checkpoint, so evaluation
        # always reproduces the training-time configuration.
        bundle = load_data(
            saved_cfg.data_dir,
            keep_first_half=saved_cfg.keep_first_half,
            with_item_embeddings=args.use_rl_rollout,
        )
        log.info("dataset:\n" + bundle.describe())

        diffusion = build_diffusion(saved_cfg, device)
        _, eval_loader = build_data_loaders(saved_cfg, bundle, device)

        rl_env = rl_agent = None
        if args.use_rl_rollout:
            from codarec.models.rl import PPOAgent, RecommendationEnv

            rl_env = RecommendationEnv(
                bundle.train,
                model,
                max_steps=saved_cfg.steps,
                device=device,
                item_embeddings=bundle.item_embeddings,
            )
            rl_agent = PPOAgent(
                state_dim=bundle.n_items,
                action_dim=2,
                hidden_dims=saved_cfg.rl_hidden_dims,
                gamma=saved_cfg.rl_gamma,
                clip_epsilon=saved_cfg.rl_clip_epsilon,
                device=device,
            )
            log.info("scoring with the PPO denoising policy (policy is untrained "
                     "unless you optimised it yourself)")

        valid_results = evaluate_ranking(
            model, diffusion, eval_loader, bundle.valid, bundle.train, saved_cfg, device,
            rl_env=rl_env, rl_agent=rl_agent,
        )
        test_history_mask = test_mask(bundle, saved_cfg)
        test_results = evaluate_ranking(
            model, diffusion, eval_loader, bundle.test, test_history_mask, saved_cfg, device,
            rl_env=rl_env, rl_agent=rl_agent,
        )

        metrics.report(log, valid_results, test_results,
                       title=f"CoDARec · {saved_cfg.dataset} · {args.split}")
        flat = metrics.as_dict(test_results, saved_cfg.topn)
        log.info("test metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in flat.items()))

    return 0


if __name__ == "__main__":
    sys.exit(main())
