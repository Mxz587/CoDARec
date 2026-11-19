#!/usr/bin/env python
"""Train CoDARec on an implicit-feedback dataset.

Example
-------
    python train.py --dataset sample_dataset --device cuda --steps 5 --lr 1e-5
"""

from __future__ import annotations

import os
import sys

from codarec.config import parse_config
from codarec.data.dataset import load_data
from codarec.engine import fit
from codarec.utils import Logger, resolve_device, set_seed


def main(argv=None) -> int:
    cfg = parse_config(
        argv, "Train CoDARec (Co-evolutionary Diffusion-Augmented Recommendation)"
    )

    device = resolve_device(cfg.device)
    set_seed(cfg.seed)

    log_path = os.path.join(cfg.log_dir, f"{cfg.log_name}.log")
    with Logger(log_path) as log:
        log.info(f"started: {log.timestamp()}")
        log.info(f"device: {device}")
        log.info("configuration:\n" + _format_config(cfg))
        log.info(f"data directory: {cfg.data_dir}")

        bundle = load_data(
            cfg.data_dir,
            keep_first_half=cfg.keep_first_half,
            with_item_embeddings=cfg.use_rl,
        )
        log.info("dataset:\n" + bundle.describe())

        result = fit(cfg, bundle, device, log)
        log.info(f"finished: {log.timestamp()}")
        log.info(f"best checkpoint: {result['checkpoint']}")

    return 0


def _format_config(cfg) -> str:
    """One ``key = value`` line per hyper-parameter."""
    return "\n".join(f"  {key} = {value}" for key, value in sorted(cfg.to_dict().items()))


if __name__ == "__main__":
    sys.exit(main())
