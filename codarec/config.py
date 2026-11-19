"""Central configuration for CoDARec.

Every hyper-parameter lives in a single :class:`Config` dataclass.  A run is
therefore fully described by one object, which is serialised next to each
checkpoint and can be reproduced exactly from the command line.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def parse_int_list(text: str) -> List[int]:
    """Parse ``"[1000, 400]"`` (or ``"1000,400"``) into ``[1000, 400]``."""
    cleaned = text.strip().strip("[]() ").replace(" ", "")
    if not cleaned:
        return []
    try:
        return [int(part) for part in cleaned.split(",") if part]
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(f"invalid integer list: {text!r}") from exc


def data_dir_of(cfg: "Config") -> str:
    """Return the directory holding the split files of ``cfg.dataset``."""
    return os.path.join(cfg.data_root, cfg.dataset)


# --------------------------------------------------------------------------- #
# Configuration object
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    """All hyper-parameters of a CoDARec experiment."""

    # ---------------------------------------------------------------- data --
    data_root: str = "data"
    dataset: str = "sample_dataset"
    keep_first_half: bool = True
    """The released sample files concatenate two user blocks; when enabled
    only the first block is kept, matching the reported settings."""

    # ------------------------------------------------------------ training --
    seed: int = 1
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 40
    epochs: int = 1000
    patience: int = 20
    """Stop when the validation selection metric has not improved for this
    many epochs (mirrors the original early-stopping rule)."""
    eval_every: int = 5
    topn: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    select_metric: str = "recall@20"
    test_with_valid: bool = False
    """When ``True`` the validation interactions are folded into the history
    mask used at test time."""

    # ------------------------------------------------------------ outputs ----
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    log_name: str = "codarec"
    run_name: str = ""
    """Optional suffix appended to the generated checkpoint name."""

    # --------------------------------------------------------------- model ---
    dims: List[int] = field(default_factory=lambda: [1000])
    emb_size: int = 10
    time_type: str = "cat"
    norm: bool = False
    dropout: float = 0.5

    # ----------------------------------------------------------- diffusion ---
    mean_type: str = "x0"  # "x0" | "eps"
    steps: int = 100
    noise_schedule: str = "linear-var"
    noise_scale: float = 0.1
    noise_min: float = 1e-4
    noise_max: float = 2e-2
    reweight: bool = True
    sampling_steps: int = 0
    sampling_noise: bool = False
    history_num_per_term: int = 10

    # ------------------------------------------------------ reinforcement  ----
    use_rl: bool = False
    """Gate for the PPO denoising policy.  The diffusion model is always
    trained; the policy is an experimental branch kept for reproducibility."""
    rl_lr: float = 3e-4
    rl_gamma: float = 0.99
    rl_clip_epsilon: float = 0.2
    rl_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])

    # --------------------------------------------------------------- misc ----
    num_workers: int = 0

    # ------------------------------------------------------------- helpers ---
    def validate(self) -> "Config":
        """Fail fast on inconsistent settings."""
        if not self.dims:
            raise ValueError("`dims` must contain at least one hidden size")
        if self.mean_type not in ("x0", "eps"):
            raise ValueError(f"unsupported mean_type {self.mean_type!r}; use 'x0' or 'eps'")
        if self.steps <= 0:
            raise ValueError("`steps` must be positive")
        if self.noise_schedule not in ("linear", "linear-var", "cosine", "binomial"):
            raise ValueError(f"unsupported noise_schedule {self.noise_schedule!r}")
        if not 0.0 <= self.noise_scale <= 1.0:
            raise ValueError("`noise_scale` must lie in [0, 1]")
        if self.noise_min <= 0.0 or self.noise_max <= 0.0:
            raise ValueError("`noise_min` and `noise_max` must be positive")
        if self.topn != sorted(self.topn) or not self.topn:
            raise ValueError("`topn` must be a non-empty ascending list")
        if not 0 <= self.sampling_steps <= self.steps:
            raise ValueError("`sampling_steps` must lie in [0, steps]")
        if self.batch_size <= 0 or self.epochs <= 0 or self.eval_every <= 0:
            raise ValueError("batch_size, epochs and eval_every must be positive")
        return self

    @property
    def data_dir(self) -> str:
        return data_dir_of(self)

    @property
    def checkpoint_name(self) -> str:
        stem = "_".join(
            part
            for part in (
                self.dataset,
                f"lr{self.lr:g}",
                f"bs{self.batch_size}",
                "dims" + "-".join(str(d) for d in self.dims),
                f"emb{self.emb_size}",
                f"{self.mean_type}",
                f"steps{self.steps}",
                f"scale{self.noise_scale:g}",
                self.log_name,
                self.run_name,
            )
            if part
        )
        return f"{stem}.pth"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def load_json(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


# --------------------------------------------------------------------------- #
# Dataset presets
# --------------------------------------------------------------------------- #
PRESETS: Dict[str, Dict[str, Any]] = {
    # Diffusive noise settings are dataset dependent.  These presets collect
    # the values used for the released models so that a run is one flag away.
    "sample_dataset": {
        "steps": 5,
        "noise_scale": 0.01,
        "noise_min": 0.001,
        "noise_max": 0.01,
    },
    "amazon-book_clean": {
        "steps": 5,
        "noise_scale": 0.0001,
        "noise_min": 0.0005,
        "noise_max": 0.005,
    },
    "ml-1m_clean": {
        "steps": 40,
        "noise_scale": 0.005,
        "noise_min": 0.005,
        "noise_max": 0.01,
    },
    "amazon-book_noisy": {
        "steps": 10,
        "noise_scale": 0.005,
        "noise_min": 0.0001,
        "noise_max": 0.0005,
    },
    "sample_noisy": {
        "steps": 5,
        "noise_scale": 0.001,
        "noise_min": 0.0005,
        "noise_max": 0.01,
    },
    "ml-1m_noisy": {
        "steps": 5,
        "noise_scale": 0.5,
        "noise_min": 0.001,
        "noise_max": 0.01,
    },
}


def apply_preset(cfg: Config, preset: str) -> Config:
    """Return a copy of ``cfg`` with the diffusion settings of ``preset``."""
    if preset not in PRESETS:
        raise ValueError(
            f"unknown preset {preset!r}; available: {', '.join(sorted(PRESETS))}"
        )
    payload = cfg.to_dict()
    payload.update(PRESETS[preset])
    return Config.from_dict(payload)


def parse_config(argv, description: str) -> Config:
    """Parse ``argv`` into a :class:`Config`, applying ``--preset`` first.

    Precedence: command line > preset > built-in defaults.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--preset", default=None, choices=sorted(PRESETS))
    known, _ = pre_parser.parse_known_args(argv)

    base = Config()
    if known.preset:
        base = apply_preset(base, known.preset)

    parser = build_arg_parser(description, defaults=base)
    return config_from_args(parser.parse_args(argv))


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
def build_arg_parser(description: str, defaults: "Config | None" = None) -> argparse.ArgumentParser:
    """Build an argument parser whose defaults mirror ``defaults`` (or Config)."""
    defaults = defaults or Config()
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--preset", default=None, choices=sorted(PRESETS),
                        help="dataset-specific diffusion preset")

    data = parser.add_argument_group("data")
    data.add_argument("--data-root", dest="data_root", default=defaults.data_root,
                      help="root directory containing one folder per dataset")
    data.add_argument("--dataset", default=defaults.dataset, help="dataset folder name")
    data.add_argument("--keep-first-half", dest="keep_first_half",
                      default=defaults.keep_first_half,
                      action=argparse.BooleanOptionalAction,
                      help="keep only the first user block of the released splits")
    data.add_argument("--num-workers", dest="num_workers", type=int,
                      default=defaults.num_workers, help="DataLoader workers")

    train = parser.add_argument_group("training")
    train.add_argument("--seed", type=int, default=defaults.seed)
    train.add_argument("--device", default=defaults.device, choices=["auto", "cpu", "cuda"])
    train.add_argument("--lr", type=float, default=defaults.lr)
    train.add_argument("--weight-decay", dest="weight_decay", type=float,
                       default=defaults.weight_decay)
    train.add_argument("--batch-size", dest="batch_size", type=int, default=defaults.batch_size)
    train.add_argument("--epochs", type=int, default=defaults.epochs)
    train.add_argument("--patience", type=int, default=defaults.patience)
    train.add_argument("--eval-every", dest="eval_every", type=int, default=defaults.eval_every)
    train.add_argument("--topn", type=parse_int_list, default=list(defaults.topn),
                       help='evaluation cut-offs, e.g. "[10, 20, 50, 100]"')
    train.add_argument("--select-metric", dest="select_metric", default=defaults.select_metric,
                       help="metric used for model selection, e.g. recall@20")
    train.add_argument("--test-with-valid", dest="test_with_valid",
                      default=defaults.test_with_valid,
                      action=argparse.BooleanOptionalAction,
                      help="also mask validation items when testing")

    out = parser.add_argument_group("output")
    out.add_argument("--save-dir", dest="save_dir", default=defaults.save_dir)
    out.add_argument("--log-dir", dest="log_dir", default=defaults.log_dir)
    out.add_argument("--log-name", dest="log_name", default=defaults.log_name)
    out.add_argument("--run-name", dest="run_name", default=defaults.run_name)

    model = parser.add_argument_group("model")
    model.add_argument("--dims", type=parse_int_list, default=list(defaults.dims),
                       help='hidden sizes of the denoising MLP, e.g. "[1000]"')
    model.add_argument("--emb-size", dest="emb_size", type=int, default=defaults.emb_size,
                       help="timestep embedding size")
    model.add_argument("--time-type", dest="time_type", default=defaults.time_type,
                       choices=["cat"])
    model.add_argument("--norm", default=defaults.norm, action=argparse.BooleanOptionalAction,
                       help="L2-normalise the input interaction vector")
    model.add_argument("--dropout", type=float, default=defaults.dropout)

    diff = parser.add_argument_group("diffusion")
    diff.add_argument("--mean-type", dest="mean_type", default=defaults.mean_type,
                      choices=["x0", "eps"])
    diff.add_argument("--steps", type=int, default=defaults.steps,
                      help="number of diffusion steps")
    diff.add_argument("--noise-schedule", dest="noise_schedule",
                      default=defaults.noise_schedule,
                      choices=["linear", "linear-var", "cosine", "binomial"])
    diff.add_argument("--noise-scale", dest="noise_scale", type=float,
                      default=defaults.noise_scale)
    diff.add_argument("--noise-min", dest="noise_min", type=float, default=defaults.noise_min)
    diff.add_argument("--noise-max", dest="noise_max", type=float, default=defaults.noise_max)
    diff.add_argument("--reweight", default=defaults.reweight,
                      action=argparse.BooleanOptionalAction,
                      help="importance weighting of the diffusion loss")
    diff.add_argument("--sampling-steps", dest="sampling_steps", type=int,
                      default=defaults.sampling_steps,
                      help="0 keeps the full sampling chain")
    diff.add_argument("--sampling-noise", dest="sampling_noise",
                      default=defaults.sampling_noise,
                      action=argparse.BooleanOptionalAction,
                      help="add noise during sampling")
    diff.add_argument("--history-num-per-term", dest="history_num_per_term", type=int,
                      default=defaults.history_num_per_term,
                      help="history length of the importance sampler")

    rl = parser.add_argument_group("reinforcement learning (experimental)")
    rl.add_argument("--use-rl", dest="use_rl", default=defaults.use_rl,
                    action=argparse.BooleanOptionalAction,
                    help="run the PPO denoising policy instead of plain sampling")
    rl.add_argument("--rl-lr", dest="rl_lr", type=float, default=defaults.rl_lr)
    rl.add_argument("--rl-gamma", dest="rl_gamma", type=float, default=defaults.rl_gamma)
    rl.add_argument("--rl-clip-epsilon", dest="rl_clip_epsilon", type=float,
                    default=defaults.rl_clip_epsilon)
    rl.add_argument("--rl-hidden-dims", dest="rl_hidden_dims", type=parse_int_list,
                    default=list(defaults.rl_hidden_dims))

    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Turn a parsed namespace into a validated :class:`Config`."""
    payload = {key: value for key, value in vars(args).items()}
    return Config.from_dict(payload).validate()
