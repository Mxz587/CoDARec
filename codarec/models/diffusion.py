"""Gaussian diffusion process (forward noise injection and reverse sampling).

The maths follows the standard DDPM formulation specialised for implicit
collaborative filtering: the "image" is a user's interaction vector over all
items and the reverse process regenerates its clean version.
"""

from __future__ import annotations

import enum
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


class ModelMeanType(enum.Enum):
    """What the denoising network predicts."""

    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts the noise


class GaussianDiffusion(nn.Module):
    """Ancestral sampling diffusion model for implicit feedback.

    Parameters
    ----------
    mean_type:
        See :class:`ModelMeanType`.
    noise_schedule:
        One of ``"linear"``, ``"linear-var"``, ``"cosine"``, ``"binomial"``.
    noise_scale, noise_min, noise_max:
        Bounds of the linear variance schedules.
    steps:
        Number of diffusion steps ``T``.
    device:
        Device on which the schedule tensors are allocated.
    history_num_per_term:
        Length of the per-timestep loss history kept by the importance sampler.
    beta_fixed:
        Fix ``beta_1`` to ``1e-5`` to avoid overfitting the first step.
    """

    def __init__(
        self,
        mean_type: ModelMeanType,
        noise_schedule: str,
        noise_scale: float,
        noise_min: float,
        noise_max: float,
        steps: int,
        device: torch.device,
        history_num_per_term: int = 10,
        beta_fixed: bool = True,
    ) -> None:
        super().__init__()
        self.mean_type = mean_type
        self.noise_schedule = noise_schedule
        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps
        self.device = torch.device(device)

        self.history_num_per_term = history_num_per_term
        self.register_buffer(
            "Lt_history",
            torch.zeros(steps, history_num_per_term, dtype=torch.float64, device=self.device),
            persistent=False,
        )
        self.register_buffer(
            "Lt_count",
            torch.zeros(steps, dtype=torch.int64, device=self.device),
            persistent=False,
        )

        if noise_scale != 0.0:
            betas = torch.tensor(self.get_betas(), dtype=torch.float64, device=self.device)
            if beta_fixed:
                # Deep Unsupervised Learning using Nonequilibrium Thermodynamics 2.4.1
                betas[0] = 1e-5
            assert betas.shape == (steps,), "betas must be 1-D with one entry per step"
            assert (betas > 0).all() and (betas <= 1).all(), "betas out of range"
            self.register_buffer("betas", betas, persistent=False)
            self.calculate_for_diffusion()

    # ------------------------------------------------------------------ #
    # Schedule construction
    # ------------------------------------------------------------------ #
    def get_betas(self) -> np.ndarray:
        """Build the beta schedule selected by ``noise_schedule``."""
        if self.noise_schedule in ("linear", "linear-var"):
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            return betas_from_linear_variance(
                self.steps, np.linspace(start, end, self.steps, dtype=np.float64)
            )
        if self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
                self.steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )
        if self.noise_schedule == "binomial":
            # Deep Unsupervised Learning using Nonequilibrium Thermodynamics 2.4.1
            return np.asarray(
                [1 / (self.steps - t + 1) for t in range(self.steps)], dtype=np.float64
            )
        raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")

    def calculate_for_diffusion(self) -> None:
        """Pre-compute every cumulative quantity needed by the process."""
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0), persistent=False)
        self.register_buffer(
            "alphas_cumprod_prev",
            torch.cat([torch.tensor([1.0], device=self.device, dtype=alphas.dtype),
                       self.alphas_cumprod[:-1]]),
            persistent=False,
        )
        self.register_buffer(
            "alphas_cumprod_next",
            torch.cat([self.alphas_cumprod[1:],
                       torch.tensor([0.0], device=self.device, dtype=alphas.dtype)]),
            persistent=False,
        )
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod),
                             persistent=False)
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - self.alphas_cumprod), persistent=False)
        self.register_buffer("log_one_minus_alphas_cumprod",
                             torch.log(1.0 - self.alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_recip_alphas_cumprod",
                             torch.sqrt(1.0 / self.alphas_cumprod), persistent=False)
        self.register_buffer("sqrt_recipm1_alphas_cumprod",
                             torch.sqrt(1.0 / self.alphas_cumprod - 1), persistent=False)

        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance, persistent=False)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.cat([posterior_variance[1].unsqueeze(0), posterior_variance[1:]])),
            persistent=False,
        )
        self.register_buffer(
            "posterior_mean_coef1",
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod),
            persistent=False,
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod),
            persistent=False,
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        reweight: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute the denoising loss for a batch of clean interaction vectors."""
        batch_size = x_start.size(0)
        ts, pt = self.sample_timesteps(batch_size, "importance")
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, ts, noise) if self.noise_scale != 0.0 else x_start

        model_output = model(x_t, ts)
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)

        if reweight:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts)
                weight = torch.where(ts == 0, torch.ones_like(weight), weight)
                loss = mse
            else:  # EPSILON
                weight = (1 - self.alphas_cumprod[ts]) / (
                    (1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts])
                )
                weight = torch.where(ts == 0, torch.ones_like(weight), weight)
                likelihood = mean_flat(
                    (x_start - self._predict_xstart_from_eps(x_t, ts, model_output)) ** 2 / 2.0
                )
                loss = torch.where(ts == 0, likelihood, mse)
            weight = weight.to(loss.dtype)
        else:
            weight = torch.ones(batch_size, device=x_start.device, dtype=mse.dtype)

        terms = {"loss": weight * loss, "mse": mse}

        # Update the per-timestep loss history used by the importance sampler.
        with torch.no_grad():
            for t, sample_loss in zip(ts.tolist(), terms["loss"].detach()):
                count = int(self.Lt_count[t])
                if count == self.history_num_per_term:
                    self.Lt_history[t, :-1] = self.Lt_history[t, 1:].clone()
                    self.Lt_history[t, -1] = sample_loss
                else:
                    self.Lt_history[t, count] = sample_loss
                    self.Lt_count[t] += 1

        terms["loss"] = terms["loss"] / pt
        return terms

    def sample_timesteps(
        self,
        batch_size: int,
        method: str = "uniform",
        uniform_prob: float = 0.001,
    ):
        """Sample diffusion timesteps; ``importance`` follows the loss history."""
        if method == "importance":
            if not bool((self.Lt_count == self.history_num_per_term).all()):
                return self.sample_timesteps(batch_size, method="uniform")

            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history ** 2, dim=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all = pt_all * (1 - uniform_prob) + uniform_prob / len(pt_all)
            assert pt_all.sum(-1) - 1.0 < 1e-5

            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)
            return t, pt

        if method == "uniform":
            t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()
            pt = torch.ones_like(t).float()
            return t, pt

        raise ValueError(f"unknown timestep sampling method {method!r}")

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward process: add noise to ``x_start`` at step ``t``."""
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def p_sample(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        steps: int,
        sampling_noise: bool = False,
    ) -> torch.Tensor:
        """Ancestral sampling: regenerate ``x_0`` from its noisy version."""
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            x_t = x_start
        else:
            t = torch.full((x_start.shape[0],), steps - 1, dtype=torch.long, device=x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.0:
            for i in indices:
                t = torch.full((x_t.shape[0],), i, dtype=torch.long, device=x_t.device)
                x_t = model(x_t, t)
            return x_t

        for i in indices:
            t = torch.full((x_t.shape[0],), i, dtype=torch.long, device=x_t.device)
            out = self.p_mean_variance(model, x_t, t)
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                x_t = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"]
        return x_t

    def p_mean_variance(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Apply the model to obtain ``p(x_{t-1} | x_t)`` and the predicted ``x_0``."""
        batch_size = x.shape[0]
        assert t.shape == (batch_size,)
        model_output = model(x, t)

        model_variance = self._extract_into_tensor(self.posterior_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x.shape
        )

        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )
        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ):
        """Mean and variance of the diffusion posterior ``q(x_{t-1} | x_t, x_0)``."""
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _predict_xstart_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Recover ``x_0`` from a noise prediction."""
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def SNR(self, t: torch.Tensor) -> torch.Tensor:
        """Signal-to-noise ratio at timestep(s) ``t``."""
        alphas_cumprod = self.alphas_cumprod.to(t.device)
        return alphas_cumprod[t] / (1 - alphas_cumprod[t])

    def _extract_into_tensor(
        self,
        arr: torch.Tensor,
        timesteps: torch.Tensor,
        broadcast_shape: torch.Size,
    ) -> torch.Tensor:
        """Gather ``arr[timesteps]`` and broadcast it to ``broadcast_shape``."""
        res = arr.to(timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


def betas_from_linear_variance(
    steps: int,
    variance: np.ndarray,
    max_beta: float = 0.999,
) -> np.ndarray:
    """Convert a linear variance schedule into betas."""
    alpha_bar = 1 - variance
    betas = [1 - alpha_bar[0]]
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    alpha_bar,
    max_beta: float = 0.999,
) -> np.ndarray:
    """Discretise an analytical ``alpha_bar`` schedule into betas."""
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def normal_kl(
    mean1,
    logvar1,
    mean2,
    logvar2,
) -> torch.Tensor:
    """KL divergence between two diagonal Gaussians (broadcasting allowed)."""
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def mean_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Mean over all non-batch dimensions."""
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


__all__ = [
    "GaussianDiffusion",
    "ModelMeanType",
    "betas_for_alpha_bar",
    "betas_from_linear_variance",
    "mean_flat",
    "normal_kl",
]
