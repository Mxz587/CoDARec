"""The denoising network of CoDARec.

The reverse diffusion process predicts the clean user interaction vector from
a noisy one.  The network is an MLP that is conditioned on the diffusion
timestep through a sinusoidal embedding concatenated to the input.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings.

    Parameters
    ----------
    timesteps:
        1-D tensor of ``N`` indices, one per batch element.
    dim:
        Dimension of the output.
    max_period:
        Controls the minimum frequency of the embeddings.

    Returns
    -------
    An ``(N, dim)`` tensor of positional embeddings on ``timesteps.device``.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class DenoisingMLP(nn.Module):
    """MLP used for the reverse diffusion process.

    Parameters
    ----------
    in_dims / out_dims:
        Layer widths.  ``in_dims[-1]`` must equal ``out_dims[0]`` (the number of
        items); the same list reversed is the natural choice.
    emb_size:
        Size of the timestep embedding concatenated to the input.
    time_type:
        Conditioning strategy; only ``"cat"`` is supported.
    norm:
        L2-normalise the noisy interaction vector before the first layer.
    dropout:
        Dropout rate applied to the input.
    """

    def __init__(
        self,
        in_dims: Sequence[int],
        out_dims: Sequence[int],
        emb_size: int,
        time_type: str = "cat",
        norm: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if time_type != "cat":
            raise ValueError(f"unsupported timestep embedding type {time_type!r}")
        if out_dims[0] != in_dims[-1]:
            raise ValueError(
                "in_dims[-1] and out_dims[0] must match, "
                f"got {in_dims[-1]} and {out_dims[0]}"
            )

        self.in_dims: List[int] = list(in_dims)
        self.out_dims: List[int] = list(out_dims)
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])]
        )
        self.out_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])]
        )

        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier-normal weights with small-normal biases (the default init)."""
        for layer in list(self.in_layers) + list(self.out_layers):
            fan_out, fan_in = layer.weight.size()
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        fan_out, fan_in = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict ``x_0`` (or the noise) from the noisy input ``x``."""
        time_emb = timestep_embedding(timesteps, self.time_emb_dim)
        emb = self.emb_layer(time_emb)

        if self.norm:
            x = F.normalize(x)
        x = x.to(time_emb.dtype)

        h = torch.cat([self.drop(x), emb], dim=-1)
        for layer in self.in_layers:
            h = torch.tanh(layer(h))

        last = len(self.out_layers) - 1
        for index, layer in enumerate(self.out_layers):
            h = layer(h)
            if index != last:
                h = torch.tanh(h)
        return h


# Historical alias kept for readability in downstream scripts.
DNN = DenoisingMLP

__all__ = ["DNN", "DenoisingMLP", "timestep_embedding"]
