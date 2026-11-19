"""PPO denoising policy of CoDARec (experimental branch).

Instead of running the fixed ancestral sampling chain, a policy network (the
"actor") chooses, at every denoising step, how strongly to follow the model
prediction and how much exploratory noise to inject.  A "critic" estimates the
state value so the actor can be trained with PPO.

The branch is intentionally gated behind ``--use-rl``: the released experiments
use plain ancestral sampling, and this module documents the co-evolution idea
together with a working, testable starting point for further research.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from codarec.models.denoising_mlp import timestep_embedding


@dataclass
class Transition:
    """All per-sample quantities produced by one denoising step."""

    state: torch.Tensor  # (batch, n_items)
    action: torch.Tensor  # (batch, action_dim)
    log_prob: torch.Tensor  # (batch,)
    reward: torch.Tensor  # (batch,)
    timestep: int


class RecommendationEnv:
    """Reverse diffusion process exposed as an RL environment.

    * state  -- the current noisy interaction vector of a batch of users;
    * action -- ``[guidance_logit, exploration_logit]``: the first coordinate
      blends model prediction with the current state, the second scales the
      exploration noise injected afterwards;
    * reward -- sparse, granted on the final step: alignment with the user's
      history (0.7) plus diversity of the top-10 recommendations (0.3).
    """

    def __init__(
        self,
        user_item_matrix,
        denoiser: nn.Module,
        max_steps: int = 10,
        device: torch.device | str = "cpu",
        item_embeddings: Optional[np.ndarray] = None,
        initial_noise_scale: float = 1.0,
        exploration_scale: float = 1e-2,
    ) -> None:
        self.device = torch.device(device)
        self.model = denoiser.to(self.device)
        self._sparse = hasattr(user_item_matrix, "toarray")
        if self._sparse:
            self._matrix = user_item_matrix.tocsr().astype(np.float32)
            self.num_users, self.num_items = self._matrix.shape
        else:
            self._matrix = torch.as_tensor(
                np.asarray(user_item_matrix), dtype=torch.float32, device=self.device
            )
            self.num_users, self.num_items = self._matrix.shape
        self.max_steps = max_steps
        self.noise_scale = initial_noise_scale
        self.exploration_scale = exploration_scale

        self.item_embeddings: Optional[torch.Tensor] = None
        if item_embeddings is not None:
            embeddings = torch.as_tensor(item_embeddings, dtype=torch.float32, device=self.device)
            if embeddings.shape[0] != self.num_items:
                raise ValueError(
                    f"item embeddings cover {embeddings.shape[0]} items, "
                    f"the interaction matrix has {self.num_items}"
                )
            self.item_embeddings = embeddings

        self.current_state: Optional[torch.Tensor] = None
        self.current_users: Optional[torch.Tensor] = None
        self.current_step = 0

    # ------------------------------------------------------------------ #
    def reset(self, users: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Start a new episode for ``users`` (a batch of user row indices)."""
        self.current_step = 0
        if users is None:
            users = torch.randint(0, self.num_users, (1,), device=self.device)
        users = torch.as_tensor(users, device=self.device).long().view(-1)
        self.current_users = users

        history = self.history_rows(self.current_users)
        self.current_state = history + torch.randn_like(history) * self.noise_scale
        return self.current_state

    def history_rows(self, users: torch.Tensor) -> torch.Tensor:
        """Dense interaction rows of ``users`` on the environment device."""
        users = users.detach().to("cpu").numpy() if torch.is_tensor(users) else np.asarray(users)
        if self._sparse:
            block = self._matrix[users]
            dense = np.asarray(block.toarray(), dtype=np.float32)
            return torch.from_numpy(dense).to(self.device)
        return self._matrix[torch.as_tensor(users, device=self.device)]

    def step(
        self, action: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, bool, dict]:
        """Apply one denoising step; returns ``(state, reward, done, info)``."""
        if self.current_state is None:
            raise RuntimeError("call reset() before step()")

        batch_size = self.current_state.shape[0]
        timestep = torch.full(
            (batch_size,), self.current_step, dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            denoised = self.model(self.current_state, timestep)

        guidance = action["guidance_scale"].view(batch_size, 1).detach()
        noise_scale = action["exploration_noise"].view(batch_size, 1).detach()
        exploration = noise_scale * torch.randn_like(denoised) * self.exploration_scale

        self.current_state = (
            guidance * denoised + (1.0 - guidance) * self.current_state + exploration
        )
        self.current_step += 1
        done = self.current_step >= self.max_steps

        reward = (
            self._final_reward(denoised)
            if done
            else torch.zeros(batch_size, device=self.device)
        )
        return self.current_state, reward, done, {"step": self.current_step}

    # ------------------------------------------------------------------ #
    def _final_reward(self, denoised: torch.Tensor) -> torch.Tensor:
        """0.7 * preference alignment + 0.3 * recommendation diversity."""
        history = self.history_rows(self.current_users)
        similarity = torch.cosine_similarity(denoised, history, dim=-1)
        return 0.7 * similarity + 0.3 * self._diversity(denoised)

    def _diversity(self, scores: torch.Tensor) -> torch.Tensor:
        """Variance of the embeddings of the top-10 recommended items."""
        top_k = min(10, scores.shape[-1])
        items = torch.topk(scores, top_k, dim=-1).indices
        if self.item_embeddings is None:
            return torch.var(torch.gather(scores, -1, items), dim=-1)
        return torch.var(self.item_embeddings[items], dim=-1).mean(dim=-1)


class PPOAgent:
    """Gaussian-policy PPO agent producing continuous denoising actions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_dims: Sequence[int] = (256, 256),
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        time_embedding_dim: int = 64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_embedding_dim = time_embedding_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        hidden = [int(width) for width in hidden_dims]
        if not hidden:
            raise ValueError("hidden_dims must not be empty")
        input_dim = state_dim + time_embedding_dim

        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.Tanh(),
            *self._extra_layers(hidden),
            nn.Linear(hidden[-1], action_dim * 2),  # mean and log-std of the action
        ).to(self.device)
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.Tanh(),
            *self._extra_layers(hidden),
            nn.Linear(hidden[-1], 1),
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr
        )

    @staticmethod
    def _extra_layers(hidden: List[int]) -> List[nn.Module]:
        layers: List[nn.Module] = []
        for d_in, d_out in zip(hidden[:-1], hidden[1:]):
            layers.extend([nn.Linear(d_in, d_out), nn.Tanh()])
        return layers

    # ------------------------------------------------------------------ #
    def _encode(self, state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Concatenate the state with its sinusoidal timestep embedding."""
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        timestep = torch.as_tensor(timestep, device=self.device).long().view(-1)
        if timestep.numel() == 1:
            timestep = timestep.expand(state.shape[0])
        elif timestep.numel() != state.shape[0]:
            raise ValueError(
                f"expected {state.shape[0]} timesteps, got {timestep.numel()}"
            )

        time_emb = timestep_embedding(timestep, self.time_embedding_dim)
        return torch.cat([state, time_emb], dim=-1)

    def _distribution(self, features: torch.Tensor) -> torch.distributions.Normal:
        mu, log_std = self.policy_net(features).chunk(2, dim=-1)
        return torch.distributions.Normal(mu, log_std.clamp(-5.0, 2.0).exp())

    def get_action(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Sample an action (or take the policy mean when ``sample=False``)."""
        distribution = self._distribution(self._encode(state, timestep))
        action = distribution.sample() if sample else distribution.mean
        return {
            "action": action,
            "log_prob": distribution.log_prob(action).sum(-1),
            "guidance_scale": torch.sigmoid(action[..., 0]),
            "exploration_noise": torch.tanh(action[..., 1]),
        }

    def evaluate_actions(
        self,
        state: torch.Tensor,
        timestep: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_prob, entropy)`` of ``actions`` under the current policy."""
        distribution = self._distribution(self._encode(state, timestep))
        return distribution.log_prob(actions).sum(-1), distribution.entropy().sum(-1)

    def state_value(self, state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """Critic estimate of the state value."""
        return self.value_net(self._encode(state, timestep)).squeeze(-1)

    # ------------------------------------------------------------------ #
    def update(self, transitions: List[Transition], epochs: int = 4) -> Dict[str, float]:
        """Run PPO updates on a list of transitions; returns mean losses."""
        if not transitions:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        batch_size = transitions[0].state.shape[0]

        states = torch.cat([t.state.detach() for t in transitions], dim=0)
        actions = torch.cat([t.action.detach() for t in transitions], dim=0)
        old_log_probs = torch.cat([t.log_prob.detach() for t in transitions], dim=0)
        timesteps = torch.cat(
            [
                torch.full((t.state.shape[0],), t.timestep, dtype=torch.long)
                for t in transitions
            ]
        )

        reward_grid = torch.stack([t.reward.detach().view(batch_size) for t in transitions])
        returns = self._discounted_returns(reward_grid).reshape(-1).to(self.device)
        advantages = returns - self.state_value(states, timesteps).detach()
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = torch.zeros((), device=self.device)
        value_loss = torch.zeros((), device=self.device)
        for _ in range(epochs):
            new_log_probs, entropy = self.evaluate_actions(states, timesteps, actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            value_loss = nn.functional.mse_loss(self.state_value(states, timesteps), returns)
            loss = policy_loss + 0.5 * value_loss - 1e-3 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.policy_net.parameters()) + list(self.value_net.parameters()), 0.5
            )
            self.optimizer.step()

        return {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
        }

    def _discounted_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """``(steps, batch)`` rewards -> discounted returns of the same shape."""
        returns = torch.zeros_like(rewards)
        running = torch.zeros(rewards.shape[1:], dtype=rewards.dtype, device=rewards.device)
        for step in reversed(range(rewards.shape[0])):
            running = rewards[step] + self.gamma * running
            returns[step] = running
        return returns


def rollout(
    env: RecommendationEnv,
    agent: PPOAgent,
    users: torch.Tensor,
    max_steps: Optional[int] = None,
    store_transitions: bool = False,
) -> Tuple[torch.Tensor, List[Transition]]:
    """Run one denoising episode; returns the final scores (and transitions)."""
    steps = max_steps if max_steps is not None else env.max_steps
    state = env.reset(users)
    transitions: List[Transition] = []

    for step in range(steps):
        timestep = torch.full((state.shape[0],), env.current_step, dtype=torch.long)
        action = agent.get_action(state, timestep)
        next_state, reward, done, _ = env.step(action)
        if store_transitions:
            transitions.append(
                Transition(
                    state=state.detach().cpu(),
                    action=action["action"].detach().cpu(),
                    log_prob=action["log_prob"].detach().cpu(),
                    reward=reward.detach().cpu(),
                    timestep=step,
                )
            )
        state = next_state

    return state, transitions


__all__ = ["PPOAgent", "RecommendationEnv", "Transition", "rollout"]
