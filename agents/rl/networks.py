"""Actor and critic networks for A2C, plus checkpoint helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from agents.rl.env import EnvConfig


def _mlp_body(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.ReLU(),
    )


class Actor(nn.Module):
    """Policy π_θ(a | s): state → logits over {HOLD, BUY, SELL}."""

    def __init__(self, obs_size: int, n_actions: int = 3, hidden: int = 512):
        super().__init__()
        self.body = _mlp_body(obs_size, hidden)
        self.head = nn.Linear(hidden // 2, n_actions)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(obs))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self(obs))


class Critic(nn.Module):
    """State-value function V_φ(s): state → scalar."""

    def __init__(self, obs_size: int, hidden: int = 512):
        super().__init__()
        self.body = _mlp_body(obs_size, hidden)
        self.head = nn.Linear(hidden // 2, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(obs)).squeeze(-1)


def save_checkpoint(path: str | Path, actor: Actor, critic: Critic,
                    env_config: EnvConfig, volume_threshold: float, extra: dict | None = None):
    """Save both networks *and* the feature settings they were trained with."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "env_config": asdict(env_config),
        "volume_threshold": volume_threshold,
        "extra": extra or {},
    }, path)


def load_checkpoint(path: str | Path, device: str = "cpu"):
    """Return ``(actor, critic, env_config, volume_threshold, extra)``."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = EnvConfig(**ckpt["env_config"])
    obs_size = 4 * cfg.window + 1
    actor, critic = Actor(obs_size), Critic(obs_size)
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    actor.eval()
    critic.eval()
    return actor, critic, cfg, ckpt["volume_threshold"], ckpt.get("extra", {})
