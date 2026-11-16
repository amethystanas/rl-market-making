"""Backtesting policies on historical bars.

A policy is any ``f(observation, env) -> action``. Market-making policies
(A2C, constant quotes) run in :class:`MarketMakingEnv`; taker policies (rule
strategies, buy & hold) run in :class:`TakerEnv`. Within one environment all
policies share the same execution model, fees and position limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Type

import numpy as np
import torch

from agents.rl.env import BUY, HOLD, SELL, EnvConfig, MarketMakingEnv, TakerEnv, _BarEnv
from agents.rl.features import BarSeries
from agents.strategies.base import Strategy
from agents.types import MarketSnapshot, Side

Policy = Callable[[np.ndarray, _BarEnv], int]


@dataclass
class BacktestResult:
    name: str
    equity: np.ndarray = field(repr=False)    # P&L in % of capital after each step
    fills: int
    exposure: float                           # mean inventory / max_inventory
    actions: np.ndarray = field(repr=False)   # action counts

    @property
    def total_return(self) -> float:
        return float(self.equity[-1])

    @property
    def max_drawdown(self) -> float:
        peak = np.maximum.accumulate(np.r_[0.0, self.equity])
        return float((peak[1:] - self.equity).max())


def run_backtest(name: str, bars: BarSeries, cfg: EnvConfig, policy: Policy,
                 env_cls: Type[_BarEnv] = MarketMakingEnv) -> BacktestResult:
    """Run ``policy`` once over the whole series, starting flat with no cash.

    Returns are expressed in % of the capital needed to hold the maximum
    position at the initial price.
    """
    env = env_cls(bars, cfg)
    obs = env.reset(start=cfg.window, length=len(bars) - cfg.window)
    capital = cfg.max_inventory * env.price

    equity, inventory, actions, fills, done = [], [], np.zeros(3, int), 0, False
    while not done:
        action = policy(obs, env)
        obs, _, done, info = env.step(action)
        actions[action] += 1
        fills += info["fills"]
        equity.append(100.0 * info["value"] / capital)
        inventory.append(info["inventory"])

    return BacktestResult(name, np.array(equity), fills,
                          float(np.mean(inventory)) / cfg.max_inventory, actions)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def actor_policy(actor: torch.nn.Module) -> Policy:
    """Greedy policy: the most probable action of the A2C actor."""
    def policy(obs: np.ndarray, env: _BarEnv) -> int:
        with torch.no_grad():
            return int(actor(torch.as_tensor(obs).unsqueeze(0)).argmax(-1))
    return policy


def constant_policy(action: int) -> Policy:
    return lambda obs, env: action


def strategy_policy(strategy: Strategy) -> Policy:
    """Adapt a live taker :class:`Strategy` (bar close used as bid and ask)."""
    def policy(obs: np.ndarray, env: _BarEnv) -> int:
        snapshot = MarketSnapshot("BACKTEST", env.price, env.price)
        side = strategy.on_market_data(snapshot, env.inventory).side
        return {Side.BUY: BUY, Side.SELL: SELL}.get(side, HOLD)
    return policy
