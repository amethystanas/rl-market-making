"""Trading environments on historical volume bars.

Two execution models share the same accounting:

* :class:`MarketMakingEnv`: used to train the A2C agent. Each step the agent
  posts a passive bid and ask around the last close and chooses *how to skew
  them*: ``NEUTRAL``, ``LEAN_BUY`` (tight bid / wide ask, accumulate) or
  ``LEAN_SELL`` (tight ask / wide bid, offload inventory). A quote is filled
  during the next bar only if the price *trades through* it by at least one
  tick, which is a conservative proxy for queue position.
* :class:`TakerEnv`: market orders at the bar close (``HOLD / BUY / SELL``).
  Used to backtest the rule-based strategies.

Both are long-only with an inventory between 0 and ``max_inventory`` lots,
like a Binance spot account. Quotes or orders that would break that limit
are not sent. The reward is the mark-to-market P&L net of fees, minus a
quadratic inventory penalty, in % of one lot's value:

    r_t = 100 * (ΔV_t - λ · I_t² · P_t) / P_t ,    V = cash + I · P
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.rl.features import BarSeries, make_observation, observation_size

# Market-making actions: (bid offset, ask offset) as a fraction of the reference price.
NEUTRAL, LEAN_SELL, LEAN_BUY = 0, 1, 2
MM_ACTION_NAMES = ("NEUTRAL", "LEAN_SELL", "LEAN_BUY")
DEFAULT_QUOTE_OFFSETS = (
    (0.00125, 0.00125),   # NEUTRAL:   symmetric, 12.5 bps each side
    (0.00300, 0.00050),   # LEAN_SELL: wide bid, tight ask -> offload inventory
    (0.00050, 0.00300),   # LEAN_BUY:  tight bid, wide ask -> accumulate inventory
)

# Taker actions
HOLD, BUY, SELL = 0, 1, 2
TAKER_ACTION_NAMES = ("HOLD", "BUY", "SELL")


@dataclass
class EnvConfig:
    window: int = 30
    max_inventory: int = 10
    cost: float = 0.0002              # fee per fill, as a fraction of notional
    inventory_penalty: float = 1e-5   # λ
    episode_length: int = 200
    tick: float = 0.01                # SOLUSDT price tick
    quote_offsets: tuple = DEFAULT_QUOTE_OFFSETS

    def __post_init__(self):          # lists after a JSON / checkpoint round trip
        self.quote_offsets = tuple(tuple(o) for o in self.quote_offsets)


def quote_prices(reference: float, action: int, cfg: EnvConfig) -> tuple[float, float]:
    """Bid / ask prices for a market-making action, rounded to the tick outward."""
    bid_off, ask_off = cfg.quote_offsets[action]
    bid = np.floor(reference * (1 - bid_off) / cfg.tick) * cfg.tick
    ask = np.ceil(reference * (1 + ask_off) / cfg.tick) * cfg.tick
    return round(float(bid), 8), round(float(ask), 8)


class _BarEnv:
    """Shared state and accounting. Bar ``t - 1`` is the latest completed bar."""

    n_actions = 3

    def __init__(self, bars: BarSeries, config: EnvConfig | None = None, seed: int | None = None):
        self.cfg = config or EnvConfig()
        if len(bars) < self.cfg.window + 2:
            raise ValueError("not enough bars for a single step")
        self.bars = bars
        self.rng = np.random.default_rng(seed)
        self.obs_size = observation_size(self.cfg.window)
        self.t, self.inventory, self.cash, self.steps, self.max_steps = self.cfg.window, 0, 0.0, 0, 0

    @property
    def price(self) -> float:
        return float(self.bars.close[self.t - 1])

    def observation(self) -> np.ndarray:
        return make_observation(self.bars, self.t, self.cfg.window,
                                self.inventory, self.cfg.max_inventory)

    def portfolio_value(self) -> float:
        return self.cash + self.inventory * self.price

    def reset(self, start: int | None = None, length: int | None = None) -> np.ndarray:
        """Start flat at bar ``start`` (random if None) for ``length`` steps."""
        w, n = self.cfg.window, len(self.bars)
        length = length or self.cfg.episode_length
        if start is None:
            start = int(self.rng.integers(w, max(w + 1, n - length)))
        self.t, self.max_steps = start, min(length, n - start)
        self.inventory, self.cash, self.steps = 0, 0.0, 0
        return self.observation()

    def _buy(self, price: float):
        self.inventory += 1
        self.cash -= price * (1 + self.cfg.cost)

    def _sell(self, price: float):
        self.inventory -= 1
        self.cash += price * (1 - self.cfg.cost)

    def _advance(self, reference: float, value_before: float, fills: int):
        self.t += 1
        self.steps += 1
        pnl = self.portfolio_value() - value_before
        penalty = self.cfg.inventory_penalty * self.inventory ** 2 * self.price
        reward = 100.0 * (pnl - penalty) / reference
        done = self.steps >= self.max_steps
        info = {"fills": fills, "pnl": pnl, "inventory": self.inventory,
                "value": self.portfolio_value()}
        return self.observation(), reward, done, info


class MarketMakingEnv(_BarEnv):
    action_names = MM_ACTION_NAMES

    def step(self, action: int):
        reference, value_before = self.price, self.portfolio_value()
        bid, ask = quote_prices(reference, action, self.cfg)
        # Quotes are only posted if allowed by the inventory *before* the bar.
        post_bid = self.inventory < self.cfg.max_inventory
        post_ask = self.inventory > 0

        nxt, tick, fills = self.t, self.cfg.tick, 0
        if post_bid and self.bars.low[nxt] < bid - tick / 2:
            self._buy(bid)
            fills += 1
        if post_ask and self.bars.high[nxt] > ask + tick / 2:
            self._sell(ask)
            fills += 1
        return self._advance(reference, value_before, fills)


class TakerEnv(_BarEnv):
    action_names = TAKER_ACTION_NAMES

    def step(self, action: int):
        reference, value_before = self.price, self.portfolio_value()
        fills = 0
        if action == BUY and self.inventory < self.cfg.max_inventory:
            self._buy(reference)
            fills = 1
        elif action == SELL and self.inventory > 0:
            self._sell(reference)
            fills = 1
        return self._advance(reference, value_before, fills)
