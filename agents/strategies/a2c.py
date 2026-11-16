"""Live A2C market maker.

The strategy rebuilds, from live trades, exactly the inputs the network was
trained on:

    trades -> VolumeBarBuilder -> last ``window`` bars -> make_observation()

and re-quotes once per completed volume bar, like one environment step during
training. Between bars it keeps its resting quotes.
"""

from __future__ import annotations

from collections import deque

import torch

from agents.rl.env import MM_ACTION_NAMES, quote_prices
from agents.rl.features import Bar, BarSeries, VolumeBarBuilder, make_observation
from agents.rl.networks import load_checkpoint
from agents.strategies.base import Strategy
from agents.types import MarketSnapshot, Signal


class A2CMarketMaker(Strategy):
    name = "a2c"

    def __init__(self, trade_feed, model_path: str = "models/a2c.pt", warm_up: bool = True):
        self.actor, self.critic, self.cfg, threshold, _ = load_checkpoint(model_path)
        self.feed = trade_feed
        self.builder = VolumeBarBuilder(threshold)
        self.bars: deque[Bar] = deque(maxlen=self.cfg.window)
        if warm_up:
            self._ingest(self.feed.history(min_volume=threshold * (self.cfg.window + 1)))

    def _ingest(self, trades) -> int:
        new_bars = 0
        for price, qty in trades:
            bar = self.builder.add_trade(price, qty)
            if bar is not None:
                self.bars.append(bar)
                new_bars += 1
        return new_bars

    def on_market_data(self, snapshot: MarketSnapshot, inventory: int) -> Signal:
        if self._ingest(self.feed.poll()) == 0:
            return Signal.hold("waiting for the next volume bar")
        if len(self.bars) < self.cfg.window:
            return Signal.hold(f"warming up ({len(self.bars)}/{self.cfg.window} bars)")

        obs = make_observation(BarSeries.from_bars(self.bars), self.cfg.window, self.cfg.window,
                               min(inventory, self.cfg.max_inventory), self.cfg.max_inventory)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs).unsqueeze(0)
            probs = torch.softmax(self.actor(obs_t), dim=-1)[0]
            value = float(self.critic(obs_t))
        action = int(probs.argmax())

        bid, ask = quote_prices(self.bars[-1].close, action, self.cfg)
        details = {"action": MM_ACTION_NAMES[action], "value": value,
                   "probabilities": dict(zip(MM_ACTION_NAMES, map(float, probs))),
                   "bar_close": self.bars[-1].close}
        return Signal.quote(bid, ask, f"{MM_ACTION_NAMES[action]} (p={probs[action]:.2f}, V={value:+.2f})",
                            details)
