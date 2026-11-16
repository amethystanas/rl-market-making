"""Common interface implemented by every strategy (rule-based or RL)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque

from agents.types import MarketSnapshot, Signal


class Strategy(ABC):
    """A strategy turns the latest market snapshot into a buy / sell / hold signal.

    ``inventory`` is the current position expressed in lots (number of
    fixed-size trades held), which lets inventory-aware strategies such as A2C
    behave the same way in training, backtesting and live trading.
    """

    name: str = "strategy"

    @abstractmethod
    def on_market_data(self, snapshot: MarketSnapshot, inventory: int) -> Signal:
        ...


class PriceWindowStrategy(Strategy):
    """Base class for indicator strategies working on a rolling window of mid prices."""

    def __init__(self, window: int):
        self.window = window
        self.prices: Deque[float] = deque(maxlen=window)

    def on_market_data(self, snapshot: MarketSnapshot, inventory: int) -> Signal:
        self.prices.append(snapshot.mid)
        if len(self.prices) < self.window:
            return Signal.hold(f"warming up ({len(self.prices)}/{self.window} prices)")
        return self.decide(list(self.prices))

    @abstractmethod
    def decide(self, prices: list[float]) -> Signal:
        ...
