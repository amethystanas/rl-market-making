"""Rule-based baseline strategies built on classic technical indicators."""

from __future__ import annotations

import random
import statistics

from agents.strategies.base import PriceWindowStrategy, Strategy
from agents.types import MarketSnapshot, Signal


class MeanReversionStrategy(PriceWindowStrategy):
    """Buy when the price is below its moving average, sell when it is above."""

    name = "mean_reversion"

    def __init__(self, window: int = 10, threshold_pct: float = 0.05):
        super().__init__(window)
        self.threshold_pct = threshold_pct

    def decide(self, prices: list[float]) -> Signal:
        sma = statistics.fmean(prices)
        deviation = (prices[-1] - sma) / sma * 100
        details = {"indicators": {"SMA": sma, "deviation %": deviation}}
        if deviation < -self.threshold_pct:
            return Signal.buy(f"price {deviation:+.3f}% below SMA {sma:.2f}", details)
        if deviation > self.threshold_pct:
            return Signal.sell(f"price {deviation:+.3f}% above SMA {sma:.2f}", details)
        return Signal.hold(f"deviation {deviation:+.3f}% within ±{self.threshold_pct}%", details)


class RSIStrategy(PriceWindowStrategy):
    """Buy when RSI signals oversold (< 30), sell when overbought (> 70)."""

    name = "rsi"

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(period + 1)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def decide(self, prices: list[float]) -> Signal:
        deltas = [b - a for a, b in zip(prices, prices[1:])]
        avg_gain = sum(d for d in deltas if d > 0) / self.period
        avg_loss = sum(-d for d in deltas if d < 0) / self.period
        if avg_loss == 0:
            rsi = 100.0 if avg_gain > 0 else 50.0
        else:
            rsi = 100 - 100 / (1 + avg_gain / avg_loss)

        details = {"indicators": {"RSI": rsi}}
        if rsi < self.oversold:
            return Signal.buy(f"RSI {rsi:.1f} oversold", details)
        if rsi > self.overbought:
            return Signal.sell(f"RSI {rsi:.1f} overbought", details)
        return Signal.hold(f"RSI {rsi:.1f} neutral", details)


class BollingerBandsStrategy(PriceWindowStrategy):
    """Buy on a touch of the lower band, sell on a touch of the upper band."""

    name = "bollinger"

    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__(period)
        self.num_std = num_std

    def decide(self, prices: list[float]) -> Signal:
        sma = statistics.fmean(prices)
        std = statistics.pstdev(prices)
        lower, upper = sma - self.num_std * std, sma + self.num_std * std
        price = prices[-1]
        details = {"indicators": {"lower band": lower, "SMA": sma, "upper band": upper}}
        if price <= lower:
            return Signal.buy(f"price {price:.2f} <= lower band {lower:.2f}", details)
        if price >= upper:
            return Signal.sell(f"price {price:.2f} >= upper band {upper:.2f}", details)
        return Signal.hold(f"price {price:.2f} inside bands [{lower:.2f}, {upper:.2f}]", details)


class RandomStrategy(Strategy):
    """Sanity-check baseline: buys or sells at random."""

    name = "random"

    def __init__(self, trade_probability: float = 0.1, seed: int | None = None):
        self.trade_probability = trade_probability
        self.rng = random.Random(seed)

    def on_market_data(self, snapshot: MarketSnapshot, inventory: int) -> Signal:
        u = self.rng.random()
        if u < self.trade_probability / 2:
            return Signal.buy("random buy")
        if u < self.trade_probability:
            return Signal.sell("random sell")
        return Signal.hold("random hold")
