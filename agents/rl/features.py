"""Volume bars and observation vectors.

This module is the single source of truth for what the A2C agent sees.
It is used by the training environment, the backtest *and* the live
strategy, so the network always receives inputs built the exact same way.

* A **volume bar** closes every time ``threshold`` units of the base asset
  have traded (e.g. 1000 SOL). Compared with time bars, volume bars sample
  more often when the market is active and give returns that are closer to
  i.i.d., which makes them a common choice for ML on market data.
* An **observation** is the last ``window`` bars normalised relative to the
  latest close, plus the current inventory: ``4 * window + 1`` features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BarSeries:
    """Column-oriented OHLCV arrays (fast slicing for training)."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray

    def __len__(self) -> int:
        return len(self.close)

    def slice(self, start: int, stop: int) -> "BarSeries":
        return BarSeries(*(a[start:stop] for a in
                           (self.open, self.high, self.low, self.close, self.volume)))

    def mirrored(self) -> "BarSeries":
        """Reflect the price path in log space around its first close.

        Up-moves become down-moves (``p -> p0² / p``, so high and low swap)
        while volumes and the size of each move are unchanged. Training on
        both paths removes the trend of a single month from the data, so the
        agent cannot just learn "the market goes down".
        """
        k = self.close[0] ** 2
        return BarSeries(k / self.open, k / self.low, k / self.high, k / self.close,
                         self.volume.copy())

    @classmethod
    def from_bars(cls, bars: Iterable[Bar]) -> "BarSeries":
        bars = list(bars)
        cols = [np.array([getattr(b, f) for b in bars], dtype=np.float64)
                for f in ("open", "high", "low", "close", "volume")]
        return cls(*cols)


# ---------------------------------------------------------------------------
# Volume bars
# ---------------------------------------------------------------------------

def volume_bars(prices: np.ndarray, quantities: np.ndarray, threshold: float) -> BarSeries:
    """Aggregate a stream of trades into volume bars (vectorised, for offline data).

    Trade ``i`` belongs to bar ``floor(cumulative_volume_i / threshold)``.
    The last, still-open bar is dropped.
    """
    prices = np.asarray(prices, dtype=np.float64)
    quantities = np.asarray(quantities, dtype=np.float64)
    bar_ids = np.floor(np.cumsum(quantities) / threshold).astype(np.int64)

    if len(prices) == 0:
        return BarSeries(*(np.empty(0) for _ in range(5)))
    starts = np.flatnonzero(np.r_[True, bar_ids[1:] != bar_ids[:-1]])
    ends = np.r_[starts[1:], len(prices)]

    # [:-1] drops the last bar, which is still open (incomplete).
    return BarSeries(
        open=prices[starts][:-1],
        high=np.maximum.reduceat(prices, starts)[:-1],
        low=np.minimum.reduceat(prices, starts)[:-1],
        close=prices[ends - 1][:-1],
        volume=np.add.reduceat(quantities, starts)[:-1],
    )


class VolumeBarBuilder:
    """Incremental version of :func:`volume_bars`, used on live trades."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._cum_volume = 0.0
        self._bar_id: int | None = None
        self._open = self._high = self._low = self._close = self._volume = 0.0

    def add_trade(self, price: float, quantity: float) -> Bar | None:
        """Add one trade. Returns the bar that this trade closed, if any."""
        self._cum_volume += quantity
        bar_id = int(np.floor(self._cum_volume / self.threshold))   # same formula as volume_bars()

        completed = None
        if self._bar_id is not None and bar_id != self._bar_id:
            completed = Bar(self._open, self._high, self._low, self._close, self._volume)
            self._bar_id = None

        if self._bar_id is None:
            self._bar_id = bar_id
            self._open = self._high = self._low = price
            self._volume = 0.0
        self._high = max(self._high, price)
        self._low = min(self._low, price)
        self._close = price
        self._volume += quantity
        return completed


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

def observation_size(window: int) -> int:
    return 4 * window + 1


def make_observation(bars: BarSeries, end: int, window: int,
                     inventory: int, max_inventory: int) -> np.ndarray:
    """Observation built from bars ``[end - window, end)`` and the inventory.

    Prices are expressed as returns relative to the latest close and volume
    relative to its window mean, so the features are scale-free and the same
    network works whatever the absolute price level.
    """
    if end < window:
        raise ValueError(f"need at least {window} bars, got {end}")
    sl = slice(end - window, end)
    base = bars.close[end - 1]
    volume = bars.volume[sl]
    return np.concatenate([
        bars.close[sl] / base - 1.0,
        bars.high[sl] / base - 1.0,
        bars.low[sl] / base - 1.0,
        volume / (volume.mean() + 1e-12) - 1.0,
        [inventory / max_inventory],
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_agg_trades(csv_path: str | Path, threshold: float) -> BarSeries:
    """Load a Binance ``aggTrades`` CSV (data.binance.vision) as volume bars.

    The converted bars are cached next to the CSV, so only the first call is slow.
    """
    import pandas as pd

    csv_path = Path(csv_path)
    cache = csv_path.with_name(f"{csv_path.stem}.bars{int(threshold)}.npz")
    if cache.exists() and cache.stat().st_mtime >= csv_path.stat().st_mtime:
        data = np.load(cache)
        return BarSeries(*(data[k] for k in ("open", "high", "low", "close", "volume")))

    # Columns: agg_id, price, quantity, first_id, last_id, timestamp, is_buyer_maker, best_match
    df = pd.read_csv(csv_path, header=None, usecols=[1, 2], names=["price", "qty"])
    if not np.issubdtype(df["price"].dtype, np.number):   # file shipped with a header row
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
    bars = volume_bars(df["price"].to_numpy(), df["qty"].to_numpy(), threshold)
    np.savez(cache, open=bars.open, high=bars.high, low=bars.low,
             close=bars.close, volume=bars.volume)
    return bars
