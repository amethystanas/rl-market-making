"""Live trading loop: C++ market feed -> strategy -> broker."""

from __future__ import annotations

import json
import time
from pathlib import Path

from agents import display
from agents.brokers import Broker, Fill
from agents.events import NullEventLog
from agents.strategies.base import Strategy
from agents.types import MarketSnapshot, Signal


class TradingAgent:
    """Polls the JSON snapshot written by ``LiveMarketData`` and acts on the strategy's signals.

    Position limits match the training environment: every order is one lot
    of ``lot_size`` and the agent's own inventory stays within
    ``[0, max_lots]`` (long-only, like a spot account).
    """

    def __init__(self, strategy: Strategy, broker: Broker, symbol: str,
                 market_state: str | Path = "data/market_state.json",
                 lot_size: float = 0.1, max_lots: int = 10, interval: float = 2.0,
                 events=None):
        self.strategy, self.broker, self.symbol = strategy, broker, symbol
        self.events = events or NullEventLog()
        self.market_state = Path(market_state)
        self.lot_size, self.max_lots, self.interval = lot_size, max_lots, interval
        # Testnet accounts start with a balance: only count what this agent traded.
        self.start_position, self.start_cash = broker.position, broker.cash
        self.trades = 0
        self.quotes: tuple[float | None, float | None] = (None, None)

    @property
    def inventory_lots(self) -> int:
        return round((self.broker.position - self.start_position) / self.lot_size)

    def read_snapshot(self) -> MarketSnapshot | None:
        try:
            snapshot = MarketSnapshot.from_feed_json(json.loads(self.market_state.read_text()))
        except (OSError, ValueError):
            return None
        if snapshot is None or (snapshot.symbol and snapshot.symbol != self.symbol):
            return None
        return snapshot

    def pnl(self, snapshot: MarketSnapshot) -> float:
        return (self.broker.cash - self.start_cash) + \
               (self.broker.position - self.start_position) * snapshot.mid

    def act(self, signal: Signal, snapshot: MarketSnapshot) -> list[Fill]:
        """Execute a signal within the position limits. Returns immediate fills."""
        lots = self.inventory_lots
        if signal.quotes is not None:
            bid, ask = signal.quotes
            bid, ask = (bid if lots < self.max_lots else None), (ask if lots > 0 else None)
            self.broker.set_quotes(bid, ask, self.lot_size)
            self.quotes = (bid, ask)
            display.quotes(bid, ask)
        elif signal.side is not None:
            allowed = lots < self.max_lots if signal.side.value == "BUY" else lots > 0
            if not allowed:
                display.info(f"{signal.side.value} skipped: position limit ({lots}/{self.max_lots} lots)")
                return []
            fill = self.broker.market_order(signal.side, self.lot_size, snapshot)
            if fill:
                return [fill]
        return []

    def _record_fills(self, fills: list[Fill]) -> list[Fill]:
        for fill in fills:
            self.trades += 1
            display.fill(fill)
            # A filled quote is no longer resting (no-op for taker strategies, which never quote).
            bid, ask = self.quotes
            self.quotes = (None, ask) if fill.side.value == "BUY" else (bid, None)
        return fills

    def step(self) -> bool:
        snapshot = self.read_snapshot()
        if snapshot is None:
            return False
        fills = self._record_fills(self.broker.update(snapshot))
        signal = self.strategy.on_market_data(snapshot, self.inventory_lots)
        display.decision(snapshot, signal, self.inventory_lots, self.max_lots, self.pnl(snapshot))
        fills += self._record_fills(self.act(signal, snapshot))

        self.events.write(
            "step", bid=snapshot.bid, ask=snapshot.ask, mid=snapshot.mid,
            signal=signal.kind, reason=signal.reason, details=signal.details,
            quotes={"bid": self.quotes[0], "ask": self.quotes[1]},
            fills=[{"side": f.side.value, "price": f.price, "quantity": f.quantity} for f in fills],
            inventory=self.inventory_lots, pnl=self.pnl(snapshot), trades=self.trades)
        return True

    def run(self) -> None:
        display.banner(self.strategy.name, self.symbol, self.market_state)
        self.events.write("start", strategy=self.strategy.name, symbol=self.symbol,
                          lot_size=self.lot_size, max_lots=self.max_lots)
        last = None
        try:
            while True:
                if not self.step():
                    display.info(f"waiting for {self.market_state} (is LiveMarketData running?)")
                last = self.read_snapshot() or last
                time.sleep(self.interval)
        except KeyboardInterrupt:
            pass
        finally:
            self.broker.close()
            if last is not None:
                display.summary(self.trades, self.inventory_lots, self.pnl(last))
