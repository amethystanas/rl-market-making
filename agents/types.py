"""Small value types shared by strategies, brokers and the trading loop."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Signal:
    """A strategy decision. There are three kinds:

    * ``Signal.buy()`` / ``Signal.sell()``: send one market order (taker strategies)
    * ``Signal.quote(bid, ask)``: replace the resting passive quotes (market maker)
    * ``Signal.hold()``: do nothing and keep any resting quotes

    Strategies only choose direction or prices. The trading loop sizes every
    order to one fixed lot, so all strategies are sized the same way.
    """

    side: Optional[Side] = None
    reason: str = ""
    quotes: Optional[tuple[float, float]] = None     # (bid, ask)
    details: Optional[dict] = None                   # extra data for the dashboard

    @staticmethod
    def buy(reason: str = "", details: Optional[dict] = None) -> "Signal":
        return Signal(Side.BUY, reason, details=details)

    @staticmethod
    def sell(reason: str = "", details: Optional[dict] = None) -> "Signal":
        return Signal(Side.SELL, reason, details=details)

    @staticmethod
    def quote(bid: float, ask: float, reason: str = "", details: Optional[dict] = None) -> "Signal":
        return Signal(None, reason, (bid, ask), details)

    @staticmethod
    def hold(reason: str = "", details: Optional[dict] = None) -> "Signal":
        return Signal(None, reason, details=details)

    @property
    def kind(self) -> str:
        if self.quotes is not None:
            return "QUOTE"
        return self.side.value if self.side is not None else "HOLD"


@dataclass(frozen=True)
class MarketSnapshot:
    """Top of book, as exported by the C++ ``LiveMarketData`` process."""

    symbol: str
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @classmethod
    def from_feed_json(cls, state: dict) -> Optional["MarketSnapshot"]:
        """Parse ``market_state.json``. The C++ engine stores prices in cents."""
        try:
            bid = float(state["bid_price"]) / 100.0
            ask = float(state["ask_price"]) / 100.0
        except (KeyError, TypeError, ValueError):
            return None
        if bid <= 0 or ask <= 0:
            return None
        return cls(symbol=state.get("symbol", ""), bid=bid, ask=ask)
