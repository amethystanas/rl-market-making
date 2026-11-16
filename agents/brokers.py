"""Order execution backends: a local paper-trading simulator and the Binance spot testnet."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from agents.types import MarketSnapshot, Side


@dataclass(frozen=True)
class Fill:
    side: Side
    price: float
    quantity: float


class Broker(ABC):
    """Tracks the position (in base asset) and cash (in quote asset)."""

    position: float = 0.0
    cash: float = 0.0

    @abstractmethod
    def market_order(self, side: Side, quantity: float, snapshot: MarketSnapshot) -> Fill | None:
        ...

    @abstractmethod
    def set_quotes(self, bid: float | None, ask: float | None, quantity: float) -> None:
        """Replace all resting quotes. ``None`` leaves that side empty."""

    @abstractmethod
    def update(self, snapshot: MarketSnapshot) -> list[Fill]:
        """Refresh state from the market and return quotes filled since the last call."""

    @abstractmethod
    def close(self) -> None:
        """Cancel everything still resting."""


class PaperBroker(Broker):
    """Local simulation: market orders fill at the touch; quotes fill when traded through.

    A resting bid is filled when the best ask drops below it, and a resting
    ask when the best bid rises above it. This is the same conservative rule
    as the backtest (the price must move *through* the quote).
    """

    def __init__(self, cash: float = 10_000.0, fee: float = 0.0002):
        self.cash, self.position, self.fee = cash, 0.0, fee
        self.bid: tuple[float, float] | None = None   # (price, quantity)
        self.ask: tuple[float, float] | None = None

    def _fill(self, side: Side, price: float, quantity: float) -> Fill:
        sign = 1 if side is Side.BUY else -1
        self.position += sign * quantity
        self.cash -= sign * price * quantity + self.fee * price * quantity
        return Fill(side, price, quantity)

    def market_order(self, side, quantity, snapshot):
        if side is Side.SELL and quantity > self.position + 1e-12:
            return None
        price = snapshot.ask if side is Side.BUY else snapshot.bid
        return self._fill(side, price, quantity)

    def set_quotes(self, bid, ask, quantity):
        self.bid = (bid, quantity) if bid is not None else None
        self.ask = (ask, quantity) if ask is not None else None

    def update(self, snapshot):
        fills = []
        if self.bid and snapshot.ask < self.bid[0]:
            fills.append(self._fill(Side.BUY, *self.bid))
            self.bid = None
        if self.ask and snapshot.bid > self.ask[0] and self.position >= self.ask[1] - 1e-12:
            fills.append(self._fill(Side.SELL, *self.ask))
            self.ask = None
        return fills

    def close(self):
        self.bid = self.ask = None


class BinanceTestnetBroker(Broker):
    """Real orders on the Binance spot testnet (https://testnet.binance.vision)."""

    def __init__(self, client, symbol: str, base_asset: str, quote_asset: str = "USDT"):
        self.client, self.symbol = client, symbol
        self.base_asset, self.quote_asset = base_asset, quote_asset
        self.filters = client.symbol_filters(symbol)
        self.open_orders: dict[int, Side] = {}
        self._refresh_balances()

    def _refresh_balances(self):
        balances = self.client.balances()
        self.position = balances.get(self.base_asset, 0.0)
        self.cash = balances.get(self.quote_asset, 0.0)

    def _qty(self, quantity: float) -> str:
        return self.client.round_to(quantity, self.filters["step"])

    def _price(self, price: float) -> str:
        return self.client.round_to(price, self.filters["tick"])

    def market_order(self, side, quantity, snapshot):
        if quantity * snapshot.mid < self.filters["min_notional"]:
            raise ValueError(f"order below the minimum notional ({self.filters['min_notional']})")
        resp = self.client.market_order(self.symbol, side.value, self._qty(quantity))
        self._refresh_balances()
        fills = resp.get("fills", [])
        qty = sum(float(f["qty"]) for f in fills)
        if qty == 0:
            return None
        price = sum(float(f["price"]) * float(f["qty"]) for f in fills) / qty
        return Fill(side, price, qty)

    def set_quotes(self, bid, ask, quantity):
        self.client.cancel_all(self.symbol)
        self.open_orders.clear()
        for side, price in ((Side.BUY, bid), (Side.SELL, ask)):
            if price is not None:
                resp = self.client.limit_order(self.symbol, side.value, self._qty(quantity), self._price(price))
                self.open_orders[resp["orderId"]] = side

    def update(self, snapshot):
        fills = []
        for order_id, side in list(self.open_orders.items()):
            order = self.client.order(self.symbol, order_id)
            if order["status"] in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                del self.open_orders[order_id]
                if order["status"] == "FILLED":
                    fills.append(Fill(side, float(order["price"]), float(order["executedQty"])))
        # Always resync: partial fills of quotes cancelled later still change the balances.
        self._refresh_balances()
        return fills

    def close(self):
        self.client.cancel_all(self.symbol)
        self.open_orders.clear()
