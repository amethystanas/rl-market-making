"""Minimal signed REST client for the Binance spot testnet."""

from __future__ import annotations

import hashlib
import hmac
import time
from decimal import ROUND_DOWN, Decimal
from urllib.parse import urlencode

import requests

TESTNET_URL = "https://testnet.binance.vision"


class BinanceError(RuntimeError):
    pass


class BinanceTestnetClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = TESTNET_URL):
        self.base_url = base_url
        self.api_secret = api_secret.encode()
        self.session = requests.Session()
        self.session.headers["X-MBX-APIKEY"] = api_key
        self._time_offset_ms = 0
        self._sync_time()

    # -- plumbing -----------------------------------------------------------

    def _sync_time(self) -> None:
        server_ms = self._request("GET", "/api/v3/time")["serverTime"]
        self._time_offset_ms = server_ms - int(time.time() * 1000)

    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False):
        params = dict(params or {})
        if signed:
            params["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
            params["recvWindow"] = 10_000
            query = urlencode(params)
            params["signature"] = hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()
        response = self.session.request(method, self.base_url + path, params=params, timeout=10)
        if response.status_code != 200:
            raise BinanceError(f"{method} {path} -> {response.status_code}: {response.text}")
        return response.json()

    # -- market info -----------------------------------------------------------

    def symbol_filters(self, symbol: str) -> dict:
        """Tick size, lot step and minimum notional of a symbol."""
        info = self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol})["symbols"][0]
        filters = {f["filterType"]: f for f in info["filters"]}
        return {
            "tick": Decimal(filters["PRICE_FILTER"]["tickSize"]).normalize(),
            "step": Decimal(filters["LOT_SIZE"]["stepSize"]).normalize(),
            "min_notional": float(filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {}))
                                  .get("minNotional", 0)),
        }

    @staticmethod
    def round_to(value: float, increment: Decimal) -> str:
        return str(Decimal(str(value)).quantize(increment, rounding=ROUND_DOWN))

    # -- account -------------------------------------------------------------

    def balances(self) -> dict[str, float]:
        account = self._request("GET", "/api/v3/account", signed=True)
        return {b["asset"]: float(b["free"]) + float(b["locked"]) for b in account["balances"]}

    # -- orders ----------------------------------------------------------------

    def market_order(self, symbol: str, side: str, quantity: str) -> dict:
        return self._request("POST", "/api/v3/order", {
            "symbol": symbol, "side": side, "type": "MARKET", "quantity": quantity,
            "newOrderRespType": "FULL"}, signed=True)

    def limit_order(self, symbol: str, side: str, quantity: str, price: str) -> dict:
        return self._request("POST", "/api/v3/order", {
            "symbol": symbol, "side": side, "type": "LIMIT", "timeInForce": "GTC",
            "quantity": quantity, "price": price}, signed=True)

    def order(self, symbol: str, order_id: int) -> dict:
        return self._request("GET", "/api/v3/order", {"symbol": symbol, "orderId": order_id}, signed=True)

    def cancel_all(self, symbol: str) -> None:
        try:
            self._request("DELETE", "/api/v3/openOrders", {"symbol": symbol}, signed=True)
        except BinanceError as e:      # -2011: no open orders to cancel
            if "-2011" not in str(e):
                raise
