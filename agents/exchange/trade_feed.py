"""Public Binance trade stream (no API key needed), used to build live volume bars."""

from __future__ import annotations

from typing import Iterable

import requests

BINANCE_API = "https://api.binance.com"


class BinanceTradeFeed:
    """Polls ``/api/v3/aggTrades`` and returns the trades it has not seen yet.

    These are the same aggregated trades as the historical files used for
    training, so live volume bars match the training bars.
    """

    def __init__(self, symbol: str, base_url: str = BINANCE_API, timeout: float = 5.0):
        self.symbol = symbol
        self.url = f"{base_url}/api/v3/aggTrades"
        self.timeout = timeout
        self.session = requests.Session()
        self.last_id: int | None = None

    def _get(self, **params) -> list[dict]:
        response = self.session.get(self.url, params={"symbol": self.symbol, "limit": 1000, **params},
                                    timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def history(self, min_volume: float, max_pages: int = 30) -> list[tuple[float, float]]:
        """Fetch at least ``min_volume`` of recent trades (oldest first) to warm up the bars."""
        pages = [self._get()]
        volume = sum(float(t["q"]) for t in pages[0])
        while volume < min_volume and len(pages) < max_pages and pages[0]:
            first_id = pages[0][0]["a"]
            page = self._get(fromId=max(0, first_id - 1000))
            page = [t for t in page if t["a"] < first_id]
            if not page:
                break
            pages.insert(0, page)
            volume += sum(float(t["q"]) for t in page)
        trades = [t for page in pages for t in page]
        if trades:
            self.last_id = trades[-1]["a"]
        return self._parse(trades)

    def poll(self) -> list[tuple[float, float]]:
        """New trades since the previous call, as ``(price, quantity)`` tuples."""
        if self.last_id is None:
            trades = self._get(limit=1)
        else:
            trades = self._get(fromId=self.last_id + 1)
        if trades:
            self.last_id = trades[-1]["a"]
        return self._parse(trades)

    @staticmethod
    def _parse(trades: Iterable[dict]) -> list[tuple[float, float]]:
        return [(float(t["p"]), float(t["q"])) for t in trades]
