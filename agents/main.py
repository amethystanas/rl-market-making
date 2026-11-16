"""Run a trading agent on the live market feed.

    python -m agents.main --strategy a2c                 # paper trading (default)
    python -m agents.main --strategy rsi --broker testnet

The C++ ``LiveMarketData`` process must be running and writing
``data/market_state.json``. Every decision is also logged to
``data/agent_events.jsonl`` for the dashboard (``python -m dashboard``).
Testnet mode reads the API keys from the ``BINANCE_TESTNET_API_KEY`` /
``BINANCE_TESTNET_API_SECRET`` environment variables.
"""

from __future__ import annotations

import argparse
import os
import sys

from agents.agent import TradingAgent
from agents.brokers import BinanceTestnetBroker, PaperBroker
from agents.events import EventLog

STRATEGIES = ("a2c", "mean_reversion", "rsi", "bollinger", "random")


def build_strategy(name: str, symbol: str, model: str):
    from agents.strategies import rules
    if name == "a2c":
        from agents.exchange.trade_feed import BinanceTradeFeed
        from agents.strategies.a2c import A2CMarketMaker
        print("Loading A2C model and warming up volume bars from recent trades...")
        return A2CMarketMaker(BinanceTradeFeed(symbol), model_path=model)
    return {
        "mean_reversion": rules.MeanReversionStrategy,
        "rsi": rules.RSIStrategy,
        "bollinger": rules.BollingerBandsStrategy,
        "random": rules.RandomStrategy,
    }[name]()


def build_broker(name: str, symbol: str, base_asset: str):
    if name == "paper":
        return PaperBroker()
    from agents.exchange.binance_client import BinanceTestnetClient
    key = os.environ.get("BINANCE_TESTNET_API_KEY")
    secret = os.environ.get("BINANCE_TESTNET_API_SECRET")
    if not key or not secret:
        sys.exit("Set BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET "
                 "(keys from https://testnet.binance.vision).")
    return BinanceTestnetBroker(BinanceTestnetClient(key, secret), symbol, base_asset)


def main() -> None:
    p = argparse.ArgumentParser(description="Run a trading agent on the live market feed.")
    p.add_argument("--strategy", choices=STRATEGIES, default="a2c")
    p.add_argument("--broker", choices=("paper", "testnet"), default="paper")
    p.add_argument("--symbol", default="SOLUSDT")
    p.add_argument("--base-asset", default="SOL")
    p.add_argument("--lot-size", type=float, default=0.1, help="base-asset quantity per trade")
    p.add_argument("--max-lots", type=int, default=10)
    p.add_argument("--interval", type=float, default=2.0, help="seconds between decisions")
    p.add_argument("--market-state", default="data/market_state.json")
    p.add_argument("--model", default="models/a2c.pt")
    p.add_argument("--events", default="data/agent_events.jsonl",
                   help="decision log read by the dashboard ('' to disable)")
    args = p.parse_args()

    agent = TradingAgent(
        strategy=build_strategy(args.strategy, args.symbol, args.model),
        broker=build_broker(args.broker, args.symbol, args.base_asset),
        symbol=args.symbol, market_state=args.market_state,
        lot_size=args.lot_size, max_lots=args.max_lots, interval=args.interval,
        events=EventLog(args.events) if args.events else None,
    )
    agent.run()


if __name__ == "__main__":
    main()
