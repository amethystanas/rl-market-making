"""Tests for the rule strategies, the paper broker and the trading loop."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.agent import TradingAgent
from agents.brokers import PaperBroker
from agents.strategies.base import Strategy
from agents.strategies.rules import (BollingerBandsStrategy, MeanReversionStrategy,
                                     RandomStrategy, RSIStrategy)
from agents.types import MarketSnapshot, Side, Signal


def feed(strategy, prices):
    signal = None
    for p in prices:
        signal = strategy.on_market_data(MarketSnapshot("X", p, p), inventory=0)
    return signal


class RuleStrategyTests(unittest.TestCase):
    def test_warm_up_holds(self):
        signal = feed(RSIStrategy(), [100] * 5)
        self.assertIsNone(signal.side)
        self.assertIn("warming up", signal.reason)

    def test_mean_reversion(self):
        self.assertEqual(feed(MeanReversionStrategy(), [100] * 9 + [98]).side, Side.BUY)
        self.assertEqual(feed(MeanReversionStrategy(), [100] * 9 + [102]).side, Side.SELL)
        self.assertIsNone(feed(MeanReversionStrategy(), [100] * 10).side)

    def test_rsi(self):
        self.assertEqual(feed(RSIStrategy(), [100 - i for i in range(15)]).side, Side.BUY)
        self.assertEqual(feed(RSIStrategy(), [100 + i for i in range(15)]).side, Side.SELL)

    def test_bollinger(self):
        base = [100, 101] * 10
        self.assertEqual(feed(BollingerBandsStrategy(), base[:-1] + [90]).side, Side.BUY)
        self.assertEqual(feed(BollingerBandsStrategy(), base[:-1] + [110]).side, Side.SELL)

    def test_random_is_reproducible(self):
        a = [feed(RandomStrategy(seed=3), [1]).side for _ in range(1)]
        b = [feed(RandomStrategy(seed=3), [1]).side for _ in range(1)]
        self.assertEqual(a, b)


class PaperBrokerTests(unittest.TestCase):
    def test_market_orders_and_fees(self):
        broker = PaperBroker(cash=1000, fee=0.001)
        snap = MarketSnapshot("X", 99.0, 101.0)
        self.assertIsNone(broker.market_order(Side.SELL, 1, snap))      # nothing to sell
        broker.market_order(Side.BUY, 1, snap)
        self.assertAlmostEqual(broker.cash, 1000 - 101 - 0.101)
        self.assertEqual(broker.position, 1)

    def test_quotes_fill_when_traded_through(self):
        broker = PaperBroker(cash=1000, fee=0.0)
        broker.set_quotes(99.0, None, 1)
        self.assertEqual(broker.update(MarketSnapshot("X", 98.9, 99.0)), [])   # touch only
        fills = broker.update(MarketSnapshot("X", 98.8, 98.9))
        self.assertEqual([(f.side, f.price) for f in fills], [(Side.BUY, 99.0)])
        self.assertEqual(broker.position, 1)


class ScriptedStrategy(Strategy):
    name = "scripted"

    def __init__(self, signals):
        self.signals = list(signals)

    def on_market_data(self, snapshot, inventory):
        return self.signals.pop(0)


class TradingAgentTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("agents.agent.display")    # keep test output quiet
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_agent(self, signals, max_lots=2):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "market_state.json"
            path.write_text(json.dumps({"symbol": "SOLUSDT", "bid_price": 10000, "ask_price": 10002}))
            agent = TradingAgent(ScriptedStrategy(signals), PaperBroker(fee=0.0), "SOLUSDT",
                                 market_state=path, lot_size=0.5, max_lots=max_lots)
            for _ in signals:
                self.assertTrue(agent.step())
        return agent

    def test_reads_cpp_snapshot_in_cents(self):
        agent = self.run_agent([Signal.hold()])
        self.assertEqual(agent.read_snapshot(), None)   # temp file is gone after the run
        snap = MarketSnapshot.from_feed_json({"symbol": "S", "bid_price": 10000, "ask_price": 10002})
        self.assertEqual((snap.bid, snap.ask), (100.0, 100.02))

    def test_position_limits(self):
        agent = self.run_agent([Signal.buy()] * 3 + [Signal.sell()] * 3, max_lots=2)
        self.assertEqual(agent.trades, 4)                # 3rd buy and 3rd sell are refused
        self.assertEqual(agent.inventory_lots, 0)

    def test_quotes_respect_inventory(self):
        agent = self.run_agent([Signal.quote(99.0, 101.0)])
        self.assertEqual(agent.broker.bid, (99.0, 0.5))
        self.assertIsNone(agent.broker.ask)              # no inventory -> no ask


if __name__ == "__main__":
    unittest.main()
