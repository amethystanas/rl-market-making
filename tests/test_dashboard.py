"""Tests for the event log written by the agent and read by the dashboard."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agents.agent import TradingAgent
from agents.brokers import PaperBroker
from agents.events import EventLog
from agents.strategies.rules import BollingerBandsStrategy
from agents.types import Signal
from dashboard.__main__ import read_events


class EventLogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_incremental_reads_and_reset(self):
        log = EventLog(self.dir / "events.jsonl")
        log.write("start", strategy="x")
        log.write("step", mid=1.0)
        first = read_events(log.path, 0)
        self.assertEqual([e["type"] for e in first["events"]], ["start", "step"])

        log.write("step", mid=2.0)
        second = read_events(log.path, first["offset"])
        self.assertEqual([e["mid"] for e in second["events"]], [2.0])
        self.assertFalse(second["reset"])

        EventLog(log.path)                       # a new agent session truncates the file
        third = read_events(log.path, second["offset"])
        self.assertTrue(third["reset"])
        self.assertEqual(third["events"], [])

    def test_half_written_line_is_not_consumed(self):
        path = self.dir / "events.jsonl"
        path.write_text('{"type": "step", "mid": 1}\n{"type": "st')
        result = read_events(path, 0)
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["offset"], len('{"type": "step", "mid": 1}\n'))

    def test_missing_file(self):
        self.assertEqual(read_events(self.dir / "nope.jsonl", 0)["events"], [])

    def test_agent_writes_one_event_per_decision(self):
        state = self.dir / "market_state.json"
        state.write_text(json.dumps({"symbol": "SOLUSDT", "bid_price": 10000, "ask_price": 10002}))
        log = EventLog(self.dir / "events.jsonl")

        class Scripted(BollingerBandsStrategy):
            signals = [Signal.buy("go", {"indicators": {"RSI": 25.0}}), Signal.quote(99.0, 101.0)]

            def on_market_data(self, snapshot, inventory):
                return self.signals.pop(0)

        with mock.patch("agents.agent.display"):
            agent = TradingAgent(Scripted(), PaperBroker(fee=0.0), "SOLUSDT", market_state=state,
                                 lot_size=0.5, max_lots=2, events=log)
            agent.step()
            agent.step()

        steps = [e for e in read_events(log.path, 0)["events"] if e["type"] == "step"]
        self.assertEqual([s["signal"] for s in steps], ["BUY", "QUOTE"])
        self.assertEqual(steps[0]["fills"], [{"side": "BUY", "price": 100.02, "quantity": 0.5}])
        self.assertEqual(steps[0]["details"]["indicators"]["RSI"], 25.0)
        self.assertEqual(steps[1]["quotes"], {"bid": 99.0, "ask": 101.0})   # inventory 1 -> both sides
        self.assertEqual(steps[1]["inventory"], 1)


if __name__ == "__main__":
    unittest.main()
