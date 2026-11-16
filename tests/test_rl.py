"""Tests for volume bars, the environments and train/live consistency.

Run with:  python -m unittest discover tests
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from agents.rl.backtest import constant_policy, run_backtest
from agents.rl.env import (BUY, HOLD, LEAN_BUY, NEUTRAL, SELL, EnvConfig,
                           MarketMakingEnv, TakerEnv, quote_prices)
from agents.rl.features import BarSeries, VolumeBarBuilder, make_observation, volume_bars
from agents.rl.networks import Actor, Critic, load_checkpoint, save_checkpoint


def random_trades(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 2e-4, n)))
    quantities = rng.exponential(10, n)
    return np.round(prices, 2), quantities


def flat_bars(prices, spread=0.0):
    prices = np.asarray(prices, float)
    return BarSeries(prices, prices + spread, prices - spread, prices, np.ones_like(prices))


class VolumeBarTests(unittest.TestCase):
    def test_incremental_builder_matches_offline_bars(self):
        prices, quantities = random_trades()
        offline = volume_bars(prices, quantities, threshold=500)

        builder = VolumeBarBuilder(500)
        live = [bar for p, q in zip(prices, quantities) if (bar := builder.add_trade(p, q))]
        live = BarSeries.from_bars(live)

        self.assertGreater(len(offline), 100)
        self.assertEqual(len(live), len(offline))
        for field in ("open", "high", "low", "close", "volume"):
            np.testing.assert_allclose(getattr(live, field), getattr(offline, field))

    def test_bars_hold_the_threshold_volume(self):
        prices, quantities = random_trades()
        bars = volume_bars(prices, quantities, threshold=500)
        self.assertTrue(np.all(bars.high >= bars.close) and np.all(bars.low <= bars.close))
        self.assertAlmostEqual(np.median(bars.volume), 500, delta=50)

    def test_observation_is_scale_free(self):
        prices, quantities = random_trades()
        bars = volume_bars(prices, quantities, 500)
        scaled = BarSeries(bars.open * 7, bars.high * 7, bars.low * 7, bars.close * 7, bars.volume * 3)
        a = make_observation(bars, 40, 30, inventory=3, max_inventory=10)
        b = make_observation(scaled, 40, 30, inventory=3, max_inventory=10)
        self.assertEqual(a.shape, (121,))
        np.testing.assert_allclose(a, b, atol=1e-6)
        self.assertAlmostEqual(a[30 - 1], 0.0)      # last close relative to itself
        self.assertAlmostEqual(a[-1], 0.3)          # inventory / max_inventory

    def test_mirror_reverses_the_trend(self):
        bars = flat_bars(np.linspace(100, 80, 50), spread=0.5)
        mirror = bars.mirrored()
        self.assertAlmostEqual(mirror.close[-1], 125.0)
        self.assertTrue(np.all(mirror.high >= mirror.low))


class EnvironmentTests(unittest.TestCase):
    cfg = EnvConfig(window=3, max_inventory=2, cost=0.001, inventory_penalty=0.0)

    def test_taker_accounting_with_costs(self):
        env = TakerEnv(flat_bars([100, 100, 100, 100, 110, 110]), self.cfg)
        env.reset(start=3, length=3)
        env.step(BUY)                        # buy at 100 * 1.001
        _, reward, _, info = env.step(HOLD)  # marked at 110
        self.assertEqual(info["inventory"], 1)
        self.assertAlmostEqual(info["value"], 110 - 100.1)
        _, _, done, info = env.step(SELL)    # sell at 110 * 0.999
        self.assertTrue(done)
        self.assertAlmostEqual(info["value"], 110 * 0.999 - 100.1)

    def test_long_only_limits(self):
        env = TakerEnv(flat_bars([100] * 10), self.cfg)
        env.reset(start=3, length=6)
        self.assertEqual(env.step(SELL)[3]["fills"], 0)          # nothing to sell
        for _ in range(3):
            info = env.step(BUY)[3]
        self.assertEqual(info["inventory"], 2)                   # capped at max_inventory

    def test_quotes_need_a_trade_through(self):
        cfg = EnvConfig(window=3, max_inventory=2, cost=0.0, inventory_penalty=0.0)
        bid, ask = quote_prices(100.0, NEUTRAL, cfg)
        self.assertEqual((bid, ask), (99.87, 100.13))
        # next bar touches the bid exactly: no fill. Then trades through it: fill at the bid.
        touch = BarSeries(*[np.array(x, float) for x in
                            ([100] * 5, [100] * 5, [100, 100, 100, bid, 100], [100] * 5, [1] * 5)])
        env = MarketMakingEnv(touch, cfg)
        env.reset(start=3, length=1)
        self.assertEqual(env.step(NEUTRAL)[3]["fills"], 0)

        through = BarSeries(*[np.array(x, float) for x in
                              ([100] * 5, [100] * 5, [100, 100, 100, bid - 0.01, 100], [100] * 5, [1] * 5)])
        env = MarketMakingEnv(through, cfg)
        env.reset(start=3, length=1)
        info = env.step(NEUTRAL)[3]
        self.assertEqual((info["fills"], info["inventory"]), (1, 1))
        self.assertAlmostEqual(env.cash, -bid)

    def test_no_ask_without_inventory(self):
        bars = flat_bars([100] * 6, spread=5.0)      # every bar trades through both quotes
        env = MarketMakingEnv(bars, self.cfg)
        env.reset(start=3, length=3)
        self.assertEqual(env.step(NEUTRAL)[3]["fills"], 1)   # only the bid could be posted
        self.assertEqual(env.step(NEUTRAL)[3]["fills"], 2)   # now both sides fill

    def test_backtest_flat_policy_is_zero(self):
        prices, quantities = random_trades()
        bars = volume_bars(prices, quantities, 500)
        result = run_backtest("flat", bars, EnvConfig(), constant_policy(HOLD), TakerEnv)
        self.assertEqual((result.total_return, result.fills), (0.0, 0))
        result = run_backtest("lean buy", bars, EnvConfig(), constant_policy(LEAN_BUY))
        self.assertGreater(result.fills, 0)


class FakeTradeFeed:
    """Replays a list of trades: everything up to ``split`` as history, then in chunks."""

    def __init__(self, prices, quantities, split, chunk=50):
        self.trades = list(zip(prices.tolist(), quantities.tolist()))
        self.split, self.chunk = split, chunk

    def history(self, min_volume):
        return self.trades[:self.split]

    def poll(self):
        out, self.split = self.trades[self.split:self.split + self.chunk], self.split + self.chunk
        return out


class LiveConsistencyTests(unittest.TestCase):
    def test_live_strategy_sees_the_training_observation(self):
        """The network must receive the same input live as in training for the same trades."""
        import torch
        from agents.strategies.a2c import A2CMarketMaker
        from agents.types import MarketSnapshot

        cfg = EnvConfig()
        threshold = 300
        prices, quantities = random_trades(40_000, seed=1)
        offline = volume_bars(prices, quantities, threshold)

        actor, critic = Actor(4 * cfg.window + 1), Critic(4 * cfg.window + 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a2c.pt"
            save_checkpoint(path, actor, critic, cfg, threshold)
            strategy = A2CMarketMaker(FakeTradeFeed(prices, quantities, split=15_000), str(path))

            seen = []
            original_forward = strategy.actor.forward
            strategy.actor.forward = lambda x: (seen.append(x.clone()), original_forward(x))[1]
            snapshot = MarketSnapshot("SOLUSDT", 100.0, 100.01)
            for _ in range(200):
                strategy.on_market_data(snapshot, inventory=4)

        self.assertGreater(len(seen), 10)
        # The last observation used live must equal the training observation at the same bar.
        live_obs = seen[-1][0].numpy()
        live_closes = np.array([b.close for b in strategy.bars])
        matches = [i for i in range(cfg.window, len(offline) + 1)
                   if np.array_equal(offline.close[i - cfg.window:i], live_closes)]
        self.assertEqual(len(matches), 1)
        idx = matches[0]
        train_obs = make_observation(offline, idx, cfg.window, 4, cfg.max_inventory)
        np.testing.assert_allclose(live_obs, train_obs, atol=1e-6)

    def test_checkpoint_round_trip(self):
        cfg = EnvConfig(cost=0.0005)
        actor, critic = Actor(121), Critic(121)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.pt"
            save_checkpoint(path, actor, critic, cfg, 1000, extra={"episode": 7})
            _, _, cfg2, threshold, extra = load_checkpoint(path)
        self.assertEqual((cfg2, threshold, extra["episode"]), (cfg, 1000, 7))


if __name__ == "__main__":
    unittest.main()
