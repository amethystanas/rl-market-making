"""Evaluate the trained A2C market maker against baselines on held-out data.

    python -m agents.rl.evaluate --data data/SOLUSDT-aggTrades-2026-02.csv

Prints three markdown tables (market makers, taker strategies, fee
sensitivity) and saves the equity curves. Returns are in % of the capital
needed to hold the maximum position.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from agents.rl.backtest import (BacktestResult, actor_policy, constant_policy,
                                run_backtest, strategy_policy)
from agents.rl.env import (BUY, HOLD, LEAN_BUY, LEAN_SELL, NEUTRAL, EnvConfig,
                           TakerEnv)
from agents.rl.features import BarSeries, load_agg_trades
from agents.rl.networks import load_checkpoint
from agents.rl.train_a2c import split_bars
from agents.strategies.rules import (BollingerBandsStrategy, MeanReversionStrategy,
                                     RandomStrategy, RSIStrategy)

FEES_BPS = (0, 2, 7.5, 10)


def market_makers(actor) -> dict:
    return {
        "A2C (greedy)": actor_policy(actor),
        "Always NEUTRAL": constant_policy(NEUTRAL),
        "Always LEAN_BUY": constant_policy(LEAN_BUY),
        "Always LEAN_SELL": constant_policy(LEAN_SELL),
    }


def taker_strategies() -> dict:
    return {
        "Buy & hold": constant_policy(BUY),
        "Mean reversion": strategy_policy(MeanReversionStrategy()),
        "RSI": strategy_policy(RSIStrategy()),
        "Bollinger bands": strategy_policy(BollingerBandsStrategy()),
        "Random": strategy_policy(RandomStrategy(seed=0)),
        "Flat (never trade)": constant_policy(HOLD),
    }


def table(results: list[BacktestResult]) -> str:
    rows = ["| Strategy | Return | Max drawdown | Fills | Avg exposure |",
            "|---|---:|---:|---:|---:|"]
    rows += [f"| {r.name} | {r.total_return:+.2f}% | {r.max_drawdown:.2f}% | {r.fills} | "
             f"{100 * r.exposure:.0f}% |" for r in results]
    return "\n".join(rows)


def fee_table(bars: BarSeries, cfg: EnvConfig, actor) -> str:
    rows = ["| Strategy | " + " | ".join(f"{f:g} bps" for f in FEES_BPS) + " |",
            "|---|" + "---:|" * len(FEES_BPS)]
    for name, policy in market_makers(actor).items():
        returns = [run_backtest(name, bars, replace(cfg, cost=f / 1e4), policy).total_return
                   for f in FEES_BPS]
        rows.append(f"| {name} | " + " | ".join(f"{r:+.2f}%" for r in returns) + " |")
    return "\n".join(rows)


def plot_equity(groups: dict[str, list[BacktestResult]], bars: BarSeries, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(groups) + 1, 1, figsize=(12, 3 + 3.2 * len(groups)), sharex=True,
                             gridspec_kw={"height_ratios": [1] + [1.6] * len(groups)})
    axes[0].plot(bars.close, color="black", lw=0.8)
    axes[0].set_ylabel("SOL / USDT")
    axes[0].set_title(title)
    for ax, (group, results) in zip(axes[1:], groups.items()):
        for r in results:
            style = {"lw": 2.4, "zorder": 5, "color": "black"} if r.name.startswith("A2C") else {"lw": 1.2}
            ax.plot(range(len(bars) - len(r.equity), len(bars)), r.equity, label=r.name, **style)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_ylabel("P&L (% of capital)")
        ax.set_title(group, fontsize=11)
        ax.legend(loc="lower left", fontsize=8, ncol=3)
    axes[-1].set_xlabel("volume bar")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", default="data/SOLUSDT-aggTrades-2026-02.csv")
    p.add_argument("--model", default="models/a2c.pt")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--figure", default="docs/figures/a2c_backtest.png")
    args = p.parse_args()

    actor, _, cfg, threshold, extra = load_checkpoint(args.model)
    splits = dict(zip(("train", "val", "test"), split_bars(load_agg_trades(args.data, threshold))))
    bars = splits[args.split]
    print(f"Model from episode {extra.get('episode')} | {args.split} split: {len(bars)} bars of "
          f"{threshold:g} units | fee {cfg.cost * 1e4:g} bps per fill\n")

    mm = [run_backtest(n, bars, cfg, pol) for n, pol in market_makers(actor).items()]
    taker = [run_backtest(n, bars, cfg, pol, TakerEnv) for n, pol in taker_strategies().items()]
    print("Market makers (passive quotes)\n\n" + table(mm))
    print("\nTaker strategies (market orders)\n\n" + table(taker))
    print("\nFee sensitivity, market makers (return)\n\n" + fee_table(bars, cfg, actor))

    plot_equity({"Market makers (passive quotes)": mm, "Taker strategies (market orders)": taker},
                bars, Path(args.figure), f"Backtest on the {args.split} split (unseen during training)")
    print(f"\nequity curves -> {args.figure}")


if __name__ == "__main__":
    main()
