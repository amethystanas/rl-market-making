"""Terminal output for the live trading loop."""

from __future__ import annotations

from datetime import datetime

GREEN, RED, YELLOW, CYAN, BOLD, RESET = ("\033[92m", "\033[91m", "\033[93m",
                                         "\033[96m", "\033[1m", "\033[0m")


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def banner(strategy: str, symbol: str, source) -> None:
    print(f"{BOLD}Trading agent | strategy={strategy} symbol={symbol} feed={source}{RESET}")
    print("Ctrl+C to stop.\n")


def info(message: str) -> None:
    print(f"[{_now()}] {message}")


def decision(snapshot, signal, lots: int, max_lots: int, pnl: float) -> None:
    if signal.quotes is not None:
        label, color = "QUOTE", CYAN
    elif signal.side is not None:
        label, color = signal.side.value, GREEN if signal.side.value == "BUY" else RED
    else:
        label, color = "HOLD", YELLOW
    print(f"[{_now()}] mid {snapshot.mid:10.2f} | {color}{label:5}{RESET} {signal.reason:<45} "
          f"| inv {lots:2d}/{max_lots} lots | P&L {pnl:+.4f}")


def quotes(bid: float | None, ask: float | None) -> None:
    fmt = lambda p: f"{p:.2f}" if p is not None else "  -  "
    print(f"           quotes  bid {fmt(bid)}  ask {fmt(ask)}")


def fill(f) -> None:
    color = GREEN if f.side.value == "BUY" else RED
    print(f"{color}{BOLD}[{_now()}] FILLED {f.side.value} {f.quantity:g} @ {f.price:.2f}{RESET}")


def summary(trades: int, lots: int, pnl: float) -> None:
    color = GREEN if pnl >= 0 else RED
    print(f"\n{BOLD}Session summary{RESET}: {trades} fills | final inventory {lots} lots | "
          f"P&L {color}{pnl:+.4f}{RESET} (quote currency, marked to mid)")
