#!/usr/bin/env bash
# Start the C++ live order book feed (writes data/market_state.json).
# Usage: scripts/run_feed.sh [SYMBOL] [REFRESH_SECONDS] [DEPTH]
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./orderbook/build/LiveMarketData "${1:-SOLUSDT}" "${2:-1}" "${3:-20}" data/market_state.json
