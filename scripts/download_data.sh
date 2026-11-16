#!/usr/bin/env bash
# Download one month of Binance aggregated trades from data.binance.vision.
# Usage: scripts/download_data.sh [SYMBOL] [YYYY-MM]
set -euo pipefail
SYMBOL="${1:-SOLUSDT}"
MONTH="${2:-2026-02}"
NAME="${SYMBOL}-aggTrades-${MONTH}"
cd "$(dirname "$0")/.."
mkdir -p data
curl -fL -o "data/${NAME}.zip" \
  "https://data.binance.vision/data/spot/monthly/aggTrades/${SYMBOL}/${NAME}.zip"
unzip -o "data/${NAME}.zip" -d data && rm "data/${NAME}.zip"
echo "Saved data/${NAME}.csv"
