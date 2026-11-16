#!/usr/bin/env bash
# Configure, build and test the C++ order book.
set -euo pipefail
cd "$(dirname "$0")/../orderbook"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
