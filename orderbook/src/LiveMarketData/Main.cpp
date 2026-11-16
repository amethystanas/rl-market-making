// LiveMarketData: polls the Binance REST order book, feeds the snapshot into
// the in-memory order book and exports the top of book as JSON so that the
// Python trading agents can consume it.
//
// Usage: LiveMarketData [SYMBOL] [REFRESH_SECONDS] [DEPTH] [OUTPUT_JSON]
//   e.g. LiveMarketData SOLUSDT 1 20 data/market_state.json

#include <chrono>
#include <curl/curl.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "LiveMarketData/Display.h"
#include "LiveMarketData/Fetcher.h"
#include "LiveMarketData/Parser.h"
#include "OrderBook/OrderBook.h"

int main(int argc, char *argv[]) {
    const std::string symbol = argc > 1 ? argv[1] : "SOLUSDT";
    const int refreshSeconds = argc > 2 ? std::stoi(argv[2]) : 1;
    const int depth = argc > 3 ? std::stoi(argv[3]) : 20;
    const std::string outputPath = argc > 4 ? argv[4] : "data/market_state.json";

    std::cout << "Binance live feed | symbol=" << symbol << " refresh=" << refreshSeconds
              << "s depth=" << depth << " output=" << outputPath << "\n";

    curl_global_init(CURL_GLOBAL_DEFAULT);
    Orderbook orderbook;

    for (int iteration = 1;; ++iteration) {
        const std::string response = FetchBinanceOrderbook(symbol, depth);

        if (response.empty()) {
            std::cerr << "Failed to fetch order book, retrying...\n";
        } else {
            try {
                const BookSnapshotMessage snapshot = ParseBinanceSnapshot(response);
                if (orderbook.ProcessMarketData(snapshot)) {
                    WriteMarketStateJson(orderbook, outputPath, iteration, symbol);
                    PrintOrderbook(orderbook, symbol, depth);
                } else {
                    std::cerr << "Failed to process market data\n";
                }
            } catch (const std::exception &e) {
                std::cerr << "Error processing snapshot: " << e.what() << "\n";
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(refreshSeconds));
    }

    curl_global_cleanup();
    return 0;
}
