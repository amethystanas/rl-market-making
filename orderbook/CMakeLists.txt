cmake_minimum_required(VERSION 3.16)
project(OrderBook LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# --------------------------------------------------------------------------
# Core order book (no external dependencies)
# --------------------------------------------------------------------------
add_library(orderbook STATIC
    src/OrderBook/Order.cpp
    src/OrderBook/OrderBook.cpp
    src/OrderBook/OrderModify.cpp
    src/OrderBook/Trade.cpp
)
target_include_directories(orderbook PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(OrderBookTests tests/tests.cpp)
target_link_libraries(OrderBookTests PRIVATE orderbook)
# The tests rely on assert(), so keep it enabled even in Release builds.
target_compile_options(OrderBookTests PRIVATE -UNDEBUG)

add_executable(OrderBookBenchmark tests/benchmark.cpp)
target_link_libraries(OrderBookBenchmark PRIVATE orderbook)

enable_testing()
add_test(NAME OrderBookTests COMMAND OrderBookTests)

# --------------------------------------------------------------------------
# Live Binance feed (downloads libcurl + nlohmann/json at configure time)
# --------------------------------------------------------------------------
option(BUILD_LIVE_FEED "Build the LiveMarketData executable" ON)

if(BUILD_LIVE_FEED)
    include(FetchContent)

    FetchContent_Declare(json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
    FetchContent_MakeAvailable(json)

    find_package(CURL QUIET)
    if(NOT CURL_FOUND)
        message(STATUS "System libcurl not found, building curl 8.4.0 from source")
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(CURL_STATICLIB ON CACHE BOOL "" FORCE)
        set(BUILD_CURL_EXE OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        set(HTTP_ONLY ON CACHE BOOL "" FORCE)
        set(CURL_DISABLE_LDAP ON CACHE BOOL "" FORCE)
        set(CURL_DISABLE_LDAPS ON CACHE BOOL "" FORCE)
        if(WIN32)
            set(CURL_USE_SCHANNEL ON CACHE BOOL "" FORCE)
        elseif(APPLE)
            set(CURL_USE_SECTRANSP ON CACHE BOOL "" FORCE)
        else()
            set(CURL_USE_OPENSSL ON CACHE BOOL "" FORCE)
        endif()
        FetchContent_Declare(curl
            GIT_REPOSITORY https://github.com/curl/curl.git
            GIT_TAG curl-8_4_0)
        FetchContent_MakeAvailable(curl)
    endif()

    add_executable(LiveMarketData
        src/LiveMarketData/Main.cpp
        src/LiveMarketData/Fetcher.cpp
        src/LiveMarketData/Parser.cpp
        src/LiveMarketData/Display.cpp
    )
    target_link_libraries(LiveMarketData PRIVATE
        orderbook CURL::libcurl nlohmann_json::nlohmann_json)
endif()
