#!/usr/bin/env bash
set -euo pipefail

cat > /app/flow_graph_tbb.cpp <<'CPP'
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <oneapi/tbb/flow_graph.h>
#include <oneapi/tbb/global_control.h>

static std::int64_t transform(std::int64_t x) {
    const std::int64_t y = (x * 17 + 5) % 1009;
    return y * y + 3 * y + 7;
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 200000;
    if (n < 1) return 2;
    oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, 4);

    oneapi::tbb::flow::graph graph;
    std::atomic<std::int64_t> sum{0};

    oneapi::tbb::flow::function_node<int, std::int64_t> transform_node(
        graph, oneapi::tbb::flow::unlimited,
        [](int i) { return transform(i); });

    oneapi::tbb::flow::function_node<std::int64_t, oneapi::tbb::flow::continue_msg> consume_node(
        graph, oneapi::tbb::flow::serial,
        [&](std::int64_t value) {
            sum.fetch_add(value, std::memory_order_relaxed);
            return oneapi::tbb::flow::continue_msg();
        });

    oneapi::tbb::flow::make_edge(transform_node, consume_node);
    for (int i = 0; i < n; ++i) transform_node.try_put(i);
    graph.wait_for_all();

    std::cout << "VALID flow_graph sum=" << sum.load() << "\n";
    return 0;
}
CPP

g++ -O3 -std=c++17 /app/flow_graph_tbb.cpp -ltbb -o /app/flow_graph_tbb
