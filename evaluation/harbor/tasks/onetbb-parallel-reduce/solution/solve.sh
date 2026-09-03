#!/usr/bin/env bash
set -euo pipefail

cat > /app/reduce_tbb.cpp <<'CPP'
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>

static std::int64_t value_at(std::size_t i) {
    return static_cast<std::int64_t>((i * 17 + 13) % 1009) - 504;
}

struct Aggregate {
    long double sum = 0.0;
    long double sumsq = 0.0;
    std::int64_t minv = std::numeric_limits<std::int64_t>::max();
    std::int64_t maxv = std::numeric_limits<std::int64_t>::min();

    void add(std::int64_t v) {
        sum += v;
        sumsq += static_cast<long double>(v) * v;
        minv = std::min(minv, v);
        maxv = std::max(maxv, v);
    }

    void join(const Aggregate& other) {
        sum += other.sum;
        sumsq += other.sumsq;
        minv = std::min(minv, other.minv);
        maxv = std::max(maxv, other.maxv);
    }
};

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 4000000;
    if (n == 0) return 2;
    oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, 4);

    std::vector<std::int64_t> data(n);
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, n), [&](const auto& r) {
        for (std::size_t i = r.begin(); i != r.end(); ++i) data[i] = value_at(i);
    });

    Aggregate result = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<std::size_t>(0, n), Aggregate{},
        [&](const auto& r, Aggregate local) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) local.add(data[i]);
            return local;
        },
        [](Aggregate a, const Aggregate& b) { a.join(b); return a; });

    const long double signature = result.sum + 0.001L * result.sumsq + result.minv * 3.0L + result.maxv * 7.0L;
    std::cout << "VALID reduce signature=" << static_cast<double>(signature) << "\n";
    return 0;
}
CPP

g++ -O3 -std=c++17 /app/reduce_tbb.cpp -ltbb -o /app/reduce_tbb
