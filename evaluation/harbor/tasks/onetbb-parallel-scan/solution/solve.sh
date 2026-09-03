#!/usr/bin/env bash
set -euo pipefail

cat > /app/scan_tbb.cpp <<'CPP'
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_scan.h>

static std::int64_t value_at(std::size_t i) {
    return static_cast<std::int64_t>((i % 11) + 1);
}

struct PrefixBody {
    const std::vector<std::int64_t>& input;
    std::vector<std::int64_t>& prefix;
    std::int64_t sum;

    PrefixBody(const std::vector<std::int64_t>& in, std::vector<std::int64_t>& out)
        : input(in), prefix(out), sum(0) {}
    PrefixBody(PrefixBody& other, oneapi::tbb::split)
        : input(other.input), prefix(other.prefix), sum(0) {}

    template <typename Tag>
    void operator()(const oneapi::tbb::blocked_range<std::size_t>& r, Tag) {
        std::int64_t temp = sum;
        for (std::size_t i = r.begin(); i != r.end(); ++i) {
            temp += input[i];
            if (Tag::is_final_scan()) prefix[i] = temp;
        }
        sum = temp;
    }

    void reverse_join(PrefixBody& rhs) { sum += rhs.sum; }
    void assign(PrefixBody& rhs) { sum = rhs.sum; }
};

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 3000000;
    if (n == 0) return 2;
    oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, 4);

    std::vector<std::int64_t> input(n), prefix(n);
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<std::size_t>(0, n), [&](const auto& r) {
        for (std::size_t i = r.begin(); i != r.end(); ++i) input[i] = value_at(i);
    });

    PrefixBody body(input, prefix);
    oneapi::tbb::parallel_scan(oneapi::tbb::blocked_range<std::size_t>(0, n), body);

    std::uint64_t signature = static_cast<std::uint64_t>(prefix.back());
    for (std::size_t i = 0; i < n; i += n / 17 + 1) {
        signature = signature * 1315423911ULL + static_cast<std::uint64_t>(prefix[i]);
    }

    std::cout << "VALID scan signature=" << signature << "\n";
    return 0;
}
CPP

g++ -O3 -std=c++17 /app/scan_tbb.cpp -ltbb -o /app/scan_tbb
