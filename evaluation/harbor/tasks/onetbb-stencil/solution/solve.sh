#!/usr/bin/env bash
set -euo pipefail

cat > /app/stencil_tbb.cpp <<'CPP'
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_reduce.h>

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 768;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 8;
    if (n < 3 || iterations < 1) return 2;
    oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, 4);

    std::vector<double> in(static_cast<std::size_t>(n) * n);
    std::vector<double> out(static_cast<std::size_t>(n) * n, 0.0);

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range2d<int>(0, n, 32, 0, n, 32), [&](const auto& r) {
        for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
            for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
                in[static_cast<std::size_t>(i) * n + j] = static_cast<double>((i * 17 + j * 13) % 101);
            }
        }
    });

    for (int iter = 0; iter < iterations; ++iter) {
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range2d<int>(1, n - 1, 32, 1, n - 1, 32), [&](const auto& r) {
            for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
                for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
                    const std::size_t idx = static_cast<std::size_t>(i) * n + j;
                    out[idx] = 0.25 * (in[idx - n] + in[idx + n] + in[idx - 1] + in[idx + 1]);
                }
            }
        });
        in.swap(out);
    }

    const double norm = oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<std::size_t>(0, in.size()), 0.0,
        [&](const auto& r, double sum) {
            for (std::size_t i = r.begin(); i != r.end(); ++i) sum += std::abs(in[i]);
            return sum;
        },
        [](double x, double y) { return x + y; });

    if (!(norm > 0.0)) {
        std::cerr << "INVALID norm=" << norm << "\n";
        return 1;
    }
    std::cout << "VALID stencil norm=" << norm << "\n";
    return 0;
}
CPP

g++ -O3 -std=c++17 /app/stencil_tbb.cpp -ltbb -o /app/stencil_tbb
