#!/usr/bin/env bash
set -euo pipefail

cat > /app/transpose_tbb.cpp <<'CPP'
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/parallel_for.h>

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 10;
    const int tile = argc > 3 ? std::atoi(argv[3]) : 32;
    if (n < 1 || iterations < 1) return 2;
    oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, 4);

    std::vector<double> a(static_cast<std::size_t>(n) * n);
    std::vector<double> b(static_cast<std::size_t>(n) * n, 0.0);

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range2d<int>(0, n, tile, 0, n, tile), [&](const auto& r) {
        for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
            for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
                a[static_cast<std::size_t>(i) * n + j] = static_cast<double>(i * n + j);
            }
        }
    });

    for (int iter = 0; iter < iterations; ++iter) {
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range2d<int>(0, n, tile, 0, n, tile), [&](const auto& r) {
            for (int i = r.rows().begin(); i != r.rows().end(); ++i) {
                for (int j = r.cols().begin(); j != r.cols().end(); ++j) {
                    b[static_cast<std::size_t>(j) * n + i] = a[static_cast<std::size_t>(i) * n + j] + iter;
                }
            }
        });
    }

    double err = 0.0;
    double signature = 0.0;
    const int probes[][2] = {{0, 0}, {n / 3, n / 5}, {n - 1, n - 1}, {n / 2, n / 2}};
    for (auto& p : probes) {
        const int i = p[0];
        const int j = p[1];
        const double expected = static_cast<double>(i * n + j + iterations - 1);
        const double actual = b[static_cast<std::size_t>(j) * n + i];
        err += std::abs(actual - expected);
        signature += actual;
    }

    if (err > 1e-8) {
        std::cerr << "TRANSPOSE_ERROR err=" << err << "\n";
        return 1;
    }
    std::cout << "VALID transpose signature=" << signature << "\n";
    return 0;
}
CPP

g++ -O3 -std=c++17 /app/transpose_tbb.cpp -ltbb -o /app/transpose_tbb
