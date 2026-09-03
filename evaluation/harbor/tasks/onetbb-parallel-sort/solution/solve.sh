#!/bin/bash
# Oracle solution: implement parallel sort using oneTBB.
set -euo pipefail

cat > /app/sort_parallel.cpp << 'EOF'
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <oneapi/tbb/parallel_sort.h>

int main() {
    const size_t N = 10'000'000;

    std::vector<int> data(N);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 1'000'000'000);
    for (auto& x : data) x = dist(rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    tbb::parallel_sort(data.begin(), data.end());
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "Parallel sort elapsed: " << elapsed << "s\n";

    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i] < data[i - 1]) {
            std::cout << "NOT_SORTED\n";
            return 1;
        }
    }
    std::cout << "SORTED\n";
    return 0;
}
EOF

g++ -O2 -std=c++17 /app/sort_parallel.cpp -o /app/sort_parallel -ltbb
echo "Build successful: /app/sort_parallel"
/app/sort_parallel
