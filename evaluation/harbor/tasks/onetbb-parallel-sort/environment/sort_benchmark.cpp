// Serial baseline — sort 10M integers with std::sort.
// The agent's task: rewrite using tbb::parallel_sort.
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

int main() {
    const size_t N = 10'000'000;

    std::vector<int> data(N);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 1'000'000'000);
    for (auto& x : data) x = dist(rng);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::sort(data.begin(), data.end());
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "Serial sort elapsed: " << elapsed << "s\n";

    // Verify sorted
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i] < data[i - 1]) {
            std::cout << "NOT_SORTED\n";
            return 1;
        }
    }
    std::cout << "SORTED\n";
    return 0;
}
