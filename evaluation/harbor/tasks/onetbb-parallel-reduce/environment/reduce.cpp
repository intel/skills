#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

static std::int64_t value_at(std::size_t i) {
    return static_cast<std::int64_t>((i * 17 + 13) % 1009) - 504;
}

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 4000000;
    if (n == 0) return 2;

    std::vector<std::int64_t> data(n);
    for (std::size_t i = 0; i < n; ++i) data[i] = value_at(i);

    long double sum = 0.0;
    long double sumsq = 0.0;
    std::int64_t minv = data[0];
    std::int64_t maxv = data[0];
    for (auto v : data) {
        sum += v;
        sumsq += static_cast<long double>(v) * v;
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
    }

    const long double signature = sum + 0.001L * sumsq + minv * 3.0L + maxv * 7.0L;
    std::cout << "VALID reduce signature=" << static_cast<double>(signature) << "\n";
    return 0;
}
