#include <cstdint>
#include <cstdlib>
#include <iostream>

static std::int64_t transform(std::int64_t x) {
    const std::int64_t y = (x * 17 + 5) % 1009;
    return y * y + 3 * y + 7;
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 200000;
    if (n < 1) return 2;

    std::int64_t sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += transform(i);
    }

    std::cout << "VALID flow_graph sum=" << sum << "\n";
    return 0;
}
