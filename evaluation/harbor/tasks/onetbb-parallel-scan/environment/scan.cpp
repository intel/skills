#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

static std::int64_t value_at(std::size_t i) {
    return static_cast<std::int64_t>((i % 11) + 1);
}

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 3000000;
    if (n == 0) return 2;

    std::vector<std::int64_t> input(n), prefix(n);
    for (std::size_t i = 0; i < n; ++i) input[i] = value_at(i);

    std::int64_t running = 0;
    for (std::size_t i = 0; i < n; ++i) {
        running += input[i];
        prefix[i] = running;
    }

    std::uint64_t signature = static_cast<std::uint64_t>(prefix.back());
    for (std::size_t i = 0; i < n; i += n / 17 + 1) {
        signature = signature * 1315423911ULL + static_cast<std::uint64_t>(prefix[i]);
    }

    std::cout << "VALID scan signature=" << signature << "\n";
    return 0;
}
