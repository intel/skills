#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 768;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 8;
    if (n < 3 || iterations < 1) return 2;

    std::vector<double> in(static_cast<std::size_t>(n) * n);
    std::vector<double> out(static_cast<std::size_t>(n) * n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            in[static_cast<std::size_t>(i) * n + j] = static_cast<double>((i * 17 + j * 13) % 101);
        }
    }

    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 1; i < n - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                const std::size_t idx = static_cast<std::size_t>(i) * n + j;
                out[idx] = 0.25 * (in[idx - n] + in[idx + n] + in[idx - 1] + in[idx + 1]);
            }
        }
        in.swap(out);
    }

    double norm = 0.0;
    for (double v : in) norm += std::abs(v);
    if (!(norm > 0.0)) {
        std::cerr << "INVALID norm=" << norm << "\n";
        return 1;
    }
    std::cout << "VALID stencil norm=" << norm << "\n";
    return 0;
}
