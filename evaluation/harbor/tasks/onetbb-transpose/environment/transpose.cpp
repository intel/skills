#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 10;
    if (n < 1 || iterations < 1) return 2;

    std::vector<double> a(static_cast<std::size_t>(n) * n);
    std::vector<double> b(static_cast<std::size_t>(n) * n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[static_cast<std::size_t>(i) * n + j] = static_cast<double>(i * n + j);
        }
    }

    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                b[static_cast<std::size_t>(j) * n + i] = a[static_cast<std::size_t>(i) * n + j] + iter;
            }
        }
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
