// Verifier harness: calls attention_tile() from the agent's /app/attention.hpp.
// Always built; run only when an Intel GPU is visible. Prints one RESULT line.
#include "attention.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

constexpr int M = 8, D = 64, KV = 256;

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<sycl::half> hQ(M * D), hK(KV * D), hV(KV * D), hO(M * D, sycl::half(NAN));
  for (auto &x : hQ) x = 2.f * dist(rng);
  for (auto &x : hK) x = dist(rng);
  for (auto &x : hV) x = dist(rng);
  // A strong key for row 2t inside each quarter of the keys, so that every quarter
  // owns some row's maximum and the cross-quarter combine step matters.
  for (int t = 0; t < 4; t++)
    for (int d = 0; d < D; d++) hK[(t * 64 + 5) * D + d] = hQ[(2 * t) * D + d];

  sycl::half *Q = sycl::malloc_device<sycl::half>(M * D, q), *K = sycl::malloc_device<sycl::half>(KV * D, q);
  sycl::half *V = sycl::malloc_device<sycl::half>(KV * D, q), *O = sycl::malloc_device<sycl::half>(M * D, q);
  q.memcpy(Q, hQ.data(), M * D * 2).wait();
  q.memcpy(K, hK.data(), KV * D * 2).wait();
  q.memcpy(V, hV.data(), KV * D * 2).wait();
  q.memcpy(O, hO.data(), M * D * 2).wait();
  attention_tile(q, Q, K, V, O).wait();
  q.memcpy(hO.data(), O, M * D * 2).wait();

  int bad = 0;
  double max_abs = 0;
  for (int m = 0; m < M; m++) {
    std::vector<double> s(KV);
    double smax = -1e300, l = 0;
    for (int k = 0; k < KV; k++) {
      double acc = 0;
      for (int d = 0; d < D; d++) acc += double(hQ[m * D + d]) * double(hK[k * D + d]);
      s[k] = acc / 8.0;
      smax = std::max(smax, s[k]);
    }
    for (int k = 0; k < KV; k++) { s[k] = std::exp(s[k] - smax); l += s[k]; }
    for (int d = 0; d < D; d++) {
      double r = 0;
      for (int k = 0; k < KV; k++) r += s[k] * double(hV[k * D + d]);
      r /= l;
      double g = double(hO[m * D + d]);
      double e = std::fabs(g - r);
      if (!(e <= 5e-3 + 1e-2 * std::fabs(r))) bad++;
      if (!(e <= max_abs)) max_abs = std::isnan(e) ? INFINITY : e;
    }
  }
  std::printf("RESULT %s bad=%d/%d max_abs_err=%g\n", bad ? "FAIL" : "PASS", bad, M * D, max_abs);
  return bad != 0;
}
