// Minimal host program that calls attention_tile() from /app/attention.hpp.
// There is no GPU in this environment, so it can be built but not run.
#include "attention.hpp"

int main() {
  sycl::queue q;
  sycl::half *Q = sycl::malloc_device<sycl::half>(8 * 64, q);
  sycl::half *K = sycl::malloc_device<sycl::half>(256 * 64, q);
  sycl::half *V = sycl::malloc_device<sycl::half>(256 * 64, q);
  sycl::half *O = sycl::malloc_device<sycl::half>(8 * 64, q);
  attention_tile(q, Q, K, V, O).wait();
  sycl::free(Q, q); sycl::free(K, q); sycl::free(V, q); sycl::free(O, q);
  return 0;
}
