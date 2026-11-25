// Reference solution: single-block attention on Intel Xe2 with ESIMD.
//   O[8 x 64] = softmax(Q[8 x 64] * K[256 x 64]^T / sqrt(64)) * V[256 x 64]
// fp16 inputs and output, row-major. One work-group of 4 work-items; work-item t
// handles keys [64t, 64t+64) and the partial results are combined through SLM.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

namespace attention_tile_impl {
using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using sycl::half;

constexpr int M = 8, D = 64, KV = 256, THREADS = 4, KV_PER_T = KV / THREADS;
constexpr int KV_TILES = KV_PER_T / 16;   // 4 key tiles of 16 per work-item
constexpr int D_TILES = D / 16;           // 4 tiles of 16 along the head dimension
constexpr uint32_t ROW_BYTES = D * sizeof(half);           // 128 B pitch (multiple of 16)
// SLM per work-item: O partial [D_TILES][8 x 16] fp32, then row max [8], row sum [8]
constexpr uint32_t O_BYTES = D_TILES * M * 16 * sizeof(float);
constexpr uint32_t STAT_BYTES = 2 * M * sizeof(float);
constexpr uint32_t T_BYTES = O_BYTES + STAT_BYTES;
constexpr uint32_t SLM_BYTES = THREADS * T_BYTES;

constexpr auto hints = properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};
using Tile = simd<float, M * 16>;   // 8 x 16 fp32 tile, [m*16+n]

// S = Q * K^T / sqrt(D) for this work-item's 64 keys: four 8 x 16 fp32 tiles.
// K is [keys x D]: a transposed u32 load of 16 keys x 8 dwords (16 fp16 of D)
// gives the VNNI b-tile directly.
ESIMD_INLINE void scores(const half *Q, const half *K, int kv0, Tile (&S)[KV_TILES]) {
  // Q as four 8 x 16 a-tiles (one per 16-wide slice of the head dimension).
  simd<half, M * 16> q[D_TILES];
#pragma unroll
  for (int s = 0; s < D_TILES; s++)
    q[s] = load_2d<half, 16, M>(Q, ROW_BYTES - 1, M - 1, ROW_BYTES - 1, 16 * s, 0, hints);
#pragma unroll
  for (int j = 0; j < KV_TILES; j++) {
    S[j] = 0.f;
#pragma unroll
    for (int s = 0; s < D_TILES; s++) {
      simd<uint32_t, 128> kt = load_2d<uint32_t, 8, 16, 1, true, false>(
          reinterpret_cast<const uint32_t *>(K), ROW_BYTES - 1, KV - 1, ROW_BYTES - 1,
          8 * s, kv0 + 16 * j, hints);
      simd<half, 256> b = kt.bit_cast_view<half>();
      S[j] = xmx::dpas<8, 8, float, float, half, half>(S[j], b, q[s]);
    }
    S[j] *= 0.125f;   // 1 / sqrt(D)
  }
}

// Local row max and row sum of exp(S - max) over this work-item's keys; P = exp(S - max).
ESIMD_INLINE void local_softmax(Tile (&S)[KV_TILES], simd<half, M * 16> (&P)[KV_TILES],
                                simd<float, M> &row_max, simd<float, M> &row_sum) {
  Tile mx = S[0];
#pragma unroll
  for (int j = 1; j < KV_TILES; j++) mx = max(mx, S[j]);
#pragma unroll
  for (int m = 0; m < M; m++) row_max[m] = hmax<float>(mx.select<16, 1>(m * 16).read());

  Tile lsum = 0.f;
#pragma unroll
  for (int j = 0; j < KV_TILES; j++) {
#pragma unroll
    for (int m = 0; m < M; m++) { float rm = row_max[m]; S[j].select<16, 1>(m * 16) -= rm; }
    S[j] = exp(S[j]);
    lsum += S[j];
    P[j] = convert<half>(S[j]);   // fp32 acc layout [m*16+key] == a-tile layout
  }
#pragma unroll
  for (int m = 0; m < M; m++) row_sum[m] = detail::sum<float, float, 16>(lsum.select<16, 1>(m * 16).read());
}

// O partial = P * V. V is [keys x D] = K-major B, so the 2D load's VNNI transform
// produces the b-tile in hardware.
ESIMD_INLINE void partial_output(const half *V, int kv0, const simd<half, M * 16> (&P)[KV_TILES],
                                 Tile (&Op)[D_TILES]) {
#pragma unroll
  for (int dt = 0; dt < D_TILES; dt++) {
    Op[dt] = 0.f;
#pragma unroll
    for (int j = 0; j < KV_TILES; j++) {
      simd<half, 256> vb = load_2d<half, 16, 16, 1, false, true>(
          V, ROW_BYTES - 1, KV - 1, ROW_BYTES - 1, 16 * dt, kv0 + 16 * j, hints);
      Op[dt] = xmx::dpas<8, 8, float, float, half, half>(Op[dt], vb, P[j]);
    }
  }
}

// Publish partial O, row max and row sum to this work-item's SLM slot.
ESIMD_INLINE void publish(int t, const Tile (&Op)[D_TILES], simd<float, M> row_max,
                          simd<float, M> row_sum) {
  const uint32_t base = t * T_BYTES;
#pragma unroll
  for (int dt = 0; dt < D_TILES; dt++)
    slm_block_store<float, M * 16>(base + dt * M * 16 * sizeof(float), Op[dt]);
  slm_block_store<float, M>(base + O_BYTES, row_max);
  slm_block_store<float, M>(base + O_BYTES + M * sizeof(float), row_sum);
}

// Head-dimension slice t of O: rescale the 4 partials to the global row max and combine.
ESIMD_INLINE Tile combine(int t) {
  simd<float, M> gmax = slm_block_load<float, M>(O_BYTES);
#pragma unroll
  for (int u = 1; u < THREADS; u++)
    gmax = max(gmax, slm_block_load<float, M>(u * T_BYTES + O_BYTES));
  Tile acc = 0.f;
  simd<float, M> L = 0.f;
#pragma unroll
  for (int u = 0; u < THREADS; u++) {
    simd<float, M> mu = slm_block_load<float, M>(u * T_BYTES + O_BYTES);
    simd<float, M> lu = slm_block_load<float, M>(u * T_BYTES + O_BYTES + M * sizeof(float));
    simd<float, M> scale = exp(mu - gmax);
    L += lu * scale;
    Tile ou = slm_block_load<float, M * 16>(u * T_BYTES + t * M * 16 * sizeof(float));
#pragma unroll
    for (int m = 0; m < M; m++) { float sm = scale[m]; acc.select<16, 1>(m * 16) += ou.select<16, 1>(m * 16) * sm; }
  }
#pragma unroll
  for (int m = 0; m < M; m++) { float lm = L[m]; acc.select<16, 1>(m * 16) /= lm; }
  return acc;
}

struct AttnTile {
  const half *Q, *K, *V;
  half *O;

  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    slm_init<SLM_BYTES>();
    const int t = it.get_local_id(0);
    const int kv0 = t * KV_PER_T;

    Tile S[KV_TILES];
    scores(Q, K, kv0, S);
    simd<half, M * 16> P[KV_TILES];
    simd<float, M> row_max, row_sum;
    local_softmax(S, P, row_max, row_sum);
    Tile Op[D_TILES];
    partial_output(V, kv0, P, Op);

    publish(t, Op, row_max, row_sum);
    barrier();   // every work-item reaches it
    // Work-item t finalizes head-dimension slice t.
    store_2d<half, 16, M>(O, ROW_BYTES - 1, M - 1, ROW_BYTES - 1, 16 * t, 0, convert<half>(combine(t)));
  }
};

}  // namespace attention_tile_impl

inline sycl::event attention_tile(sycl::queue &q, const sycl::half *Q, const sycl::half *K,
                                  const sycl::half *V, sycl::half *O) {
  using namespace attention_tile_impl;
  return q.parallel_for(sycl::nd_range<1>(THREADS, THREADS), AttnTile{Q, K, V, O});
}
