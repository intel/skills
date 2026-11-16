// fp16_dpas_ult.cpp  —  Unit Level Test: all DPAS load/store patterns on XE2
//
// DPAS register contract (FP16, SD=8 → 16 fp16 K-elements per call):
//   dpas<SystolicDepth=8, RepeatCount=8, T, CT, BT, AT>(acc, b_tile, a_tile)
//   a_tile : simd<half,128>  layout [m*16+k]            8 M-rows  × 16 K-cols
//   b_tile : simd<half,256>  uint32[k_pair*16+n_col]    K=16 × N=16  VNNI
//   acc    : simd<half,128>  layout [m*16+n]            8 M-rows  × 16 N-cols
//
// Four cases tested:
//
//  Case 1  Usage1 + B[N,K]:  M=8  N=16 K=16
//    a_tile ← A[M,K]   via lsc_load_2d  BW=16 BH=8   no-transform
//    b_tile ← B[N,K]   via lsc_gather<uint32_t,8,u32,N=16>  SOA → VNNI
//    dpas(acc, b_tile, a_tile) → acc[m*16+n] = C[m,n]
//    store  → lsc_store_2d C[8,16]
//
//  Case 2  Usage1 + B_T[K,N]:  M=8  N=16 K=16
//    a_tile ← A[M,K]   via lsc_load_2d  BW=16 BH=8   no-transform
//    b_tile ← B_T[K,N] via lsc_load_2d  BW=16 BH=16  VNNI=true
//    dpas(acc, b_tile, a_tile) → acc[m*16+n] = C[m,n]
//    store  → lsc_store_2d C[8,16]
//
//  Case 3  Usage2 + B[N,K]:  M=16 N=8  K=16
//    b_tile ← A[M,K]   via lsc_gather<uint32_t,8,u32,N=16>  SOA → VNNI
//    a_tile ← B[N,K]   via lsc_load_2d  BW=16 BH=8   no-transform
//    dpas(acc, b_tile=A_vnni, a_tile=B) → acc[n_i*16+m_j] = C[m_j,n_i]  transposed
//    store  → pack column pairs to u32, lsc_scatter<uint32_t,4,u32,N=16>
//
//  Case 4  Usage2 + B_T[K,N]:  M=16 N=8  K=16
//    b_tile ← A[M,K]   via lsc_gather<uint32_t,8,u32,N=16>  SOA → VNNI
//    a_tile ← B_T[K,N] via transposed u32 load_2d (4 u32 × 16 K-rows) + deinterleave
//    dpas(acc, b_tile=A_vnni, a_tile=B) → acc[n_i*16+m_j] = C[m_j,n_i]
//    store  → pack column pairs to u32, lsc_scatter<uint32_t,4,u32,N=16>
//
//  fp16 gather/scatter with NElts > 1 (lsc_gather/lsc_scatter<sycl::half,8,u16>) is
//  rejected at vISA ("only D32 and D64 support vector load"), hence the u32 forms.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>

using namespace sycl;
using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
namespace xesimd = sycl::ext::intel::experimental::esimd;

// ────────────────────────────────────────────────────────────────────────────
// Shared helper: load a_tile from A[M×K] via lsc_load_2d (BW=16, BH=8)
//   a_tile[m*16+k] = A[m, k]  row-major, no transform
// ────────────────────────────────────────────────────────────────────────────
// (inlined in each kernel below for clarity)


// ============================================================
// Case 1: Usage1 + B[N,K]   M=8 N=16 K=16
//   a_tile ← A[8,K]   via lsc_load_2d  BW=16 BH=8
//   b_tile ← B[16,K]  via lsc_gather<uint32_t,8,u32,N=16>  (SOA→VNNI)
// ============================================================
struct DPAS_ULT_C1 {
    const sycl::half* A;    // [8  × K]
    const sycl::half* B;    // [16 × K]  B[N,K]
    sycl::half*       C;    // [8  × 16]
    int K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        // ── a_tile: A[8×K] via lsc_load_2d ──────────────────────────────────
        xesimd::config_2d_mem_access<sycl::half, 16, 8, 1> payA(
            A, (uint32_t)K*2u-1u, 7u, (uint32_t)K*2u-1u, 0u, 0u);
        simd<sycl::half, 8*16> a_tile = xesimd::lsc_load_2d<
            sycl::half, 16, 8, 1, false/*T*/, false/*VNNI*/,
            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);
        // a_tile[m*16+k] = A[m, k] ✓

        // ── b_tile: B[16×K] (N×K) via lsc_gather<uint32_t,8,u32,N=16> ──────
        // lsc_gather SOA: result[k_pair*16 + n_col] = uint32 at (offset[n_col]+k_pair*4)
        //               = {B[n_col, k_pair*2], B[n_col, k_pair*2+1]}  VNNI ✓
        const uint32_t* B_u32 = reinterpret_cast<const uint32_t*>(B);
        simd<uint32_t, 16> b_off;
        #pragma unroll
        for (int n = 0; n < 16; n++)
            b_off[n] = (uint32_t)n * (uint32_t)K * 2u;   // byte offset to B[n, 0]
        simd<sycl::half, 16*16> b_tile;
        b_tile.template bit_cast_view<uint32_t>() =
            xesimd::lsc_gather<uint32_t, 8,
                xesimd::lsc_data_size::u32,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached,
                16, uint32_t>(B_u32, b_off);

        // ── DPAS ─────────────────────────────────────────────────────────────
        simd<sycl::half, 8*16> acc(0.f);
        acc = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(
            acc, b_tile, a_tile);
        // acc[m*16+n] = sum_k A[m,k]*B[n,k] = C[m,n] ✓

        // ── Store C[8×16] via lsc_store_2d ───────────────────────────────────
        xesimd::lsc_store_2d<sycl::half, 16, 8,
            xesimd::cache_hint::write_back, xesimd::cache_hint::write_back>(
            C, 31u, 7u, 31u, 0u, 0u, acc);
    }
};

// ============================================================
// Case 2: Usage1 + B_T[K,N]   M=8 N=16 K=16
//   a_tile ← A[8,K]    via lsc_load_2d  BW=16 BH=8
//   b_tile ← B_T[K,16] via lsc_load_2d  BW=16 BH=16  VNNI=true
// ============================================================
struct DPAS_ULT_C2 {
    const sycl::half* A;    // [8  × K]
    const sycl::half* B_T;  // [K  × 16]  B_T[K,N]
    sycl::half*       C;    // [8  × 16]
    int K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        // ── a_tile: A[8×K] via lsc_load_2d (same as Case 1) ─────────────────
        xesimd::config_2d_mem_access<sycl::half, 16, 8, 1> payA(
            A, (uint32_t)K*2u-1u, 7u, (uint32_t)K*2u-1u, 0u, 0u);
        simd<sycl::half, 8*16> a_tile = xesimd::lsc_load_2d<
            sycl::half, 16, 8, 1, false, false,
            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payA);

        // ── b_tile: B_T[K×16] (K,N) via lsc_load_2d VNNI=true ───────────────
        // Surface: width = N=16 cols (fp16), height = K rows
        // VNNI=true: hardware pairs consecutive K-rows →
        //   b_tile uint32[k_pair*16+n] = {B_T[k*2,n], B_T[k*2+1,n]}
        //                             = {B[n,k*2],   B[n,k*2+1]}    VNNI ✓
        // set_x = n_start (fp16 element col), set_y = k_start (row)
        xesimd::config_2d_mem_access<sycl::half, 16, 16, 1> payB(
            B_T,
            (uint32_t)16 * 2u - 1u,   // surfW = N*2-1 = 31 bytes
            (uint32_t)K - 1u,          // surfH = K-1
            (uint32_t)16 * 2u - 1u,   // surfP
            0u, 0u);                   // x=n_start, y=k_start
        simd<sycl::half, 16*16> b_tile = xesimd::lsc_load_2d<
            sycl::half, 16, 16, 1, false/*T*/, true/*VNNI*/,
            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);

        // ── DPAS ─────────────────────────────────────────────────────────────
        simd<sycl::half, 8*16> acc(0.f);
        acc = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(
            acc, b_tile, a_tile);

        // ── Store C[8×16] ─────────────────────────────────────────────────────
        xesimd::lsc_store_2d<sycl::half, 16, 8,
            xesimd::cache_hint::write_back, xesimd::cache_hint::write_back>(
            C, 31u, 7u, 31u, 0u, 0u, acc);
    }
};

// ============================================================
// Case 3: Usage2 + B[N,K]   M=16 N=8 K=16
//   b_tile ← A[16,K]  via lsc_gather<uint32_t,8,u32,N=16>  (SOA→VNNI)
//   a_tile ← B[8,K]   via lsc_load_2d  BW=16 BH=8   no-transform
//   acc[n_i*16+m_j] = C[m_j, n_i]  transposed → lsc_scatter write-back
// ============================================================
struct DPAS_ULT_C3 {
    const sycl::half* A;    // [16 × K]
    const sycl::half* B;    // [8  × K]  B[N,K]
    sycl::half*       C;    // [16 × 8]
    int K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        // ── b_tile: A[16×K] (M×K) via lsc_gather<uint32_t,8,u32,N=16> ───────
        // 16 M-lanes; SOA: b_tile[k_pair*16+m] = {A[m,k*2], A[m,k*2+1]}  VNNI ✓
        const uint32_t* A_u32 = reinterpret_cast<const uint32_t*>(A);
        simd<uint32_t, 16> b_off;
        #pragma unroll
        for (int m = 0; m < 16; m++)
            b_off[m] = (uint32_t)m * (uint32_t)K * 2u;   // byte offset to A[m, 0]
        simd<sycl::half, 16*16> b_tile;
        b_tile.template bit_cast_view<uint32_t>() =
            xesimd::lsc_gather<uint32_t, 8,
                xesimd::lsc_data_size::u32,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached,
                16, uint32_t>(A_u32, b_off);

        // ── a_tile: B[8×K] (N×K) via lsc_load_2d BW=16 BH=8 ────────────────
        // a_tile[n*16+k] = B[n, k] ✓
        xesimd::config_2d_mem_access<sycl::half, 16, 8, 1> payB(
            B, (uint32_t)K*2u-1u, 7u, (uint32_t)K*2u-1u, 0u, 0u);
        simd<sycl::half, 8*16> a_tile = xesimd::lsc_load_2d<
            sycl::half, 16, 8, 1, false, false,
            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(payB);

        // ── DPAS ─────────────────────────────────────────────────────────────
        // dpas(acc, b_tile=A_vnni, a_tile=B)
        // acc[n_i*16+m_j] = sum_k B[n_i,k]*A[m_j,k] = C[m_j, n_i]  (transposed)
        simd<sycl::half, 8*16> acc(0.f);
        acc = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(
            acc, b_tile, a_tile);

        // ── Write-back C[16×8]: pack to u32 + lsc_scatter<uint32_t,4,u32,N=16> ─
        // acc[ni*16+mj] = C[mj, ni]
        // A half vector scatter (lsc_scatter<half,8,u16>) is rejected at vISA
        // ("only D32 and D64 support vector load"), so pack column pairs
        // (2j, 2j+1) of each row into one u32: packed[j*16+mj] = C[mj, 2j..2j+1].
        // scatter SOA: data[j*16+lane] → C_u32[ offset[lane]/4 + j ]
        // Requires 4-byte aligned row starts (even column offset and row stride).
        simd<uint32_t, 4*16> packed;
        auto packed_h = packed.template bit_cast_view<sycl::half>();
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            packed_h.template select<16, 2>(j*32)     = acc.template select<16, 1>((2*j)   * 16);
            packed_h.template select<16, 2>(j*32 + 1) = acc.template select<16, 1>((2*j+1) * 16);
        }
        simd<uint32_t, 16> sc_off;
        #pragma unroll
        for (int mj = 0; mj < 16; mj++)
            sc_off[mj] = (uint32_t)mj * 8u * 2u;   // byte offset to C[mj, 0]
        xesimd::lsc_scatter<uint32_t, 4,
            xesimd::lsc_data_size::u32,
            xesimd::cache_hint::write_back, xesimd::cache_hint::write_back,
            16, uint32_t>(reinterpret_cast<uint32_t*>(C), sc_off, packed);
    }
};

// ============================================================
// Case 4: Usage2 + B_T[K,N]   M=16 N=8 K=16
//   b_tile ← A[16,K]  via lsc_gather<uint32_t,8,u32,N=16>  (SOA→VNNI)
//   a_tile ← B_T[K,8] via transposed u32 load_2d + deinterleave
//
// WHY not a direct fp16 gather or fp16 transpose:
//   lsc_gather<sycl::half,8,u16> is rejected at vISA ("only D32 and D64 support
//   vector load") and Transposed=true is rejected for 2-byte types.
//   A transposed u32 2D load of B_T (4 u32 = 8 fp16 wide, 16 K-rows high) gives
//   p[j*16+k] = {B_T[k][2j], B_T[k][2j+1]} = {B[2j,k], B[2j+1,k]}
//   (different N rows, same k). a_tile[n*16+k] = B[n,k] needs the two halves of
//   each u32 split into rows 2j and 2j+1: 8 strided register moves.
// ============================================================
struct DPAS_ULT_C4 {
    const sycl::half* A;    // [16 × K]
    const sycl::half* B_T;  // [K  × 8]  B_T[K,N]
    sycl::half*       C;    // [16 × 8]
    int K;

    void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
        // ── b_tile: A[16×K] via gather (same as Case 3) ──────────────────────
        const uint32_t* A_u32 = reinterpret_cast<const uint32_t*>(A);
        simd<uint32_t, 16> b_off;
        #pragma unroll
        for (int m = 0; m < 16; m++)
            b_off[m] = (uint32_t)m * (uint32_t)K * 2u;
        simd<sycl::half, 16*16> b_tile;
        b_tile.template bit_cast_view<uint32_t>() =
            xesimd::lsc_gather<uint32_t, 8,
                xesimd::lsc_data_size::u32,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached,
                16, uint32_t>(A_u32, b_off);

        // ── a_tile: B_T[K×8] via transposed u32 load_2d + deinterleave ───────
        // Surface: width = pitch = 8 fp16 = 16 B (pitch must be a multiple of 16 B),
        // height = K rows. Block: 4 u32 wide × 16 rows, transposed.
        // p[j*16+k] = {B_T[k][2j], B_T[k][2j+1]} = {B[2j,k], B[2j+1,k]}
        simd<uint32_t, 4*16> p = load_2d<uint32_t, 4, 16, 1, /*Transposed=*/true, false>(
            reinterpret_cast<const uint32_t*>(B_T), 8u*2u-1u, (uint32_t)K-1u, 8u*2u-1u, 0, 0,
            properties{cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>});
        auto p_h = p.template bit_cast_view<sycl::half>();
        simd<sycl::half, 8*16> a_tile;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            a_tile.template select<16, 1>((2*j)   * 16) = p_h.template select<16, 2>(j*32);
            a_tile.template select<16, 1>((2*j+1) * 16) = p_h.template select<16, 2>(j*32 + 1);
        }
        // a_tile[n*16+k] = B[n, k]  ✓

        // ── DPAS ─────────────────────────────────────────────────────────────
        simd<sycl::half, 8*16> acc(0.f);
        acc = xmx::dpas<8, 8, sycl::half, sycl::half, sycl::half, sycl::half>(
            acc, b_tile, a_tile);
        // acc[n_i*16+m_j] = sum_k B[n_i,k]*A[m_j,k] = C[m_j,n_i]  (transposed)

        // ── Write-back C[16×8]: pack to u32 + lsc_scatter<uint32_t,4,u32,N=16> ─
        // Same as Case 3: packed[j*16+mj] = C[mj, 2j..2j+1].
        simd<uint32_t, 4*16> packed;
        auto packed_h = packed.template bit_cast_view<sycl::half>();
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            packed_h.template select<16, 2>(j*32)     = acc.template select<16, 1>((2*j)   * 16);
            packed_h.template select<16, 2>(j*32 + 1) = acc.template select<16, 1>((2*j+1) * 16);
        }
        simd<uint32_t, 16> sc_off;
        #pragma unroll
        for (int mj = 0; mj < 16; mj++)
            sc_off[mj] = (uint32_t)mj * 8u * 2u;   // byte offset to C[mj, 0]
        xesimd::lsc_scatter<uint32_t, 4,
            xesimd::lsc_data_size::u32,
            xesimd::cache_hint::write_back, xesimd::cache_hint::write_back,
            16, uint32_t>(reinterpret_cast<uint32_t*>(C), sc_off, packed);
    }
};

// ── CPU reference: C[M×N] = A[M×K] * B[N×K]^T  (B given as N×K row-major) ─
static void cpu_gemm(const std::vector<sycl::half>& A,
                     const std::vector<sycl::half>& B,
                     std::vector<float>& C,
                     int M, int N, int K)
{
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float s = 0.f;
            for (int k = 0; k < K; k++)
                s += (float)A[m*K+k] * (float)B[n*K+k];
            C[m*N+n] = s;
        }
}

static bool verify(const std::vector<sycl::half>& gpu,
                   const std::vector<float>& cpu,
                   int M, int N, const char* label)
{
    float rms = 0.f, ref = 0.f;
    for (int i = 0; i < M*N; i++) {
        float d = (float)gpu[i] - cpu[i];
        rms += d*d; ref += cpu[i]*cpu[i];
    }
    float rel = std::sqrt(rms / (ref + 1e-9f)) * 100.f;
    bool pass = rel < 1.f;
    printf("  %-30s rel RMS = %.4f%%  %s\n", label, rel, pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    printf("=== fp16_dpas_ult: all DPAS load pattern cases ===\n\n");
    printf("dpas<8,8,half,half,half,half>(acc, b_tile, a_tile)\n");
    printf("  a_tile: simd<half,128>  [m*16+k]           8×16 row-major\n");
    printf("  b_tile: simd<half,256>  uint32[k_pair*16+n_col]  16×16 VNNI\n");
    printf("  acc   : simd<half,128>  [m*16+n]           8×16 row-major\n\n");

    try {
        queue q(gpu_selector_v, property::queue::in_order{});
        printf("Device: %s\n\n",
               q.get_device().get_info<sycl::info::device::name>().c_str());

        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dis(-1.f, 1.f);
        bool all_pass = true;

        // ── Cases 1 & 2: Usage 1, M=8 N=16 K=16 ─────────────────────────────
        {
            constexpr int M=8, N=16, K=16;
            printf("── Usage 1  M=%d N=%d K=%d  (acc[m*16+n] = C[m,n], direct store)\n",M,N,K);

            std::vector<sycl::half> hA(M*K), hB(N*K), hB_T(K*N), hC(M*N);
            std::vector<float> cpu_C(M*N);
            for (auto& v : hA) v = sycl::half(dis(gen));
            for (auto& v : hB) v = sycl::half(dis(gen));
            for (int n = 0; n < N; n++)          // build B_T[K×N] from B[N×K]
                for (int k = 0; k < K; k++)
                    hB_T[k*N+n] = hB[n*K+k];
            cpu_gemm(hA, hB, cpu_C, M, N, K);

            auto* dA   = malloc_device<sycl::half>(M*K, q);
            auto* dB   = malloc_device<sycl::half>(N*K, q);
            auto* dB_T = malloc_device<sycl::half>(K*N, q);
            auto* dC   = malloc_device<sycl::half>(M*N, q);
            q.memcpy(dA,   hA.data(),   M*K*2).wait();
            q.memcpy(dB,   hB.data(),   N*K*2).wait();
            q.memcpy(dB_T, hB_T.data(), K*N*2).wait();

            // Case 1: B[N,K] gather
            q.memset(dC, 0, M*N*2).wait();
            q.submit([&](handler& h){ h.parallel_for(nd_range<1>(1,1),
                DPAS_ULT_C1{dA, dB, dC, K}); }).wait();
            q.memcpy(hC.data(), dC, M*N*2).wait();
            all_pass &= verify(hC, cpu_C, M, N,
                "C1: b_tile←B[N,K] gather");

            // Case 2: B_T[K,N] load VNNI
            q.memset(dC, 0, M*N*2).wait();
            q.submit([&](handler& h){ h.parallel_for(nd_range<1>(1,1),
                DPAS_ULT_C2{dA, dB_T, dC, K}); }).wait();
            q.memcpy(hC.data(), dC, M*N*2).wait();
            all_pass &= verify(hC, cpu_C, M, N,
                "C2: b_tile←B_T[K,N] load_2d VNNI");

            free(dA,q); free(dB,q); free(dB_T,q); free(dC,q);
        }

        printf("\n");

        // ── Cases 3 & 4: Usage 2, M=16 N=8 K=16 ─────────────────────────────
        {
            constexpr int M=16, N=8, K=16;
            printf("── Usage 2  M=%d N=%d K=%d  (acc[n_i*16+m_j]=C[m_j,n_i], scalar WB)\n",M,N,K);

            std::vector<sycl::half> hA(M*K), hB(N*K), hB_T(K*N), hC(M*N);
            std::vector<float> cpu_C(M*N);
            for (auto& v : hA) v = sycl::half(dis(gen));
            for (auto& v : hB) v = sycl::half(dis(gen));
            for (int n = 0; n < N; n++)
                for (int k = 0; k < K; k++)
                    hB_T[k*N+n] = hB[n*K+k];
            cpu_gemm(hA, hB, cpu_C, M, N, K);

            auto* dA   = malloc_device<sycl::half>(M*K, q);
            auto* dB   = malloc_device<sycl::half>(N*K, q);
            auto* dB_T = malloc_device<sycl::half>(K*N, q);
            auto* dC   = malloc_device<sycl::half>(M*N, q);
            q.memcpy(dA,   hA.data(),   M*K*2).wait();
            q.memcpy(dB,   hB.data(),   N*K*2).wait();
            q.memcpy(dB_T, hB_T.data(), K*N*2).wait();

            // Case 3: B[N,K] load_2d
            q.memset(dC, 0, M*N*2).wait();
            q.submit([&](handler& h){ h.parallel_for(nd_range<1>(1,1),
                DPAS_ULT_C3{dA, dB, dC, K}); }).wait();
            q.memcpy(hC.data(), dC, M*N*2).wait();
            all_pass &= verify(hC, cpu_C, M, N,
                "C3: a_tile←B[N,K] load_2d");

            // Case 4: B_T[K,N] load_2d transpose
            q.memset(dC, 0, M*N*2).wait();
            q.submit([&](handler& h){ h.parallel_for(nd_range<1>(1,1),
                DPAS_ULT_C4{dA, dB_T, dC, K}); }).wait();
            q.memcpy(hC.data(), dC, M*N*2).wait();
            all_pass &= verify(hC, cpu_C, M, N,
                "C4: a_tile←B_T[K,N] load_2d transpose");

            free(dA,q); free(dB,q); free(dB_T,q); free(dC,q);
        }

        printf("\n%s\n", all_pass ? "ALL PASS" : "SOME FAILED");
        return all_pass ? 0 : 1;

    } catch (const sycl::exception& e) {
        fprintf(stderr, "SYCL: %s\n", e.what());
        return 1;
    }
}
