// Round-trip probe for make_layout_sa / make_layout_ra (A path) of the mxfp8 kernel.
//
// Goal (assumption-free): find whether the sa(write) and ra(read) LDS layouts are
// mutually consistent AND consistent with the hardware MMA A-operand layout.
//
// Method:
//   1. WRITE side: replicate exactly what async_load does for stage0/half0 A:
//      each thread's u_ga gives the source element offset (== m*B_K + k inside the
//      128x128 A tile), u_sa gives the destination LDS offset. async_load moves
//      VEC_A_GLOBAL contiguous elements. So for every LDS slot we record which
//      global (m,k) landed there:  lds_to_mk[lds_off + j] = ga_off + j.
//   2. READ side: replicate what `load<VEC_A>(s_a, u_ra + ...)` does for wave0:
//      u_ra gives LDS offsets; each issue reads VEC_A contiguous. Decode the (m,k)
//      each v_a element picks up via the lds_to_mk table.
//   3. ORACLE: canonical tiled_mma.layout_a offsets (with stride {B_K,1}) decode
//      directly to (m,k). Self-check against the §5.2 probe table, then compare
//      the ra-decoded table against it. Divergence pinpoints the bug.
//
// No MMA math, no fp8 data round-trip: only address bookkeeping (exact in int32).

#include <opus/opus.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "gemm_a8w8_mxfp8_common.h"
#include "gemm_a8w8_mxfp8_kernel_template.hpp"

using opus::operator""_I;
using T = gemm_a8w8_mxfp8_traits<>;

#define CHECK_HIP(call) do { hipError_t s_ = call; if (s_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(s_)); exit(1);} } while(0)

// LDS map size for one stage/half A region.
constexpr int LDS_A = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int TILE_A_ELEMS = T::HALF_B_M * T::B_K;          // 128*128
constexpr int VA_PER_WAVE = T::E_M * T::E_K * 8;            // v_a element count per wave (elem_a=8)

// -1 sentinel = LDS slot never written.
__global__ void probe_kernel(int* lds_to_mk,      // [LDS_A]  lds_off -> global(m*B_K+k)
                             int* ra_mk,          // [64 * VA_PER_WAVE] wave0: (lane,slot)->(m,k) packed
                             int* canon_mk) {     // [64 * VA_PER_WAVE] canonical layout_a decode
    const int tid = threadIdx.x;
    const int wave_id = tid / T::WARP_SIZE;
    const int lane_id = tid % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    const int stride_a = T::B_K;   // single 128x128 tile, row-major

    // ---- WRITE side: build lds_to_mk exactly as async_load(stage0,half0) would ----
    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, stride_a);
    auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    auto ga_off = opus::layout_to_offsets<T::VEC_A_GLOBAL>(u_ga);
    auto sa_off = opus::layout_to_offsets<T::VEC_A_GLOBAL>(u_sa);
    constexpr int n_ga = decltype(ga_off)::size();
    for (int i = 0; i < n_ga; i++) {
        for (int j = 0; j < T::VEC_A_GLOBAL; j++) {
            int g = ga_off[i] + j;          // element offset in A tile == m*B_K + k
            // buffer_load_lds hardware scatters lane L to dst + L*VEC (u_sa has no lane_id).
            int l = sa_off[i] + lane_id * T::VEC_A_GLOBAL + j;
            if (l >= 0 && l < LDS_A) lds_to_mk[l] = g;
        }
    }
    __syncthreads();

    // ---- READ side (wave0 only): decode what ra delivers ----
    if (wave_id == 0) {
        auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
        auto ra_off = opus::layout_to_offsets<T::VEC_A>(u_ra);
        constexpr int n_ra = decltype(ra_off)::size();
        for (int i = 0; i < n_ra; i++) {
            for (int j = 0; j < T::VEC_A; j++) {
                int slot = i * T::VEC_A + j;
                int l = ra_off[i] + j;
                int g = (l >= 0 && l < LDS_A) ? lds_to_mk[l] : -1;
                ra_mk[lane_id * VA_PER_WAVE + slot] = g;
            }
        }

        // ---- ORACLE: canonical tiled layout_a for wave0 ----
        auto mma = opus::make_tiled_mma<opus::fp8_t, opus::fp8_t, opus::fp32_t>(
            opus::seq<T::E_M, T::E_N, T::E_K>{}, opus::seq<T::T_M, T::T_N, T::T_K>{},
            opus::seq<T::W_M, T::W_N, T::W_K>{}, opus::mfma_adaptor_swap_ab{});
        // p_coord_a order: (tile_m, grpm_a, tile_k, grpk_a) = (wave_id_m, lane%16, 0, lane/16)
        auto p_coord_a = opus::make_tuple(wave_id_m, lane_id % T::W_M, 0, lane_id / T::W_M);
        auto u_ca = opus::partition_layout_a<T::VEC_A>(mma, opus::make_tuple(stride_a, 1_I), p_coord_a);
        auto ca_off = opus::layout_to_offsets<T::VEC_A>(u_ca);
        constexpr int n_ca = decltype(ca_off)::size();
        for (int i = 0; i < n_ca; i++) {
            for (int j = 0; j < T::VEC_A; j++) {
                int slot = i * T::VEC_A + j;
                canon_mk[lane_id * VA_PER_WAVE + slot] = ca_off[i] + j;  // == m*B_K + k
            }
        }
    }
}

int main() {
    printf("Traits: E_M=%d E_N=%d E_K=%d T_M=%d T_N=%d W_K=%d VEC_A=%d VEC_A_GLOBAL=%d\n",
           T::E_M, T::E_N, T::E_K, T::T_M, T::T_N, T::W_K, T::VEC_A, T::VEC_A_GLOBAL);
    printf("LDS_A=%d TILE_A_ELEMS=%d VA_PER_WAVE=%d\n\n", LDS_A, TILE_A_ELEMS, VA_PER_WAVE);

    int *d_lds, *d_ra, *d_canon;
    CHECK_HIP(hipMalloc(&d_lds, LDS_A * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_ra, 64 * VA_PER_WAVE * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_canon, 64 * VA_PER_WAVE * sizeof(int)));
    CHECK_HIP(hipMemset(d_lds, 0xFF, LDS_A * sizeof(int)));   // -1
    CHECK_HIP(hipMemset(d_ra, 0xFF, 64 * VA_PER_WAVE * sizeof(int)));
    CHECK_HIP(hipMemset(d_canon, 0xFF, 64 * VA_PER_WAVE * sizeof(int)));

    probe_kernel<<<1, T::BLOCK_SIZE>>>(d_lds, d_ra, d_canon);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    auto h_ra = (int*)malloc(64 * VA_PER_WAVE * sizeof(int));
    auto h_canon = (int*)malloc(64 * VA_PER_WAVE * sizeof(int));
    auto h_lds = (int*)malloc(LDS_A * sizeof(int));
    CHECK_HIP(hipMemcpy(h_ra, d_ra, 64 * VA_PER_WAVE * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_canon, d_canon, 64 * VA_PER_WAVE * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_lds, d_lds, LDS_A * sizeof(int), hipMemcpyDeviceToHost));

    // Coverage: how many LDS slots got written, any (m,k) unreachable?
    int written = 0; for (int i = 0; i < LDS_A; i++) if (h_lds[i] != -1) written++;
    printf("LDS slots written: %d (tile has %d elements)\n\n", written, TILE_A_ELEMS);

    auto decode = [](int g, int& m, int& k) { if (g < 0) { m = k = -1; } else { m = g / T::B_K; k = g % T::B_K; } };

    // Self-check the oracle against §5.2 probe table for the BASE 16x16x32 slice.
    // Probe table: A:(L,i) -> A[row][gk*8+i], row=L%16, gk=L/16. With E_K/E_M repeats,
    // canonical stacks: slot layout = [E_M][E_K][pack8]. For lane L, expect within each
    // (em,ek) block: k = ek*32 + gk*8 + i ; m = em*(W_M*T_M) + wave_id_m*W_M + row.
    // wave0 => wave_id_m=0. Verify a few lanes.
    printf("=== ORACLE self-check (canonical layout_a) vs probe-table prediction ===\n");
    int mism_oracle = 0;
    for (int L = 0; L < 64; L++) {
        int row = L % T::W_M, gk = L / T::W_M;
        for (int em = 0; em < T::E_M; em++)
        for (int ek = 0; ek < T::E_K; ek++)
        for (int i = 0; i < 8; i++) {
            int slot = (em * T::E_K + ek) * 8 + i;
            int m, k; decode(h_canon[L * VA_PER_WAVE + slot], m, k);
            int exp_m = em * (T::W_M * T::T_M) + 0 * T::W_M + row;
            int exp_k = ek * (T::W_K) + gk * 8 + i;   // W_K=32
            if (m != exp_m || k != exp_k) {
                if (mism_oracle < 12)
                    printf("  MISMATCH L=%d slot=%d em=%d ek=%d i=%d: canon=(m%d,k%d) pred=(m%d,k%d)\n",
                           L, slot, em, ek, i, m, k, exp_m, exp_k);
                mism_oracle++;
            }
        }
    }
    printf("oracle self-check mismatches: %d %s\n\n", mism_oracle,
           mism_oracle == 0 ? "(oracle trustworthy)" : "(!! probe-table prediction wrong OR canonical construction wrong — inspect)");

    // Main comparison: ra vs canonical.
    printf("=== ra (round-trip through sa) vs canonical layout_a ===\n");
    int mism = 0, unreach = 0;
    for (int L = 0; L < 64; L++) {
        for (int s = 0; s < VA_PER_WAVE; s++) {
            int rg = h_ra[L * VA_PER_WAVE + s];
            int cg = h_canon[L * VA_PER_WAVE + s];
            if (rg < 0) unreach++;
            if (rg != cg) {
                if (mism < 24) {
                    int rm, rk, cm, ck; decode(rg, rm, rk); decode(cg, cm, ck);
                    printf("  L=%2d slot=%2d: ra=(m%3d,k%3d)  canon=(m%3d,k%3d)\n", L, s, rm, rk, cm, ck);
                }
                mism++;
            }
        }
    }
    printf("\nra vs canonical: %d / %d mismatches, %d unreachable LDS reads\n",
           mism, 64 * VA_PER_WAVE, unreach);
    printf("RESULT: %s\n", mism == 0 ? "ra/sa CONSISTENT with hardware layout" : "ra/sa BROKEN (see mismatches above)");

    // Dump lane0 full picture for eyeballing.
    printf("\n=== lane0 detail (slot: ra(m,k) | canon(m,k)) ===\n");
    for (int s = 0; s < VA_PER_WAVE; s++) {
        int rm, rk, cm, ck; decode(h_ra[s], rm, rk); decode(h_canon[s], cm, ck);
        printf(" slot%2d: ra(%3d,%3d) canon(%3d,%3d)%s\n", s, rm, rk, cm, ck,
               (h_ra[s] == h_canon[s]) ? "" : "   <-- DIFF");
    }
    return 0;
}
