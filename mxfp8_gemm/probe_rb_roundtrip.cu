// Round-trip probe for make_layout_sb / make_layout_rb (B path). Mirror of probe_ra.
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

constexpr int LDS_B = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int TILE_B_ELEMS = T::HALF_B_N * T::B_K;
constexpr int VB_PER_WAVE = T::E_N * T::E_K * 8;   // elem_b=8

__global__ void probe_kernel(int* lds_to_nk, int* rb_nk, int* canon_nk) {
    const int tid = threadIdx.x;
    const int wave_id = tid / T::WARP_SIZE;
    const int lane_id = tid % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    const int stride_b = T::B_K;

    auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, stride_b);
    auto u_sb = make_layout_sb<T>(wave_id_m, wave_id_n);
    auto gb_off = opus::layout_to_offsets<T::VEC_B_GLOBAL>(u_gb);
    auto sb_off = opus::layout_to_offsets<T::VEC_B_GLOBAL>(u_sb);
    constexpr int n_gb = decltype(gb_off)::size();
    for (int i = 0; i < n_gb; i++)
        for (int j = 0; j < T::VEC_B_GLOBAL; j++) {
            int g = gb_off[i] + j;                          // n*B_K + k
            int l = sb_off[i] + lane_id * T::VEC_B_GLOBAL + j;
            if (l >= 0 && l < LDS_B) lds_to_nk[l] = g;
        }
    __syncthreads();

    if (wave_id == 0) {
        auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);
        auto rb_off = opus::layout_to_offsets<T::VEC_B>(u_rb);
        constexpr int n_rb = decltype(rb_off)::size();
        for (int i = 0; i < n_rb; i++)
            for (int j = 0; j < T::VEC_B; j++) {
                int slot = i * T::VEC_B + j;
                int l = rb_off[i] + j;
                rb_nk[lane_id * VB_PER_WAVE + slot] = (l >= 0 && l < LDS_B) ? lds_to_nk[l] : -1;
            }

        auto mma = opus::make_tiled_mma<opus::fp8_t, opus::fp8_t, opus::fp32_t>(
            opus::seq<T::E_M, T::E_N, T::E_K>{}, opus::seq<T::T_M, T::T_N, T::T_K>{},
            opus::seq<T::W_M, T::W_N, T::W_K>{}, opus::mfma_adaptor_swap_ab{});
        auto p_coord_b = opus::make_tuple(wave_id_n, lane_id % T::W_N, 0, lane_id / T::W_N);
        auto u_cb = opus::partition_layout_b<T::VEC_B>(mma, opus::make_tuple(stride_b, 1_I), p_coord_b);
        auto cb_off = opus::layout_to_offsets<T::VEC_B>(u_cb);
        constexpr int n_cb = decltype(cb_off)::size();
        for (int i = 0; i < n_cb; i++)
            for (int j = 0; j < T::VEC_B; j++) {
                int slot = i * T::VEC_B + j;
                canon_nk[lane_id * VB_PER_WAVE + slot] = cb_off[i] + j;
            }
    }
}

int main() {
    printf("Traits: E_N=%d E_K=%d T_N=%d W_N=%d W_K=%d VEC_B=%d VEC_B_GLOBAL=%d\n",
           T::E_N, T::E_K, T::T_N, T::W_N, T::W_K, T::VEC_B, T::VEC_B_GLOBAL);
    printf("LDS_B=%d TILE_B_ELEMS=%d VB_PER_WAVE=%d\n\n", LDS_B, TILE_B_ELEMS, VB_PER_WAVE);

    int *d_lds, *d_rb, *d_canon;
    CHECK_HIP(hipMalloc(&d_lds, LDS_B * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_rb, 64 * VB_PER_WAVE * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_canon, 64 * VB_PER_WAVE * sizeof(int)));
    CHECK_HIP(hipMemset(d_lds, 0xFF, LDS_B * sizeof(int)));
    CHECK_HIP(hipMemset(d_rb, 0xFF, 64 * VB_PER_WAVE * sizeof(int)));
    CHECK_HIP(hipMemset(d_canon, 0xFF, 64 * VB_PER_WAVE * sizeof(int)));

    probe_kernel<<<1, T::BLOCK_SIZE>>>(d_lds, d_rb, d_canon);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    auto h_rb = (int*)malloc(64 * VB_PER_WAVE * sizeof(int));
    auto h_canon = (int*)malloc(64 * VB_PER_WAVE * sizeof(int));
    auto h_lds = (int*)malloc(LDS_B * sizeof(int));
    CHECK_HIP(hipMemcpy(h_rb, d_rb, 64 * VB_PER_WAVE * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_canon, d_canon, 64 * VB_PER_WAVE * sizeof(int), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(h_lds, d_lds, LDS_B * sizeof(int), hipMemcpyDeviceToHost));

    int written = 0; for (int i = 0; i < LDS_B; i++) if (h_lds[i] != -1) written++;
    printf("LDS slots written: %d (tile has %d elements)\n\n", written, TILE_B_ELEMS);

    auto decode = [](int g, int& n, int& k) { if (g < 0) { n = k = -1; } else { n = g / T::B_K; k = g % T::B_K; } };

    // Oracle self-check vs probe table: B:(L,i)->B[gk*8+i][col], col=L%16, gk=L/16.
    printf("=== ORACLE self-check (canonical layout_b) vs probe-table prediction ===\n");
    int mism_o = 0;
    for (int L = 0; L < 64; L++) {
        int col = L % T::W_N, gk = L / T::W_N;
        for (int en = 0; en < T::E_N; en++)
        for (int ek = 0; ek < T::E_K; ek++)
        for (int i = 0; i < 8; i++) {
            int slot = (en * T::E_K + ek) * 8 + i;
            int n, k; decode(h_canon[L * VB_PER_WAVE + slot], n, k);
            int exp_n = en * (T::W_N * T::T_N) + 0 * T::W_N + col;
            int exp_k = ek * T::W_K + gk * 8 + i;
            if (n != exp_n || k != exp_k) {
                if (mism_o < 12) printf("  MISMATCH L=%d slot=%d en=%d ek=%d i=%d: canon=(n%d,k%d) pred=(n%d,k%d)\n",
                    L, slot, en, ek, i, n, k, exp_n, exp_k);
                mism_o++;
            }
        }
    }
    printf("oracle self-check mismatches: %d %s\n\n", mism_o, mism_o == 0 ? "(oracle trustworthy)" : "(!! inspect)");

    printf("=== rb (round-trip through sb) vs canonical layout_b ===\n");
    int mism = 0, unreach = 0;
    for (int L = 0; L < 64; L++)
        for (int s = 0; s < VB_PER_WAVE; s++) {
            int rg = h_rb[L * VB_PER_WAVE + s], cg = h_canon[L * VB_PER_WAVE + s];
            if (rg < 0) unreach++;
            if (rg != cg) {
                if (mism < 24) { int rn, rk, cn, ck; decode(rg, rn, rk); decode(cg, cn, ck);
                    printf("  L=%2d slot=%2d: rb=(n%3d,k%3d)  canon=(n%3d,k%3d)\n", L, s, rn, rk, cn, ck); }
                mism++;
            }
        }
    printf("\nrb vs canonical: %d / %d mismatches, %d unreachable\n", mism, 64 * VB_PER_WAVE, unreach);
    printf("RESULT: %s\n", mism == 0 ? "rb/sb CONSISTENT with hardware layout" : "rb/sb BROKEN");
    return 0;
}
