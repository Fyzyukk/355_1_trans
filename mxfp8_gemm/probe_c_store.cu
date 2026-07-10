// Probe for the C-store layout: partition_layout_c + c_offset.
// Verify (lane, fragment i) -> (m,n) matches the §5.2 probe table for the
// swap_ab 16x16x32 accumulator, and that it is a bijection over each 128x128 subtile.
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

constexpr int VC_PER_WAVE = T::E_M * T::E_N * 4;    // elem_c=4  -> 32 fragments/lane
constexpr int STRIDE_C = T::HALF_B_N;               // one 128x128 subtile, row-major width 128

constexpr int NUM_WAVES = T::T_M * T::T_N;           // 8 waves cover one 128x128 subtile

__global__ void probe_kernel(int* c_mn) {           // all waves: (wave,lane,frag)-> m*STRIDE_C + n
    const int tid = threadIdx.x;
    const int wave_id = tid / T::WARP_SIZE;
    const int lane_id = tid % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    if (wave_id >= NUM_WAVES) return;

    // Handwritten C-store layout (VEC_C=4: the 4 pk fragments are contiguous columns).
    auto u_gc = make_layout_gc<T>(lane_id, wave_id_m, wave_id_n, STRIDE_C);
    auto gc_off = opus::layout_to_offsets<T::VEC_C>(u_gc);
    constexpr int n_gc = decltype(gc_off)::size();
    for (int i = 0; i < n_gc; i++)
        for (int j = 0; j < T::VEC_C; j++) {
            int slot = i * T::VEC_C + j;
            c_mn[(wave_id * 64 + lane_id) * VC_PER_WAVE + slot] = gc_off[i] + j;
        }
}

int main() {
    printf("Traits: E_M=%d E_N=%d T_M=%d T_N=%d W_M=%d W_N=%d VEC_C=%d\n",
           T::E_M, T::E_N, T::T_M, T::T_N, T::W_M, T::W_N, T::VEC_C);
    printf("VC_PER_WAVE=%d  subtile=%dx%d\n\n", VC_PER_WAVE, T::HALF_B_M, T::HALF_B_N);

    const int total = NUM_WAVES * 64 * VC_PER_WAVE;
    int *d_c; CHECK_HIP(hipMalloc(&d_c, total * sizeof(int)));
    CHECK_HIP(hipMemset(d_c, 0xFF, total * sizeof(int)));
    probe_kernel<<<1, T::BLOCK_SIZE>>>(d_c);
    CHECK_HIP(hipGetLastError()); CHECK_HIP(hipDeviceSynchronize());
    auto h = (int*)malloc(total * sizeof(int));
    CHECK_HIP(hipMemcpy(h, d_c, total * sizeof(int), hipMemcpyDeviceToHost));

    auto decode = [](int g, int& m, int& n) { m = g / STRIDE_C; n = g % STRIDE_C; };

    // Bijection over 128x128 across all 8 waves?
    int subtile = T::HALF_B_M * T::HALF_B_N;
    auto hit = (int*)calloc(subtile, sizeof(int));
    int oob = 0;
    for (int W = 0; W < NUM_WAVES; W++)
    for (int L = 0; L < 64; L++)
        for (int s = 0; s < VC_PER_WAVE; s++) {
            int g = h[(W * 64 + L) * VC_PER_WAVE + s];
            if (g < 0 || g >= subtile) oob++; else hit[g]++;
        }
    int missed = 0, dup = 0;
    for (int i = 0; i < subtile; i++) { if (hit[i] == 0) missed++; if (hit[i] > 1) dup++; }
    printf("=== C-store bijection over one 128x128 subtile (all %d waves) ===\n", NUM_WAVES);
    printf("cells covered: %d/%d, missed=%d, duplicated=%d, oob=%d\n",
           subtile - missed, subtile, missed, dup, oob);
    printf("bijection: %s\n\n", (missed == 0 && dup == 0 && oob == 0) ? "OK" : "BROKEN");

    // Fragment ordering vs empirical probe_c_mma ground truth (swap_ab tiled mma).
    // scale_c_group indexes fragment i = m_rep*(E_N*4) + n_rep*4 + pk.
    // Real accumulator layout (C-layout 0): m = row, n = gk*4 + pk.
    //   m = m_rep*(W_M*T_M) + wave_id_m*W_M + row , row = lane%W_M
    //   n = n_rep*(W_N*T_N) + wave_id_n*W_N + gk*4 + pk , gk = lane/W_M
    // wave0 => wave_id_m=wave_id_n=0.
    printf("=== C fragment ordering vs empirical mma ground truth ===\n");
    int mism = 0;
    for (int L = 0; L < 64; L++) {
        int row = L % T::W_M, gk = L / T::W_M;
        for (int mr = 0; mr < T::E_M; mr++)
        for (int nr = 0; nr < T::E_N; nr++)
        for (int pk = 0; pk < 4; pk++) {
            int slot = mr * (T::E_N * 4) + nr * 4 + pk;
            int m, n; decode(h[L * VC_PER_WAVE + slot], m, n);
            int exp_m = mr * (T::W_M * T::T_M) + row;
            int exp_n = nr * (T::W_N * T::T_N) + gk * 4 + pk;
            if (m != exp_m || n != exp_n) {
                if (mism < 16) printf("  L=%2d slot=%2d mr=%d nr=%d pk=%d: got(m%3d,n%3d) pred(m%3d,n%3d)\n",
                    L, slot, mr, nr, pk, m, n, exp_m, exp_n);
                mism++;
            }
        }
    }
    printf("fragment-ordering mismatches: %d %s\n", mism,
           mism == 0 ? "(C-store matches empirical mma)" : "(C-store fragment order WRONG)");
    return 0;
}
