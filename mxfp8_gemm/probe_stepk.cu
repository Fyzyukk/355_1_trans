// task#5: pin down step_k semantics.
// Verify numerically that   Σ_kg step_k<kg>(v_a, v_b, 0)  ==  mma(v_a, v_b)
// using the EXACT tiled swap_ab mma the kernel uses (E_M,E_N,E_K / W 16x16x32).
// If this holds, splitting the K-groups via step_k and applying per-group scale
// before accumulation is mathematically equivalent to the fused mma.

#include "opus/opus.hpp"
#include "gemm_a8w8_mxfp8_common.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace opus;

#define CHECK_HIP(call) do { hipError_t s_ = call; if (s_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(s_)); exit(1);} } while(0)

using T = gemm_a8w8_mxfp8_traits<>;

__global__ void probe(const unsigned char* __restrict__ dA,
                      const unsigned char* __restrict__ dB,
                      float* __restrict__ dFull,
                      float* __restrict__ dSum,
                      int* __restrict__ dMismatch) {
#if defined(__gfx950__)
    int lane = (int)__builtin_amdgcn_workitem_id_x();

    auto mma = make_tiled_mma<fp8_t, fp8_t, fp32_t>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    constexpr int A_LEN = decltype(mma)::tile_a_len;   // E_M*E_K*8
    constexpr int B_LEN = decltype(mma)::tile_b_len;   // E_N*E_K*8
    constexpr int C_LEN = decltype(mma)::tile_c_len;   // E_M*E_N*elem_c

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b;
    #pragma unroll
    for (int i = 0; i < A_LEN; i++) v_a[i] = __builtin_bit_cast(fp8_t, dA[lane * A_LEN + i]);
    #pragma unroll
    for (int i = 0; i < B_LEN; i++) v_b[i] = __builtin_bit_cast(fp8_t, dB[lane * B_LEN + i]);

    // reference: fused mma over all K-groups
    auto v_full = mma(v_a, v_b);

    // sum of per-group step_k, each accumulating only its K-group onto a fresh zero base
    typename decltype(mma)::vtype_c v_sum;
    clear(v_sum);
    typename decltype(mma)::vtype_c zero_c;
    clear(zero_c);
    static_for<T::E_K>([&](auto kg){
        auto v_part = mma.step_k(number<decltype(kg)::value>{}, v_a, v_b, zero_c);
        #pragma unroll
        for (int i = 0; i < C_LEN; i++) v_sum[i] += v_part[i];
    });

    int local_mismatch = 0;
    #pragma unroll
    for (int i = 0; i < C_LEN; i++) {
        dFull[lane * C_LEN + i] = v_full[i];
        dSum[lane * C_LEN + i]  = v_sum[i];
        if (v_full[i] != v_sum[i]) local_mismatch++;
    }
    atomicAdd(dMismatch, local_mismatch);
#endif
}

int main() {
    constexpr int A_LEN = T::E_M * T::E_K * 8;
    constexpr int B_LEN = T::E_N * T::E_K * 8;
    constexpr int C_LEN = T::E_M * T::E_N * 4;
    const int lanes = 64;

    std::vector<unsigned char> hA(lanes * A_LEN), hB(lanes * B_LEN);
    // small fp8-e4m3 exact-ish values: use bytes that encode small magnitudes.
    // 0x38=1, 0x40=2, 0x44=3, 0x48=4, 0x3c=1.5 ... keep values tiny to avoid overflow.
    unsigned char pool[8] = {0x00, 0x38, 0x40, 0x44, 0x48, 0x30, 0x34, 0x3c};
    for (int i = 0; i < (int)hA.size(); i++) hA[i] = pool[(i * 7 + 3) % 8];
    for (int i = 0; i < (int)hB.size(); i++) hB[i] = pool[(i * 5 + 1) % 8];
    // Break uniformity: zero out most A K-groups per lane so per-group contributions differ,
    // giving distinct fragment values instead of a symmetric constant.
    for (int lane = 0; lane < lanes; lane++)
        for (int i = 0; i < A_LEN; i++)
            if (((i / 8) + lane) % 3 != 0) hA[lane * A_LEN + i] = 0x00;

    unsigned char *dA, *dB; float *dFull, *dSum; int *dMismatch;
    CHECK_HIP(hipMalloc(&dA, hA.size()));
    CHECK_HIP(hipMalloc(&dB, hB.size()));
    CHECK_HIP(hipMalloc(&dFull, lanes * C_LEN * sizeof(float)));
    CHECK_HIP(hipMalloc(&dSum,  lanes * C_LEN * sizeof(float)));
    CHECK_HIP(hipMalloc(&dMismatch, sizeof(int)));
    CHECK_HIP(hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB.data(), hB.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(dMismatch, 0, sizeof(int)));

    probe<<<1, lanes>>>(dA, dB, dFull, dSum, dMismatch);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    int mismatch = -1;
    CHECK_HIP(hipMemcpy(&mismatch, dMismatch, sizeof(int), hipMemcpyDeviceToHost));
    std::vector<float> full(lanes * C_LEN), sum(lanes * C_LEN);
    CHECK_HIP(hipMemcpy(full.data(), dFull, full.size() * sizeof(float), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(sum.data(),  dSum,  sum.size()  * sizeof(float), hipMemcpyDeviceToHost));

    printf("E_M=%d E_N=%d E_K=%d  A_LEN=%d B_LEN=%d C_LEN=%d\n",
           T::E_M, T::E_N, T::E_K, A_LEN, B_LEN, C_LEN);
    printf("total elements compared: %d,  mismatches: %d\n", lanes * C_LEN, mismatch);
    // show a few sample lanes
    for (int lane = 0; lane < 4; lane++) {
        printf("L%02d full:", lane);
        for (int i = 0; i < C_LEN; i++) printf(" %.0f", full[lane * C_LEN + i]);
        printf("\n     sum :");
        for (int i = 0; i < C_LEN; i++) printf(" %.0f", sum[lane * C_LEN + i]);
        printf("\n");
    }
    printf(mismatch == 0 ? "\nRESULT: MATCH  (Σ step_k == mma)\n"
                         : "\nRESULT: MISMATCH\n");
    return mismatch == 0 ? 0 : 1;
}
