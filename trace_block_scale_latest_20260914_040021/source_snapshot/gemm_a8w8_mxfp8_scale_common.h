#pragma once

struct BlockscaleCommon {
    // Scale vector widths in bytes: one packed operand, or both tile halves.
    static constexpr int VEC_SF = 4;
    static constexpr int VEC_SF_PAIR = 2 * VEC_SF;
};

// Standalone ABI for the gfx950 four-Wave64 blockscale bpreshuffle kernel.
// A/B/SFA/SFB are inputs and C is output; no global scratch/workspace is used.
// Keep the field order identical to the validated historical 96-byte ABI.
struct opus_gemm_scale_kargs {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    int m;
    int n;
    int k;
    int batch;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_a_batch;
    int stride_b_batch;
    int stride_c_batch;
    const void* __restrict__ ptr_sfa;
    const void* __restrict__ ptr_sfb;
    // SFA is column-major [K/128,M]: stride_sfa is the K-column stride.
    // SFB is contiguous [N/128,K/128]: stride_sfb is the N-block row stride.
    int stride_sfa;
    int stride_sfb;
    int stride_sfa_batch;
    int stride_sfb_batch;
};

static_assert(sizeof(opus_gemm_scale_kargs) == 96);
static_assert(__builtin_offsetof(opus_gemm_scale_kargs, ptr_sfa) == 64);
static_assert(__builtin_offsetof(opus_gemm_scale_kargs, stride_sfa) == 80);

__host__ __device__ inline int ceil_div_scale(int a, int b) {
    return (a + b - 1) / b;
}
