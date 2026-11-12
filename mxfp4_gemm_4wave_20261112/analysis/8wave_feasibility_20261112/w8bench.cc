// 1 vs 2 waves/SIMD under the power cap: a fake MXFP4 GEMM inner loop.
// Each wave runs "steps" of 32 fp4 scaled MFMAs (16 accumulators, kk0/kk1 pairs back to back on the
// same C, like m10), NDS ds_read_b128 per step into the double-buffered operand registers (from a
// random LDS region), NBL global->LDS DMA b128 per step that walk the A/B K tiles of a real 8192^3
// GEMM tile schedule (same L2 reuse pattern), and one s_barrier per K tile (128 MFMA per SIMD).
// The SAME compiled kernel is launched with 256 threads (1 wave/SIMD) or 512 threads (2 waves/SIMD);
// with 512 threads each wave runs half the steps, so the work per SIMD is identical.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

#ifndef NDS
#define NDS 8
#endif
#ifndef NBL
#define NBL 4
#endif
#ifndef DEPPAIR
#define DEPPAIR 1   // 1: kk0/kk1 on the same C back to back; 0: kk1 goes to another C (no chain)
#endif
static_assert(NBL == 4 || NBL == 0, "DMA mapping assumes 4 per step");
#ifndef BAR
#define BAR 1
#endif

constexpr int LDS_R = 32 * 1024;   // ds_read source
constexpr int LDS_D = 64 * 1024;   // DMA destination (one K tile of A+B)
constexpr int LDS_PAD = 32 * 1024; // forces 1 block per CU (128 KB > 80 KB)

__device__ __forceinline__ void sgb_mfma() { __builtin_amdgcn_sched_group_barrier(0x008, 1, 0); }
__device__ __forceinline__ void sgb_ds() { __builtin_amdgcn_sched_group_barrier(0x100, 1, 0); }
__device__ __forceinline__ void sgb_vm() { __builtin_amdgcn_sched_group_barrier(0x020, 1, 0); }

template<int CUR>
__device__ __forceinline__ void step(i32x4 (&fr)[2][16], f32x4 (&c)[16], const char* __restrict__ lds_rw,
                                     const char* __restrict__ gsrc, int g_lane_off, int row_stride_step,
                                     char* lds_d, int lds_d_off, int lds_d_step, int sa, int sb) {
    constexpr int NXT = CUR ^ 1;
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int f = 0; f < NDS; ++f)
        fr[NXT][f % 16] = *reinterpret_cast<const i32x4*>(lds_rw + ((CUR * NDS + f) * 1024) % LDS_R);
#pragma unroll
    for (int j = 0; j < NBL; ++j) {
        const char* src = gsrc + (size_t)(j * row_stride_step) + g_lane_off;
        __builtin_amdgcn_global_load_lds((__attribute__((address_space(1))) void*)(src),
                                         (__attribute__((address_space(3))) void*)((__attribute__((address_space(3))) char*)lds_d + lds_d_off + j * lds_d_step),
                                         16, 0, 0);
    }
#pragma unroll
    for (int m = 0; m < 4; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const i32x4 a0 = fr[CUR][m], b0 = fr[CUR][4 + n], a1 = fr[CUR][8 + m], b1 = fr[CUR][12 + n];
            const i32x8 a08 = __builtin_shufflevector(a0, a0, 0, 1, 2, 3, -1, -1, -1, -1);
            const i32x8 b08 = __builtin_shufflevector(b0, b0, 0, 1, 2, 3, -1, -1, -1, -1);
            const i32x8 a18 = __builtin_shufflevector(a1, a1, 0, 1, 2, 3, -1, -1, -1, -1);
            const i32x8 b18 = __builtin_shufflevector(b1, b1, 0, 1, 2, 3, -1, -1, -1, -1);
            const int q = m * 4 + n;
            c[q] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a08, b08, c[q], 4, 4, 0, sa, 0, sb);
#if DEPPAIR
            c[q] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a18, b18, c[q], 4, 4, 0, sa, 0, sb);
#else
            c[q ^ 8] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a18, b18, c[q ^ 8], 4, 4, 0, sa, 0, sb);
#endif
        }
    }
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        sgb_mfma();
#pragma unroll
        for (int f = 0; f < NDS; ++f)
            if ((f * 32) / NDS == i) sgb_ds();
#pragma unroll
        for (int j = 0; j < NBL; ++j)
            if ((j * 32) / NBL + 1 == i) sgb_vm();
    }
    __builtin_amdgcn_sched_barrier(0);
}

__global__ __launch_bounds__(512) void kern(const char* __restrict__ ga, const char* __restrict__ gb,
                                            const int* __restrict__ rnd, float* out, int steps, int zero, unsigned long long* clk) {
    const unsigned long long t0 = __builtin_amdgcn_s_memtime(), r0 = __builtin_amdgcn_s_memrealtime();
    __shared__ __attribute__((aligned(16))) char lds[LDS_R + LDS_D + LDS_PAD];
    const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6, nw = blockDim.x >> 6;
    // random LDS content (operands) and pad touch
    for (int i = tid; i < (LDS_R + LDS_D + LDS_PAD) / 4; i += blockDim.x)
        reinterpret_cast<int*>(lds)[i] = zero ? 0 : rnd[(i * 7 + blockIdx.x) & 65535];
    __syncthreads();
    int sa = zero ? 0 : rnd[lane], sb = zero ? 0 : rnd[64 + lane];
    sa = 0x7f7f7f7f ^ (sa & 0x03030303); sb = 0x7f7f7f7f ^ (sb & 0x03030303);
    i32x4 fr[2][16];
#pragma unroll
    for (int f = 0; f < 16; ++f) { fr[0][f] = *reinterpret_cast<const i32x4*>(lds + f * 1024 + lane * 16); fr[1][f] = fr[0][f]; }
    f32x4 c[16];
#pragma unroll
    for (int q = 0; q < 16; ++q) c[q] = f32x4{0, 0, 0, 0};
    const int my_steps = steps * 4 / nw;               // same MFMA work per SIMD
    const int spk = 16 / nw;                           // steps per K tile (128 MFMA per SIMD; 64 DMA insts if NBL=4)
    char* lds_d = lds + LDS_R;
    int ktile = 0, tile = blockIdx.x, kstep = 0;
    const char* lds_rw = lds + ((wave * 4096) % LDS_R) + lane * 16;
    const int g_lane_off = (lane / 8) * 4096 + (lane % 8) * 16;
    const int row_stride_step = nw * 8 * 4096;
    for (int s = 0; s < my_steps; s += 2) {
        const int a_row0 = (tile % 32) * 256, b_row0 = ((tile / 32) % 32) * 256;
        auto src_of = [&](int ks) {
            const int i0 = (ks * NBL) * nw + wave;
            const char* base = i0 < 32 ? ga + (size_t)(a_row0 + (i0 & 31) * 8) * 4096 : gb + (size_t)(b_row0 + (i0 & 31) * 8) * 4096;
            return base + ktile * 128;
        };
        const int i0 = (kstep * NBL) * nw + wave, i1 = ((kstep + 1) * NBL) * nw + wave;
        step<0>(fr, c, lds_rw, src_of(kstep), g_lane_off, row_stride_step, lds_d, i0 * 1024, nw * 1024, sa, sb);
        step<1>(fr, c, lds_rw, src_of(kstep + 1), g_lane_off, row_stride_step, lds_d, i1 * 1024, nw * 1024, sa, sb);
        kstep += 2;
        if (kstep >= spk) {
            kstep = 0;
#if BAR
            if (nw == 4) __builtin_amdgcn_s_waitcnt(0x4F70);   // vmcnt(16): previous K tile landed
            else __builtin_amdgcn_s_waitcnt(0x0F78);           // vmcnt(8)
            __builtin_amdgcn_s_barrier();
#endif
            if (++ktile == 32) { ktile = 0; tile += gridDim.x; }
        }
    }
    float r = 0;
#pragma unroll
    for (int q = 0; q < 16; ++q) r += c[q][0] + c[q][1] + c[q][2] + c[q][3];
    if (r == 12345.f) out[0] = r;
    if (threadIdx.x == 0) {
        const unsigned long long t1 = __builtin_amdgcn_s_memtime(), r1 = __builtin_amdgcn_s_memrealtime();
        atomicAdd(clk, t1 - t0); atomicAdd(clk + 1, r1 - r0);
    }
}

int main(int argc, char** argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 256;
    int steps = argc > 2 ? atoi(argv[2]) : 1024;      // steps per wave at 256 threads
    int launches = argc > 3 ? atoi(argv[3]) : 2000;
    int warm = argc > 4 ? atoi(argv[4]) : 200;
    int zero = argc > 5 ? atoi(argv[5]) : 0;
    const int blocks = 256;
    size_t mat = (size_t)8192 * 4096;
    std::vector<int> h(65536);
    srand(1);
    for (auto& x : h) x = (rand() << 16) ^ rand();
    std::vector<char> hm(mat);
    for (size_t i = 0; i < mat; ++i) hm[i] = (char)(rand() & 0xff);
    char *ga, *gb; int* d; float* o; unsigned long long* clk;
    hipMalloc(&clk, 16);
    hipMalloc(&ga, mat); hipMalloc(&gb, mat); hipMalloc(&d, 65536 * 4); hipMalloc(&o, 4);
    hipMemcpy(ga, hm.data(), mat, hipMemcpyHostToDevice);
    hipMemcpy(gb, hm.data(), mat, hipMemcpyHostToDevice);
    hipMemcpy(d, h.data(), 65536 * 4, hipMemcpyHostToDevice);
    for (int i = 0; i < warm; ++i) kern<<<blocks, threads>>>(ga, gb, d, o, steps, zero, clk);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); hipDeviceSynchronize();
    hipMemset(clk, 0, 16);
    hipEventRecord(e0);
    for (int i = 0; i < launches; ++i) kern<<<blocks, threads>>>(ga, gb, d, o, steps, zero, clk);
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms; hipEventElapsedTime(&ms, e0, e1);
    double flops = 2.0 * 16 * 16 * 128 * 32.0 * steps * 4 /*waves@256*/ * blocks * (double)launches;
    unsigned long long hc[2]; hipMemcpy(hc, clk, 16, hipMemcpyDeviceToHost);
    const double mhz = 100.0 * (double)hc[0] / (double)hc[1];
    const double peak = 256.0 * 4 * 65536 / 16 * mhz * 1e6;   // fp4 16x16x128 = 16 cycles/SIMD
    printf("clk %.0f MHz  MFMA-pipe util %.1f%%  ", mhz, 100.0 * flops / (ms * 1e-3) / peak);
    printf("threads=%d NDS=%d NBL=%d DEP=%d BAR=%d steps=%d per-launch %.1f us  %.3f PFLOPS\n", threads, NDS, NBL,
           DEPPAIR, BAR, steps, ms * 1e3 / launches, flops / (ms * 1e-3) / 1e15);
}
