// MFMA-only: accumulator chain length L (L dependent MFMAs on the same C back to back), 1 or 2 waves/SIMD.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
#ifndef L
#define L 2
#endif
__global__ __launch_bounds__(512) void k(const int* __restrict__ src, float* out, int iters, unsigned long long* clk) {
    __shared__ int pad[24 * 1024 + 1024];   // > 80 KB: 1 block per CU
    const unsigned long long t0 = __builtin_amdgcn_s_memtime(), r0 = __builtin_amdgcn_s_memrealtime();
    const int lane = threadIdx.x & 63, nw = blockDim.x / 64;
    pad[threadIdx.x] = 0;
    i32x8 a[8], b[8];
#pragma unroll
    for (int s = 0; s < 8; ++s)
#pragma unroll
        for (int j = 0; j < 4; ++j) { a[s][j] = src[(s * 8 + j) * 256 + lane]; b[s][j] = src[(64 + s * 8 + j) * 256 + lane]; }
    int sa = 0x7f7f7f7f ^ (src[lane] & 0x03030303), sb = 0x7f7f7f7f ^ (src[64 + lane] & 0x03030303);
    f32x4 c[16] = {};
    for (int i = threadIdx.x; i < 24 * 1024; i += blockDim.x) pad[i] = src[i & 32767];
    __syncthreads();
    const int* lp = pad + (threadIdx.x / 64) * 1024 + lane * 4;
    const int my = iters * 4 / nw;
    for (int it = 0; it < my; ++it) {
        i32x8 an[8], bn[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int4 x = *reinterpret_cast<const int4*>(lp + s * 256), y = *reinterpret_cast<const int4*>(lp + 2048 + s * 256);
            an[s] = i32x8{x.x, x.y, x.z, x.w, 0, 0, 0, 0}; bn[s] = i32x8{y.x, y.y, y.z, y.w, 0, 0, 0, 0};
        }
#pragma unroll
        for (int i = 0; i < 64; ++i) {          // 64 MFMAs per iteration
            const int q = (i / L) % 16;
            c[q] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a[i % 8], b[(i / 8 + i) % 8], c[q], 4, 4, 0, sa, 0, sb);
            if (i % 4 == 0) __builtin_amdgcn_sched_group_barrier(0x008, 4, 0), __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
        }
#pragma unroll
        for (int s = 0; s < 8; ++s) { a[s] = an[s]; b[s] = bn[s]; }
    }
    float r = pad[(threadIdx.x + 1) % 512]; for (int q = 0; q < 16; ++q) r += c[q][0] + c[q][1] + c[q][2] + c[q][3];
    if (r == 12345.f) out[0] = r;
    if (threadIdx.x == 0) { const unsigned long long t1 = __builtin_amdgcn_s_memtime(), r1 = __builtin_amdgcn_s_memrealtime(); atomicAdd(clk, t1 - t0); atomicAdd(clk + 1, r1 - r0); }
}
int main(int argc, char** argv) {
    int threads = atoi(argv[1]), iters = 1024, launches = argc > 2 ? atoi(argv[2]) : 8000;
    int n = 130 * 256; int* h = (int*)malloc(n * 4); srand(1); for (int i = 0; i < n; ++i) h[i] = (rand() << 16) ^ rand();
    int* d; float* o; unsigned long long* clk; hipMalloc(&d, n * 4); hipMalloc(&o, 4); hipMalloc(&clk, 16); hipMemcpy(d, h, n * 4, hipMemcpyHostToDevice);
    for (int i = 0; i < 300; ++i) k<<<256, threads>>>(d, o, iters, clk);
    hipDeviceSynchronize(); hipMemset(clk, 0, 16);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipEventRecord(e0); for (int i = 0; i < launches; ++i) k<<<256, threads>>>(d, o, iters, clk); hipEventRecord(e1); hipEventSynchronize(e1);
    float ms; hipEventElapsedTime(&ms, e0, e1);
    unsigned long long hc[2]; hipMemcpy(hc, clk, 16, hipMemcpyDeviceToHost); double mhz = 100.0 * hc[0] / hc[1];
    double flops = 2.0 * 16 * 16 * 128 * 64.0 * iters * 4 * 256 * (double)launches;
    double pf = flops / (ms * 1e-3) / 1e15;
    printf("L=%d threads=%d clk %.0f MHz util %.1f%% %.3f PFLOPS\n", L, threads, mhz, 100.0 * pf * 1e15 / (256.0 * 4 * 4096 * mhz * 1e6), pf);
}
