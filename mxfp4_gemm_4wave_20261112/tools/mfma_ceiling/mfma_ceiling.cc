// Power-capped ceiling: only fp4 scaled MFMA 16x16x128 with random operands that change every MFMA.
// grid = 256 * BLK blocks of 256 threads (4 waves, one per SIMD), like the GEMM.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
#ifndef NSET
#define NSET 8
#endif
__global__ __launch_bounds__(256) void k(const int* __restrict__ src, float* out, int iters, int zero) {
    const int lane = threadIdx.x;
    i32x8 a[NSET], b[NSET];
    #pragma unroll
    for (int s = 0; s < NSET; ++s) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            a[s][j] = zero ? 0 : src[(s * 8 + j) * 256 + lane];
            b[s][j] = zero ? 0 : src[(NSET * 8 + s * 8 + j) * 256 + lane];
        }
    }
    int sa = zero ? 0 : src[16 * NSET * 256 + lane], sb = zero ? 0 : src[16 * NSET * 256 + 256 + lane];
    sa = 0x7f7f7f7f ^ (sa & 0x03030303); sb = 0x7f7f7f7f ^ (sb & 0x03030303);
    f32x4 c[8] = {};
    for (int it = 0; it < iters; ++it) {
        #pragma unroll
        for (int s = 0; s < NSET; ++s) {
            #pragma unroll
            for (int q = 0; q < 8; ++q)   // 8 independent accumulators
                c[q] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a[s], b[(s + q) % NSET], c[q], 4, 4, 0, sa, 0, sb);
        }
    }
    float r = 0; for (int q = 0; q < 8; ++q) r += c[q][0] + c[q][1] + c[q][2] + c[q][3];
    if (r == 12345.f) out[0] = r;
}
int main(int argc, char** argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 256, launches = argc > 2 ? atoi(argv[2]) : 20000, zero = argc > 3 ? atoi(argv[3]) : 0;
    int blocks = 256 * 4;
    size_t n = (16 * NSET + 2) * 256; int* h = (int*)malloc(n * 4); srand(1); for (size_t i = 0; i < n; ++i) h[i] = (rand() << 16) ^ rand();
    int* d; float* o; hipMalloc(&d, n * 4); hipMalloc(&o, 4); hipMemcpy(d, h, n * 4, hipMemcpyHostToDevice);
    for (int i = 0; i < 200; ++i) k<<<blocks, 256>>>(d, o, iters, zero);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1); hipDeviceSynchronize();
    hipEventRecord(e0); for (int i = 0; i < launches; ++i) k<<<blocks, 256>>>(d, o, iters, zero); hipEventRecord(e1); hipEventSynchronize(e1);
    float ms; hipEventElapsedTime(&ms, e0, e1);
    double flops = 2.0 * 16 * 16 * 128 * 8.0 * NSET * iters * (blocks * 4.0) * launches;
    printf("zero=%d per-launch %.1f us, %.2f PFLOPS\n", zero, ms * 1e3 / launches, flops / (ms * 1e-3) / 1e15);
}
