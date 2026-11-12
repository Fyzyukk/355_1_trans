// off(lane) = (lane % P) * 16 + (lane / P) * S  ; ds_read_b128, 1000 per wave, 1024 waves
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
typedef int i32x4 __attribute__((ext_vector_type(4)));
__global__ __launch_bounds__(256) void k(int* out, int iters, int P, int S, int X) {
    __shared__ __attribute__((aligned(16))) char lds[65536];
    const int t = threadIdx.x, lane = t & 63;
    for (int i = t * 16; i < 65536; i += 256 * 16) *(i32x4*)(lds + i) = i32x4{i, i + 1, i + 2, i + 3};
    __syncthreads();
    int off = (lane % P) * 16 + (lane / P) * S;
    if (X == 1) off = (lane & 15) * 128 + (lane >> 4) * 16;                               // kernel-like: 16 rows x 128 B, g chunks (no r%2 split)
    if (X == 2) { int r = lane & 15, g = lane >> 4; off = ((r >> 1) + 8 * (r & 1)) * 128 + g * 16; }   // kernel A/B read pattern (kk = 0)
    if (X == 3) { int r = lane & 15, g = lane >> 4; int row = (r >> 1) + 8 * (r & 1); off = row * 128 + ((g ^ ((row >> 1) & 3)) * 16) + ((row & 1) ? 64 : 0); } // swizzle try A
    if (X == 4) { int r = lane & 15, g = lane >> 4; int row = (r >> 1) + 8 * (r & 1); int c = g ^ (((row >> 1) & 1) << 2 | 0); off = row * 128 + (((g + 4 * ((row >> 1) & 1)) ^ 0) & 7) * 16; } // swizzle try B
    if (X == 5) { int r = lane & 15, g = lane >> 4; int row = (r >> 1) + 8 * (r & 1); off = row * 128 + ((g ^ (row & 7)) & 7) * 16; } // swizzle try C: chunk ^= row%8
    i32x4 acc = {0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
        i32x4 v;
        asm volatile("ds_read_b128 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"((unsigned)(size_t)(lds) + off) : "memory");
        acc += v;
    }
    if (acc[0] == 12345) out[0] = acc[1] + lds[t];
}
int main(int argc, char** argv) {
    int P = atoi(argv[1]), S = atoi(argv[2]), X = argc > 3 ? atoi(argv[3]) : 0; int* o; (void)hipMalloc(&o, 4);
    hipLaunchKernelGGL(k, dim3(256), dim3(256), 0, 0, o, 1000, P, S, X); (void)hipDeviceSynchronize(); printf("ok\n");
}
