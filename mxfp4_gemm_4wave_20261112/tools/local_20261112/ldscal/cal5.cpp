// Kernel A/B read: lane (r = lane%16, g = lane/16): addr = (r%2)*1056 + (r/2)*128 + ((g ^ h(r)) * 16) + kk*64
// h(r) = 2-bit table entry H >> (2*r) & 3   (r = 0..15)
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
typedef int i32x4 __attribute__((ext_vector_type(4)));
__global__ __launch_bounds__(256) void k(int* out, int iters, unsigned H, int kk, int PAR) {
    __shared__ __attribute__((aligned(16))) char lds[65536];
    const int t = threadIdx.x, lane = t & 63;
    for (int i = t * 16; i < 65536; i += 256 * 16) *(i32x4*)(lds + i) = i32x4{i, i + 1, i + 2, i + 3};
    __syncthreads();
    const int r = lane & 15, g = lane >> 4, h = (H >> (2 * r)) & 3;
    const int off = (r & 1) * PAR + (r >> 1) * 128 + ((g ^ h) * 16) + kk * 64;
    i32x4 acc = {0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
        i32x4 v;
        asm volatile("ds_read_b128 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"((unsigned)(size_t)(lds) + off) : "memory");
        acc += v;
    }
    if (acc[0] == 12345) out[0] = acc[1] + lds[t];
}
int main(int argc, char** argv) {
    int* o; (void)hipMalloc(&o, 4);
    int PAR = atoi(argv[1]); for (int i = 2; i < argc; ++i) { unsigned H = strtoul(argv[i], 0, 0);
        hipLaunchKernelGGL(k, dim3(256), dim3(256), 0, 0, o, 1000, H, 0, PAR); (void)hipDeviceSynchronize(); }
    printf("ok\n");
}
