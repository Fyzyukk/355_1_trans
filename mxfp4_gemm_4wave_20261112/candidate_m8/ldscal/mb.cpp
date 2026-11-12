#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
typedef int i32x4 __attribute__((ext_vector_type(4)));
typedef int i32x2 __attribute__((ext_vector_type(2)));
#define ITERS 1000
// mode 0: ds_write2_b64 offset1:8 ; 1: ds_read_b128 ; 2: ds_write_b128 ; 3: ds_read2st64_b64 offset1:4 ; 4: ds_read_b64; 5: ds_write_b64
__global__ __launch_bounds__(256) void k(int* out, const unsigned* addr, int mode) {
    __shared__ __attribute__((aligned(16))) char lds[65536];
    const int t = threadIdx.x;
    for (int i = t * 16; i < 65536; i += 256 * 16) *(i32x4*)(lds + i) = i32x4{i, i + 1, i + 2, i + 3};
    __syncthreads();
    unsigned a = (unsigned)(size_t)lds + addr[t];
    i32x4 acc = {0,0,0,0};
    i32x2 w = {t, t+1}; i32x4 w4 = {t,t,t,t};
    if (mode == 0) { for (int it = 0; it < ITERS; ++it) asm volatile("ds_write2_b64 %0, %1, %2 offset1:8\n s_waitcnt lgkmcnt(0)" :: "v"(a), "v"(w), "v"(w) : "memory"); }
    else if (mode == 1) { for (int it = 0; it < ITERS; ++it) { i32x4 v; asm volatile("ds_read_b128 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(a) : "memory"); acc += v; } }
    else if (mode == 2) { for (int it = 0; it < ITERS; ++it) asm volatile("ds_write_b128 %0, %1\n s_waitcnt lgkmcnt(0)" :: "v"(a), "v"(w4) : "memory"); }
    else if (mode == 3) { for (int it = 0; it < ITERS; ++it) { i32x4 v; asm volatile("ds_read2st64_b64 %0, %1 offset1:4\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(a) : "memory"); acc += v; } }
    else if (mode == 4) { for (int it = 0; it < ITERS; ++it) { i32x2 v; asm volatile("ds_read_b64 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(a) : "memory"); acc.x += v.x; } }
    else if (mode == 5) { for (int it = 0; it < ITERS; ++it) asm volatile("ds_write_b64 %0, %1\n s_waitcnt lgkmcnt(0)" :: "v"(a), "v"(w) : "memory"); }
    else if (mode == 6) { for (int it = 0; it < ITERS; ++it) { i32x4 v; asm volatile("ds_read2_b64 %0, %1 offset0:1 offset1:0\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(a) : "memory"); acc += v; } }
    else if (mode == 7) { for (int it = 0; it < ITERS; ++it) asm volatile("ds_write2st64_b64 %0, %1, %2 offset1:4\n s_waitcnt lgkmcnt(0)" :: "v"(a), "v"(w), "v"(w) : "memory"); }
    else if (mode == 8) { for (int it = 0; it < ITERS; ++it) { i32x4 v; asm volatile("ds_read2st64_b64 %0, %1 offset1:2\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(a) : "memory"); acc += v; } }
    else if (mode == 9) { for (int it = 0; it < ITERS; ++it) { i32x4 v; asm volatile("ds_read2_b64 %0, %1 offset1:1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(a) : "memory"); acc += v; } }
    if (acc[0] == 12345) out[0] = acc[1] + lds[t];
}
// file format: lines "mode a0 a1 ... a255"
int main(int argc, char** argv) {
    int* o; (void)hipMalloc(&o, 4); unsigned* d; (void)hipMalloc(&d, 256*4);
    FILE* f = fopen(argv[1], "r"); int mode; int n = 0;
    while (fscanf(f, "%d", &mode) == 1) { std::vector<unsigned> h(256); for (int i = 0; i < 256; i++) fscanf(f, "%u", &h[i]);
        (void)hipMemcpy(d, h.data(), 1024, hipMemcpyHostToDevice);
        hipLaunchKernelGGL(k, dim3(256), dim3(256), 0, 0, o, d, mode); (void)hipDeviceSynchronize(); n++; }
    printf("ran %d\n", n);
}
