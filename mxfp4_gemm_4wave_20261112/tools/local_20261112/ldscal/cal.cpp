// LDS bank-conflict counter calibration on gfx950: each mode issues ITERS ds_read_b128 per wave.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
typedef int i32x4 __attribute__((ext_vector_type(4)));
template<int MODE>
__global__ __launch_bounds__(256) void k(int* out, int iters) {
    __shared__ __attribute__((aligned(16))) char lds[65536];
    const int t = threadIdx.x, lane = t & 63;
    for (int i = t * 16; i < 65536; i += 256 * 16) *(i32x4*)(lds + i) = i32x4{i, i + 1, i + 2, i + 3};
    __syncthreads();
    int off;
    if (MODE == 0) off = lane * 16;                       // contiguous 1 KB: conflict-free
    else if (MODE == 1) off = lane * 32;                   // stride 32 B
    else if (MODE == 2) off = lane * 256;                  // stride 256 B: every lane same bank group
    else off = (lane % 16) * 16 + (lane / 16) * 4224;      // 4 row groups, row stride 4224 B (kernel's A layout offsets)
    i32x4 acc = {0, 0, 0, 0};
    for (int it = 0; it < iters; ++it) {
        i32x4 v;
        asm volatile("ds_read_b128 %0, %1\n s_waitcnt lgkmcnt(0)" : "=v"(v) : "v"(off + ((it & 1) << 12)));
        acc += v;
    }
    if (acc[0] == 12345) out[0] = acc[1];
}
int main(int argc, char** argv) {
    int mode = atoi(argv[1]), iters = 1000; int* o; hipMalloc(&o, 4);
    void (*f[4])(int*, int) = {k<0>, k<1>, k<2>, k<3>};
    hipLaunchKernelGGL(f[mode], dim3(256), dim3(256), 0, 0, o, iters);   // 256 blocks x 4 waves
    hipDeviceSynchronize(); printf("mode %d done\n", mode);
}
