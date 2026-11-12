// launch_6696 <co> <symbol> M N K iters : PR6696 a8w8 blockscale mxscale 4-wave pin_agpr code object,
// random fp8 / e8m0 data (over-allocated), grid (N/256, M/256), no verification (trace only).
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP %s @%d\n", hipGetErrorString(e), __LINE__); exit(1);} } while (0)
struct kargs_t {   // == opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 (96 B)
    const void* ptr_a; const void* ptr_b; void* ptr_c; int m, n, k, batch;
    int stride_a, stride_b, stride_c, stride_a_batch, stride_b_batch, stride_c_batch;
    const void* ptr_sfa; const void* ptr_sfb; int stride_sfa, stride_sfb, stride_sfa_batch, stride_sfb_batch;
};
static_assert(sizeof(kargs_t) == 96, "abi");
int main(int argc, char** argv) {
    const char* co = argv[1]; const char* sym = argv[2]; int M = atoi(argv[3]), N = atoi(argv[4]), K = atoi(argv[5]), it = atoi(argv[6]);
    size_t a = size_t(M) * K, b = size_t(N) * K, s = size_t(M + N) * K / 16 + 4096;
    std::vector<unsigned char> h(std::max(a, b)); for (auto& v : h) v = rand() & 0x77;   // finite fp8 e4m3
    std::vector<unsigned char> hs(s); for (auto& v : hs) v = 124 + rand() % 7;
    void *dA, *dB, *dC, *dSA, *dSB;
    CK(hipMalloc(&dA, a)); CK(hipMalloc(&dB, b)); CK(hipMalloc(&dC, size_t(M) * N * 2)); CK(hipMalloc(&dSA, s)); CK(hipMalloc(&dSB, s));
    CK(hipMemcpy(dA, h.data(), a, hipMemcpyHostToDevice)); CK(hipMemcpy(dB, h.data(), b, hipMemcpyHostToDevice));
    CK(hipMemcpy(dSA, hs.data(), s, hipMemcpyHostToDevice)); CK(hipMemcpy(dSB, hs.data(), s, hipMemcpyHostToDevice));
    kargs_t k{}; k.ptr_a = dA; k.ptr_b = dB; k.ptr_c = dC; k.m = M; k.n = N; k.k = K; k.batch = 1;
    k.stride_a = K; k.stride_b = K; k.stride_c = N; k.stride_a_batch = M * K; k.stride_b_batch = N * K; k.stride_c_batch = M * N;
    k.ptr_sfa = dSA; k.ptr_sfb = dSB; k.stride_sfa = K / 32; k.stride_sfb = K / 32; k.stride_sfa_batch = M * (K / 32); k.stride_sfb_batch = N * (K / 32);
    hipModule_t mod; hipFunction_t fn; CK(hipModuleLoad(&mod, co)); CK(hipModuleGetFunction(&fn, mod, sym));
    size_t sz = sizeof(k); void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &k, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    auto launch = [&]() { CK(hipModuleLaunchKernel(fn, N / 256, M / 256, 1, 256, 1, 1, 0, 0, nullptr, cfg)); };
    for (int i = 0; i < it; ++i) launch();
    CK(hipDeviceSynchronize());
    if (getenv("TIME")) {
        hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
        for (int i = 0; i < 1000; ++i) launch();
        for (int r = 0; r < 3; ++r) { CK(hipEventRecord(e0)); for (int i = 0; i < 200; ++i) launch(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1));
            float ms; CK(hipEventElapsedTime(&ms, e0, e1)); printf("time %.1f us, %.0f TFLOPS (fp8)\n", ms * 1e3 / 200, 2.0 * M * N * double(K) / (ms * 1e-3 / 200) / 1e12); }
    }
    printf("done\n");
}
