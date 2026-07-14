// Variant of the ldsbypass host driver that builds the compact per-lane B'' fragment
// buffer entirely on the CPU (no GPU repack kernel, no LDS anywhere in B'' generation).
//
// The main gemm kernel is byte-for-byte the same as the GPU-repack variant: it reads B''
// through bpp_frag_offset() and never touches LDS for B. The only difference here is *who*
// produces B''. Instead of launching gemm_a8w8_blockscale_ldsbypass_repack_kernel (which
// stages B' through LDS with u_gb/u_sb and reads it back with u_rb), we replay those exact
// opus layouts on the host: fill a CPU array that stands in for LDS via u_gb->u_sb, then
// read it out via u_rb, writing the fragment bytes contiguously into B''. Because host and
// device share the same opus layout evaluation, B'' is identical to the GPU-repack output
// by construction -- no hand-derived permutation.
#include <hip/hip_fp8.h>
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include <omp.h>

#include "gemm_a8w8_blockscale_ldsbypass_common.h"

using opus::operator""_I;

template<class Traits>
__global__ void gemm_a8w8_blockscale_ldsbypass_kernel(opus_gemm_kargs kargs);

#define CHECK_HIP(call)                                                                                   \
    do {                                                                                                  \
        hipError_t status_ = call;                                                                        \
        if (status_ != hipSuccess) {                                                                      \
            fprintf(stderr, "HIP error (%s:%d): %s\n", __FILE__, __LINE__, hipGetErrorString(status_));   \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define CHECK_HIP_KERNEL_LAUNCH() CHECK_HIP(hipGetLastError())

using GemmTraits = gemm_a8w8_blockscale_traits<>;
using host_fp8_t = __hip_fp8_e4m3;
using fp32_t = float;

// ---------------------------------------------------------------------------------------
// Host-callable copies of opus's unfold_x_stride / unfold_p_coord. These are the ONLY opus
// layout helpers marked __device__-only (opus.hpp:2703 / 2711); their bodies are pure
// constexpr type arithmetic, so we re-tag them __host__ __device__ verbatim. Everything else
// (make_layout, layout_to_offsets, coord_to_linear, ...) is already __host__ __device__.
// Kept in sync with opus.hpp:2660-2716.
namespace hostlayout {
using opus::number; using opus::seq; using opus::make_index_seq; using opus::underscore;
using opus::p_dim; using opus::y_dim; using opus::remove_cvref_t; using opus::index_t;

template<typename Dim, index_t... Js>
inline __host__ __device__ constexpr index_t dim_offset_sum(seq<Js...>) {
    return (static_cast<index_t>(opus::size<decltype(opus::get<Js>(Dim{}))>()) + ... + 0); }
template<typename Dim, index_t... Js>
inline __host__ __device__ constexpr index_t p_count_in(seq<Js...>) {
    return ((std::is_same_v<remove_cvref_t<decltype(opus::get<Js>(Dim{}))>, p_dim> ? 1 : 0) + ... + 0); }
template<typename Dim, typename Coord, index_t... Is>
inline __host__ __device__ constexpr auto unfold_p_coord_impl(const Coord& coord, seq<Is...>) {
    return opus::make_tuple( [&]() -> decltype(auto) {
        if constexpr (std::is_same_v<remove_cvref_t<decltype(opus::get<Is>(Dim{}))>, p_dim>)
            return opus::get< p_count_in<Dim>(make_index_seq<Is>{}) >(coord);
        else return underscore{};
    }()... ); }
template<typename Dim, index_t J, index_t... Gs>
inline __host__ __device__ constexpr index_t unfold_find_group(seq<Gs...>) {
    index_t acc = 0, r = 0;
    ((void)(acc += opus::size<decltype(opus::get<Gs>(Dim{}))>(), (acc <= J ? (void)(r = Gs + 1) : (void)0)), ...);
    return r; }
template<typename Dim, typename Shape, typename Stride, index_t J>
inline __host__ __device__ constexpr auto unfold_x_stride_at(const Stride& stride) {
    constexpr index_t G = unfold_find_group<Dim, J>(make_index_seq<opus::size<Dim>()>{});
    constexpr index_t group_end = dim_offset_sum<Dim>(make_index_seq<G + 1>{});
    return opus::impl::packed_stride_at<Shape, J>(make_index_seq<group_end - J - 1>{}) * opus::get<G>(stride); }
template<typename Dim, typename Shape, typename Stride, index_t... Js>
inline __host__ __device__ constexpr auto unfold_x_stride_flat(const Stride& stride, seq<Js...>) {
    return opus::make_tuple(unfold_x_stride_at<Dim, Shape, Stride, Js>(stride)...); }

template<typename Dim, typename Coord>
inline __host__ __device__ constexpr auto unfold_p_coord(const Dim&, const Coord& coord) {
    using FDim = remove_cvref_t<decltype(opus::flatten_tuple(Dim{}))>;
    return unfold_p_coord_impl<FDim, Coord>(coord, make_index_seq<opus::size<FDim>()>{}); }
template<typename Dim, typename Shape, typename Stride>
inline __host__ __device__ constexpr auto unfold_x_stride(const Dim&, const Shape&, const Stride& stride) {
    return unfold_x_stride_flat<Dim, Shape, Stride>(stride, make_index_seq<opus::size<Shape>()>{}); }
}

// Host copies of the three B layouts from the kernel template. Identical shape/dim/coord to
// make_layout_gb / make_layout_sb / make_layout_rb (template.hpp:114-182); the only change is
// hostlayout:: unfold and dropping the make_layout<VEC_B> cached variant in favour of the
// default (layout_linear) -- offsets are obtained via layout_to_offsets<VEC_B> either way.
template<class Tr>
static auto h_make_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    constexpr int threads_k = Tr::B_K / Tr::VEC_B;
    constexpr int threads_n_per_block = Tr::BLOCK_SIZE / threads_k;
    constexpr int threads_n_per_wave = Tr::WARP_SIZE / threads_k;
    constexpr auto shape = opus::make_tuple(
        opus::number<Tr::HALF_B_N / threads_n_per_block>{}, opus::number<Tr::T_N>{},
        opus::number<threads_n_per_wave>{}, opus::number<Tr::T_M>{},
        opus::number<threads_k>{}, opus::number<Tr::VEC_B>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout(shape,
        hostlayout::unfold_x_stride(dim, shape, opus::tuple{stride_b, 1_I}),
        hostlayout::unfold_p_coord(dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m, lane_id % threads_k}));
}
template<class Tr>
static auto h_make_sb(int wave_id_m, int wave_id_n) {
    constexpr int num_waves = Tr::BLOCK_SIZE / Tr::WARP_SIZE;
    constexpr auto shape = opus::make_tuple(
        opus::number<Tr::smem_n_rep / num_waves>{}, opus::number<Tr::T_N>{},
        opus::number<Tr::T_M>{}, opus::number<Tr::VEC_B>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(shape,
        hostlayout::unfold_x_stride(dim, shape, opus::tuple{opus::number<Tr::smem_linear_wave + Tr::smem_padding>{}, 1_I}),
        hostlayout::unfold_p_coord(dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class Tr>
static auto h_make_rb(int lane_id, int wave_id_n) {
    constexpr auto shape = opus::make_tuple(
        opus::number<Tr::E_N>{}, opus::number<Tr::T_M>{}, opus::number<Tr::T_N>{},
        opus::number<Tr::W_N / Tr::T_M>{}, opus::number<Tr::E_K>{},
        opus::number<Tr::W_N * Tr::W_K / Tr::WARP_SIZE / Tr::VEC_B>{},
        opus::number<Tr::WARP_SIZE / Tr::W_N>{}, opus::number<Tr::VEC_B>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int lane_id_n = lane_id % Tr::W_N;
    return opus::make_layout(shape,
        hostlayout::unfold_x_stride(dim, shape, opus::tuple{opus::number<Tr::smem_linear_wave + Tr::smem_padding>{}, 1_I}),
        hostlayout::unfold_p_coord(dim, opus::tuple{lane_id_n % Tr::T_M, wave_id_n, lane_id_n / Tr::T_M, lane_id / Tr::W_N}));
}

// Compact B'' fragment offset -- host copy of bpp_frag_offset (template.hpp:207), pure integer
// arithmetic so it is trivially host-callable. Both the main kernel (load) and this packer
// (store) address B'' through this one formula.
template<class Tr>
static int h_bpp_frag_offset(int col_tile, int num_kt, int k_tile, int half_n, int wave_id_n, int lane_id) {
    const int frag_idx = ((((col_tile * num_kt) + k_tile) * Tr::HALVES_N + half_n) * Tr::T_N + wave_id_n) * Tr::WARP_SIZE + lane_id;
    return frag_idx * Tr::FRAG_B_ELEMS;
}

// CPU repack: replay the GPU repack kernel on the host. For every (batch, col_tile, k_tile)
// tile and every half of the N dimension, stage the preshuffled B' slab into a CPU "LDS"
// array via u_gb->u_sb, then read it back via u_rb and write the per-lane fragment into B''.
// Mirrors gemm_a8w8_blockscale_ldsbypass_repack_kernel (template.hpp:219-270) step for step.
template<class Tr>
static void host_pack_b(const host_fp8_t* b_prime, host_fp8_t* b_pp,
                        int batch, int N, int K, int stride_b /*=BLOCK_K*/) {
    constexpr int num_waves = Tr::BLOCK_SIZE / Tr::WARP_SIZE;
    constexpr int smem_b_elem = Tr::smem_n_rep * (Tr::smem_linear_wave + Tr::smem_padding);
    const int num_tiles_n = ceil_div(N, Tr::B_N);
    const int num_kt = ceil_div(K, Tr::B_K);

    // Per-issue offsets are the same for every wave/lane up to the p-coord, but opus bakes the
    // p-coord into the layout, so we rebuild layouts per (wave_id, lane). Cost is trivial
    // (offline, tiny loops); correctness comes first.
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch; ++b) {
        for (int col_tile = 0; col_tile < num_tiles_n; ++col_tile) {
            for (int k_tile = 0; k_tile < num_kt; ++k_tile) {
                const int col = col_tile * Tr::B_N;
                const host_fp8_t* g_b = b_prime + static_cast<std::size_t>(b) * N * K + static_cast<std::size_t>(col) * stride_b;
                host_fp8_t* g_bpp = b_pp + static_cast<std::size_t>(b) * N * K;

                std::vector<host_fp8_t> lds(smem_b_elem);

                for (int half = 0; half < Tr::HALVES_N; ++half) {
                    // gb_offset: same as template.hpp:250 -- half picks the N half, k_tile jumps a K-panel.
                    const int gb_base = half * Tr::HALF_B_N * stride_b + k_tile * N * Tr::B_K;

                    // Stage B' -> LDS via u_gb (gmem read offsets) and u_sb (smem write offsets).
                    for (int wave_id = 0; wave_id < num_waves; ++wave_id) {
                        const int wave_id_m = wave_id % Tr::T_M;
                        const int wave_id_n = wave_id / Tr::T_M;
                        for (int lane_id = 0; lane_id < Tr::WARP_SIZE; ++lane_id) {
                            auto u_gb = h_make_gb<Tr>(lane_id, wave_id_m, wave_id_n, stride_b);
                            auto u_sb = h_make_sb<Tr>(wave_id_m, wave_id_n);
                            auto gb_off = opus::layout_to_offsets<Tr::VEC_B>(u_gb);
                            auto sb_off = opus::layout_to_offsets<Tr::VEC_B>(u_sb);
                            // CDNA buffer_load_lds scatters the 64 lanes of a wave into
                            // consecutive LDS slots: u_sb gives the per-issue wave base (no lane
                            // term), the hardware adds lane*VEC_B. Replicate that lane stride here.
                            for (int i = 0; i < (int)gb_off.size(); ++i) {
                                for (int e = 0; e < Tr::VEC_B; ++e) {
                                    lds[sb_off[i] + lane_id * Tr::VEC_B + e] = g_b[gb_base + gb_off[i] + e];
                                }
                            }
                        }
                    }

                    // Read LDS -> per-lane fragment via u_rb, write contiguously into B''.
                    // Fragment is per-(wave_n, lane), shared across T_M M-waves -> emit once (wave_id_m==0).
                    for (int wave_id_n = 0; wave_id_n < Tr::T_N; ++wave_id_n) {
                        for (int lane_id = 0; lane_id < Tr::WARP_SIZE; ++lane_id) {
                            auto u_rb = h_make_rb<Tr>(lane_id, wave_id_n);
                            auto rb_off = opus::layout_to_offsets<Tr::VEC_B>(u_rb);
                            const int bpp_off = h_bpp_frag_offset<Tr>(col_tile, num_kt, k_tile, half, wave_id_n, lane_id);
                            for (int i = 0; i < (int)rb_off.size(); ++i) {
                                for (int e = 0; e < Tr::VEC_B; ++e) {
                                    g_bpp[bpp_off + i * Tr::VEC_B + e] = lds[rb_off[i] + e];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template<typename T>
void rand_vector(T* ptr, std::size_t size, fp32_t min_val = 0.0f, fp32_t max_val = 1.0f) {
    #pragma omp parallel
    {
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());
        std::uniform_real_distribution<fp32_t> dis(min_val, max_val);
        #pragma omp for
        for (std::size_t i = 0; i < size; ++i) {
            ptr[i] = static_cast<T>(dis(gen));
        }
    }
}

template<typename T>
bool valid_vector(const T* ref, const T* result, int n, fp32_t threshold = 1e-3f) {
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        const fp32_t diff = std::abs(static_cast<fp32_t>(ref[i]) - static_cast<fp32_t>(result[i]));
        if (diff > threshold) {
            if (errors < 10) {
                std::printf("Error at %d: ref=%.6f, result=%.6f, diff=%.6f\n",
                            i, static_cast<fp32_t>(ref[i]), static_cast<fp32_t>(result[i]), diff);
            }
            ++errors;
            if (errors >= 10) {
                break;
            }
        }
    }
    return errors == 0;
}

void gemm_ref(const host_fp8_t* a, const host_fp8_t* b, const fp32_t* sfa, const fp32_t* sfb, fp32_t* c,
              int m, int n, int k, int lda, int ldb, int ldc, int stride_sfa, int stride_sfb,
              int group_m, int group_n, int group_k) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const host_fp8_t* a_row = a + i * lda;
            const host_fp8_t* b_row = b + j * ldb;
            const int m_group = i / group_m;
            const int n_group = j / group_n;
            fp32_t sum = 0.0f;
            for (int k_group_idx = 0; k_group_idx < k / group_k; ++k_group_idx) {
                const fp32_t scale_a = sfa[k_group_idx * stride_sfa + m_group];
                const fp32_t scale_b = sfb[n_group * stride_sfb + k_group_idx];
                const fp32_t scale = scale_a * scale_b;
                const int p_begin = k_group_idx * group_k;
                const int p_end = p_begin + group_k;
                for (int p = p_begin; p < p_end; ++p) {
                    sum += static_cast<fp32_t>(a_row[p]) * static_cast<fp32_t>(b_row[p]) * scale;
                }
            }
            c[i * ldc + j] = sum;
        }
    }
}

// B preshuffle into K-panels (same as the GPU-repack variant / bpreshuffle).
void shuffle_b(const host_fp8_t* src, host_fp8_t* dst, int batch, int N, int K, int block_k) {
    const int num_kt = K / block_k;
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; ++b) {
        for (int n = 0; n < N; ++n) {
            const host_fp8_t* s = src + static_cast<std::size_t>(b) * N * K + static_cast<std::size_t>(n) * K;
            for (int kt = 0; kt < num_kt; ++kt) {
                host_fp8_t* d = dst + static_cast<std::size_t>(b) * N * K
                              + (static_cast<std::size_t>(kt) * N + n) * block_k;
                for (int ki = 0; ki < block_k; ++ki) {
                    d[ki] = s[static_cast<std::size_t>(kt) * block_k + ki];
                }
            }
        }
    }
}

template<class Traits>
void benchmark_kernel(const opus_gemm_kargs& kargs, dim3 grid, dim3 block, int warmup = 200, int iterations = 100) {
    for (int i = 0; i < warmup; ++i) {
        gemm_a8w8_blockscale_ldsbypass_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP_KERNEL_LAUNCH();
    }

    hipEvent_t start;
    hipEvent_t stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        gemm_a8w8_blockscale_ldsbypass_kernel<Traits><<<grid, block>>>(kargs);
        CHECK_HIP_KERNEL_LAUNCH();
    }

    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    fp32_t total_time = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&total_time, start, stop));

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));

    const fp32_t avg_time = total_time / iterations;
    const std::size_t flop = static_cast<std::size_t>(2) * kargs.m * kargs.n * kargs.k * kargs.batch;
    const fp32_t tflops = static_cast<fp32_t>(flop) / 1.0e9f / avg_time;

    std::printf("Kernel Performance: avg_time=%.4f ms, %.2f TFlops\n", avg_time, tflops);
}

int main(int argc, char** argv) {
    constexpr int BLOCK_M = GemmTraits::B_M;
    constexpr int BLOCK_N = GemmTraits::B_N;
    constexpr int BLOCK_K = GemmTraits::B_K;
    constexpr int BLOCK_SIZE = GemmTraits::BLOCK_SIZE;

    int M = 256;
    int N = 512;
    int K = 256;
    int batch = 8;
    int verify = 0;

    auto parse_val = [](const char* arg, const char* flag) -> const char* {
        const std::size_t len = std::strlen(flag);
        if (std::strncmp(arg, flag, len) == 0) {
            if (arg[len] == '=') {
                return arg + len + 1;
            }
            if (arg[len] == '\0') {
                return reinterpret_cast<const char*>(1);
            }
        }
        return nullptr;
    };
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        const char* val = nullptr;
        auto try_parse = [&](int& target, const char* short_flag, const char* long_flag) {
            if ((val = parse_val(arg, short_flag)) || (long_flag && (val = parse_val(arg, long_flag)))) {
                if (val == reinterpret_cast<const char*>(1)) {
                    if (i + 1 < argc) {
                        target = std::atoi(argv[++i]);
                    }
                } else {
                    target = std::atoi(val);
                }
                return true;
            }
            return false;
        };
        if (try_parse(M, "-m", "--m")) continue;
        if (try_parse(N, "-n", "--n")) continue;
        if (try_parse(K, "-k", "--k")) continue;
        if (try_parse(batch, "-b", "--b")) continue;
        if (try_parse(verify, "-v", "--verify")) continue;
    }

    if (M <= 0 || N <= 0 || K <= 0 || batch <= 0) {
        std::cerr << "Invalid problem size: M, N, K and batch must be positive.\n";
        return 1;
    }

    constexpr int GROUP_M = GemmTraits::GROUP_M;
    constexpr int GROUP_N = GemmTraits::GROUP_N;
    constexpr int GROUP_K = GemmTraits::GROUP_K;
    if (M % GROUP_M != 0 || N % GROUP_N != 0 || K % GROUP_K != 0) {
        std::cerr << "M/N/K must be multiples of GROUP_M/GROUP_N/GROUP_K ("
                  << GROUP_M << "," << GROUP_N << "," << GROUP_K << ") for scale factors.\n";
        return 1;
    }

    const int num_groups_m = M / GROUP_M;
    const int num_groups_n = N / GROUP_N;
    const int num_groups_k = K / GROUP_K;

    auto host_a = std::make_unique<host_fp8_t[]>(static_cast<std::size_t>(batch) * M * K);
    auto host_b = std::make_unique<host_fp8_t[]>(static_cast<std::size_t>(batch) * N * K);
    std::unique_ptr<fp32_t[]> host_c;
    std::unique_ptr<fp32_t[]> host_c_out;
    if (verify) {
        host_c = std::make_unique<fp32_t[]>(static_cast<std::size_t>(batch) * M * N);
        host_c_out = std::make_unique<fp32_t[]>(static_cast<std::size_t>(batch) * M * N);
    }

    const std::size_t sfa_count = static_cast<std::size_t>(batch) * num_groups_m * num_groups_k;
    const std::size_t sfb_count = static_cast<std::size_t>(batch) * num_groups_n * num_groups_k;
    auto host_sfa = std::make_unique<fp32_t[]>(sfa_count);
    auto host_sfb = std::make_unique<fp32_t[]>(sfb_count);

    rand_vector(host_a.get(), static_cast<std::size_t>(batch) * M * K, 0.0f, 1.0f);
    rand_vector(host_b.get(), static_cast<std::size_t>(batch) * N * K, -0.5f, 0.5f);
    rand_vector(host_sfa.get(), sfa_count, 0.8f, 1.2f);
    rand_vector(host_sfb.get(), sfb_count, 0.8f, 1.2f);

    // Stage 1: host K-panel preshuffle (B -> B'), keeps host_b as [N,K] for the CPU reference.
    auto host_b_shuffled = std::make_unique<host_fp8_t[]>(static_cast<std::size_t>(batch) * N * K);
    shuffle_b(host_b.get(), host_b_shuffled.get(), batch, N, K, BLOCK_K);

    // Stage 2: host fragment repack (B' -> B''), replacing the GPU repack kernel entirely.
    auto host_b_pp = std::make_unique<host_fp8_t[]>(static_cast<std::size_t>(batch) * N * K);
    host_pack_b<GemmTraits>(host_b_shuffled.get(), host_b_pp.get(), batch, N, K, BLOCK_K);

    void* dev_a = nullptr;
    void* dev_b_pp = nullptr;
    void* dev_sfa = nullptr;
    void* dev_sfb = nullptr;
    fp32_t* dev_c = nullptr;
    CHECK_HIP(hipMalloc(&dev_a, static_cast<std::size_t>(batch) * M * K * sizeof(host_fp8_t)));
    CHECK_HIP(hipMalloc(&dev_b_pp, static_cast<std::size_t>(batch) * N * K * sizeof(host_fp8_t)));
    CHECK_HIP(hipMalloc(&dev_c, static_cast<std::size_t>(batch) * M * N * sizeof(fp32_t)));
    CHECK_HIP(hipMalloc(&dev_sfa, sfa_count * sizeof(fp32_t)));
    CHECK_HIP(hipMalloc(&dev_sfb, sfb_count * sizeof(fp32_t)));

    CHECK_HIP(hipMemcpy(dev_a, host_a.get(), static_cast<std::size_t>(batch) * M * K * sizeof(host_fp8_t), hipMemcpyHostToDevice));
    // B'' is produced on the host; upload it directly. No dev_b / no GPU repack kernel.
    CHECK_HIP(hipMemcpy(dev_b_pp, host_b_pp.get(), static_cast<std::size_t>(batch) * N * K * sizeof(host_fp8_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_sfa, host_sfa.get(), sfa_count * sizeof(fp32_t), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dev_sfb, host_sfb.get(), sfb_count * sizeof(fp32_t), hipMemcpyHostToDevice));

    opus_gemm_kargs kargs{};
    kargs.ptr_a = dev_a;
    kargs.ptr_b = nullptr;         // unused: no GPU repack input
    kargs.ptr_b_pp = dev_b_pp;     // host-packed compact fragments -> main gemm input
    kargs.ptr_c = dev_c;
    kargs.m = M;
    kargs.n = N;
    kargs.k = K;
    kargs.batch = batch;
    kargs.stride_a = K;
    kargs.stride_b = BLOCK_K;
    kargs.stride_c = N;
    kargs.stride_a_batch = M * K;
    kargs.stride_b_batch = N * K;
    kargs.stride_b_pp_batch = N * K;
    kargs.stride_c_batch = M * N;
    kargs.ptr_sfa = dev_sfa;
    kargs.ptr_sfb = dev_sfb;
    kargs.stride_sfa = num_groups_m;
    kargs.stride_sfb = num_groups_k;
    kargs.stride_sfa_batch = num_groups_m * num_groups_k;
    kargs.stride_sfb_batch = num_groups_n * num_groups_k;

    const int num_tiles_m = ceil_div(M, BLOCK_M);
    const int num_tiles_n = ceil_div(N, BLOCK_N);
    dim3 grid(num_tiles_m * num_tiles_n, 1, batch);
    dim3 block(BLOCK_SIZE);

    std::printf("Launching GEMM kernel (host-packed B''): M=%d, N=%d, K=%d, grid=(%u,%u,%u), block=%d\n",
                M, N, K, grid.x, grid.y, grid.z, BLOCK_SIZE);

    gemm_a8w8_blockscale_ldsbypass_kernel<GemmTraits><<<grid, block>>>(kargs);
    CHECK_HIP_KERNEL_LAUNCH();

    if (verify) {
        std::printf("\nValidating GPU results against CPU reference...\n");
        CHECK_HIP(hipMemcpy(host_c_out.get(), dev_c, static_cast<std::size_t>(batch) * M * N * sizeof(fp32_t),
                            hipMemcpyDeviceToHost));

        bool all_valid = true;
        for (int b = 0; b < batch; ++b) {
            gemm_ref(host_a.get() + static_cast<std::size_t>(b) * M * K,
                     host_b.get() + static_cast<std::size_t>(b) * N * K,
                     host_sfa.get() + static_cast<std::size_t>(b) * kargs.stride_sfa_batch,
                     host_sfb.get() + static_cast<std::size_t>(b) * kargs.stride_sfb_batch,
                     host_c.get() + static_cast<std::size_t>(b) * M * N,
                     M, N, K, K, K, N, kargs.stride_sfa, kargs.stride_sfb, GROUP_M, GROUP_N, GROUP_K);
            const bool valid = valid_vector(host_c.get() + static_cast<std::size_t>(b) * M * N,
                                            host_c_out.get() + static_cast<std::size_t>(b) * M * N, M * N, 1e-3f);
            std::printf("[GEMM batch %d/%d: %dx%dx%d, block_%dx%dx%d] %s\n",
                        b + 1, batch, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, valid ? "VALID" : "FAIL");
            all_valid = all_valid && valid;
        }

        std::printf("\n[Overall] %s\n", all_valid ? "ALL BATCHES VALID" : "SOME BATCHES FAILED");
    }

    std::printf("\n");
    benchmark_kernel<GemmTraits>(kargs, grid, block);
    std::printf("\n");

    CHECK_HIP(hipFree(dev_a));
    CHECK_HIP(hipFree(dev_b_pp));
    CHECK_HIP(hipFree(dev_c));
    CHECK_HIP(hipFree(dev_sfa));
    CHECK_HIP(hipFree(dev_sfb));

    return 0;
}
