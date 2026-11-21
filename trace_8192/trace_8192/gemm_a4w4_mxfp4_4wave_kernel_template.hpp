#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "gemm_a4w4_mxfp4_4wave_traits.hpp"

using opus::operator""_I;

// ===========================================================================
// A: global -> LDS -> registers (the fp8 4-wave geometry: 128-byte rows).
// ===========================================================================
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K_BYTES / T::VEC_A;                 // 8
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;     // 32
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;       // 8
    static_assert(threads_k == T::B_K / T::VEC_A_ELEMS);               // 256 fp4 / 32 fp4 per lane

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_M / threads_m_per_block>{},             // 4  y issue
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<threads_m_per_wave>{},                            // 8  p lane / 8
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<threads_k>{},                                     // 8  p lane % 8
        opus::number<T::VEC_A>{});                                     // 16 y bytes
    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m, lane_id % threads_k}));
}

template<class T>
__device__ inline auto make_layout_sa(int wave_id_m, int wave_id_n) {
    static_assert(T::smem_sub_a == T::WARP_SIZE / (T::B_K_BYTES / T::VEC_A), "one wave issue = one chunk");

    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::smem_m_rep / T::NUM_WAVES>{},                  // 4  y issue
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::VEC_A>{});                                     // 16 y bytes
    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave_a + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template<class T>
__device__ inline auto make_layout_ra(int lane_id, int wave_id_m) {
    static_assert(T::smem_sub_a * T::T_M == T::W_M && T::T_M == 2);
    constexpr int reads_per_operand = T::W_M * T::W_K_BYTES / T::WARP_SIZE / T::VEC_A;   // 1 b128 per fp4 operand
    static_assert(reads_per_operand == 1);
    static_assert(T::WARP_SIZE / T::W_M * T::VEC_A_ELEMS == T::W_K);   // 4 lane groups x 32 fp4 = K 128

    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},                                        // 4  y M repeat
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::T_M>{},                                        // 2  p r % 2
        opus::number<T::W_M / T::T_M>{},                               // 8  p r / 2
        opus::number<T::E_K>{},                                        // 2  p kk, fixed to 0 (keeps the 128 B row stride); kk -> + kk * W_K_BYTES
        opus::number<reads_per_operand>{},                             // 1  y b128 per operand
        opus::number<T::WARP_SIZE / T::W_M>{},                         // 4  p g = lane / 16
        opus::number<T::VEC_A>{});                                     // 16 y bytes
    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int lane_m = lane_id % T::W_M;
    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave_a + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_m % T::T_M, lane_m / T::T_M, 0_I, lane_id / T::W_M}));
}

template<class T>
__device__ inline auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    constexpr int threads_k = T::B_K_BYTES / T::VEC_B;                 // 8
    constexpr int threads_n_per_block = T::BLOCK_SIZE / threads_k;     // 32
    constexpr int threads_n_per_wave = T::WARP_SIZE / threads_k;       // 8
    static_assert(threads_k == T::B_K / T::VEC_B_ELEMS);

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_N / threads_n_per_block>{},             // 4  y issue
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<threads_n_per_wave>{},                            // 8  p lane / 8
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<threads_k>{},                                     // 8  p lane % 8
        opus::number<T::VEC_B>{});                                     // 16 y bytes
    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_m, lane_id / threads_k, wave_id_n, lane_id % threads_k}));
}

template<class T>
__device__ inline auto make_layout_sb(int wave_id_m, int wave_id_n) {
    static_assert(T::smem_sub_b == T::WARP_SIZE / (T::B_K_BYTES / T::VEC_B), "one wave issue = one chunk");

    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::smem_n_rep / T::NUM_WAVES>{},                  // 4  y issue
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::VEC_B>{});                                     // 16 y bytes
    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sb_block_shape,
        opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave_b + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_m, wave_id_n}));
}

template<class T>
__device__ inline auto make_layout_rb(int lane_id, int wave_id_n) {
    static_assert(T::smem_sub_b * T::T_N == T::W_N && T::T_N == 2);
    constexpr int reads_per_operand = T::W_N * T::W_K_BYTES / T::WARP_SIZE / T::VEC_B;   // 1 b128 per fp4 operand
    static_assert(reads_per_operand == 1);
    static_assert(T::WARP_SIZE / T::W_N * T::VEC_B_ELEMS == T::W_K);

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},                                        // 4  y N repeat
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::T_N>{},                                        // 2  p r % 2
        opus::number<T::W_N / T::T_N>{},                               // 8  p r / 2
        opus::number<T::E_K>{},                                        // 2  p kk, fixed to 0 (keeps the 128 B row stride); kk -> + kk * W_K_BYTES
        opus::number<reads_per_operand>{},                             // 1  y b128 per operand
        opus::number<T::WARP_SIZE / T::W_N>{},                         // 4  p g = lane / 16
        opus::number<T::VEC_B>{});                                     // 16 y bytes
    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int lane_n = lane_id % T::W_N;
    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave_b + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_n % T::T_N, lane_n / T::T_N, 0_I, lane_id / T::W_N}));
}

template<class T>
__device__ inline auto make_layout_gsfa(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
    constexpr auto gsfa_block_shape = opus::make_tuple(
        opus::number<T::B_M / T::BLOCK_SIZE>{},                        // 1  y M repeat
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::WARP_SIZE>{},                                  // 64 p lane = row
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // 16 y bytes
    constexpr auto gsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        gsfa_block_shape,
        opus::unfold_x_stride(gsfa_block_dim, gsfa_block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(gsfa_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

template<class T>
__device__ inline auto make_layout_ssfa(int wave_id_m, int wave_id_n) {
    constexpr int wave_bytes = T::WARP_SIZE * T::VEC_GLOBAL_SCALE;    // 1024
    constexpr auto ssfa_block_shape = opus::make_tuple(
        opus::number<T::B_M / T::BLOCK_SIZE>{},                        // 1  y M repeat
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // 16 y bytes
    constexpr auto ssfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        ssfa_block_shape,
        opus::unfold_x_stride(ssfa_block_dim, ssfa_block_shape, opus::tuple{opus::number<wave_bytes>{}, 1_I}),
        opus::unfold_p_coord(ssfa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template<class T>
__device__ inline auto make_layout_gsfb(int lane_id, int wave_id_m, int wave_id_n, int stride_sfb) {
    constexpr auto gsfb_block_shape = opus::make_tuple(
        opus::number<T::B_N / T::BLOCK_SIZE>{},                        // 1  y N repeat
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::WARP_SIZE>{},                                  // 64 p lane = row
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // 16 y bytes
    constexpr auto gsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        gsfb_block_shape,
        opus::unfold_x_stride(gsfb_block_dim, gsfb_block_shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(gsfb_block_dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
}

template<class T>
__device__ inline auto make_layout_ssfb(int wave_id_m, int wave_id_n) {
    constexpr int wave_bytes = T::WARP_SIZE * T::VEC_GLOBAL_SCALE;    // 1024
    constexpr auto ssfb_block_shape = opus::make_tuple(
        opus::number<T::B_N / T::BLOCK_SIZE>{},                        // 1  y N repeat
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // 16 y bytes
    constexpr auto ssfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        ssfb_block_shape,
        opus::unfold_x_stride(ssfb_block_dim, ssfb_block_shape, opus::tuple{opus::number<wave_bytes>{}, 1_I}),
        opus::unfold_p_coord(ssfb_block_dim, opus::tuple{wave_id_m, wave_id_n}));
}

template<class T>
__device__ inline auto make_layout_rsfa(int lane_id, int wave_id_m) {
    constexpr auto rsfa_block_shape = opus::make_tuple(
        opus::number<T::B_M / T::HALF_B_M>{},                          // 2  y half h   (128 rows)
        opus::number<T::WARP_SIZE / T::W_M>{},                         // 4  p g as e   (32 rows)
        opus::number<T::T_M>{},                                        // 2  p wave_m   (16 rows)
        opus::number<T::W_M>{},                                        // 16 p r        (1 row)
        opus::number<T::VEC_LDS_SCALE>{});                             // 8  y bytes
    constexpr auto rsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_LDS_SCALE>(
        rsfa_block_shape,
        opus::unfold_x_stride(rsfa_block_dim, rsfa_block_shape, opus::tuple{opus::number<T::VEC_GLOBAL_SCALE>{}, 1_I}),
        opus::unfold_p_coord(rsfa_block_dim, opus::tuple{lane_id / T::W_M, wave_id_m, lane_id % T::W_M}));
}

template<class T>
__device__ inline auto make_layout_rsfb(int lane_id, int wave_id_n) {
    constexpr auto rsfb_block_shape = opus::make_tuple(
        opus::number<T::B_N / T::HALF_B_N>{},                          // 2  y half h   (128 rows)
        opus::number<T::WARP_SIZE / T::W_N>{},                         // 4  p g as e   (32 rows)
        opus::number<T::T_N>{},                                        // 2  p wave_n   (16 rows)
        opus::number<T::W_N>{},                                        // 16 p r        (1 row)
        opus::number<T::VEC_LDS_SCALE>{});                             // 8  y bytes
    constexpr auto rsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_LDS_SCALE>(
        rsfb_block_shape,
        opus::unfold_x_stride(rsfb_block_dim, rsfb_block_shape, opus::tuple{opus::number<T::VEC_GLOBAL_SCALE>{}, 1_I}),
        opus::unfold_p_coord(rsfb_block_dim, opus::tuple{lane_id / T::W_N, wave_id_n, lane_id % T::W_N}));
}

__device__ inline unsigned int transpose_scale_dword_4x4(unsigned int packed, int scale_call) {
    const auto rows01_or_23 = __builtin_amdgcn_permlane16_swap(packed, packed, false, true);
    const unsigned int select_q01 = 0x06020400u + (scale_call & 1) * 0x01010101u;
    const unsigned int q01_or_q23 = __builtin_amdgcn_perm(rows01_or_23[1], rows01_or_23[0], select_q01);
    const auto rows0123 = __builtin_amdgcn_permlane32_swap(q01_or_q23, q01_or_q23, false, true);
    const unsigned int select_q = 0x05040100u + (scale_call >> 1) * 0x02020202u;
    return __builtin_amdgcn_perm(rows0123[1], rows0123[0], select_q);
}

// sched_group_barrier masks (LLVM AMDGPU)
constexpr int SCHED_VALU = 0x002;
constexpr int SCHED_MFMA = 0x008;
constexpr int SCHED_VMEM_READ = 0x020;
constexpr int SCHED_DS_READ = 0x100;

// REPS x (MFMAS MFMAs, then OTHERS instructions of OTHER_MASK) in one scheduling region
// (co-execution pattern of gcnasm block_scale_gemm's sched_barrier_pairs).
template<int REPS, int MFMAS, int OTHER_MASK, int OTHERS, int GROUP = 0>
__device__ inline void sched_pairs() {
    opus::static_for<REPS>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, MFMAS, GROUP);
        __builtin_amdgcn_sched_group_barrier(OTHER_MASK, OTHERS, GROUP);
    });
}

template<class T, int M_REPEAT, int N_REPEAT, class MMA, class VC>
__device__ inline void mma_scale_one(
    MMA& mma,
    const typename opus::remove_cvref_t<MMA>::vtype_a& v_a,
    const typename opus::remove_cvref_t<MMA>::vtype_b& v_b,
    VC& v_c,
    unsigned int v_sfa,
    unsigned int v_sfb) {
    (void)mma;
    using tiled_mma = opus::remove_cvref_t<MMA>;
    using base_mma = typename tiled_mma::MMA;
    using i32x4_t = opus::vector_t<opus::i32_t, 4>;
    using i32x8_t = opus::vector_t<opus::i32_t, 8>;
    using va_dw_t = opus::vector_t<opus::i32_t, sizeof(typename tiled_mma::vtype_a) / 4>;
    using vb_dw_t = opus::vector_t<opus::i32_t, sizeof(typename tiled_mma::vtype_b) / 4>;

    constexpr int a_offset = M_REPEAT * (T::VEC_A / 4);   // dwords
    constexpr int b_offset = N_REPEAT * (T::VEC_B / 4);
    constexpr int c_len = tiled_mma::mma_c_len;
    constexpr int c_offset = (M_REPEAT * T::E_N + N_REPEAT) * c_len;

    const i32x4_t s_a = opus::slice(__builtin_bit_cast(va_dw_t, v_a), opus::number<a_offset>{}, opus::number<a_offset + 4>{});
    const i32x4_t s_b = opus::slice(__builtin_bit_cast(vb_dw_t, v_b), opus::number<b_offset>{}, opus::number<b_offset + 4>{});
    const i32x8_t s_a8 = __builtin_shufflevector(s_a, s_a, 0, 1, 2, 3, -1, -1, -1, -1);
    const i32x8_t s_b8 = __builtin_shufflevector(s_b, s_b, 0, 1, 2, 3, -1, -1, -1, -1);
    auto s_c = opus::slice(v_c, opus::number<c_offset>{}, opus::number<c_offset + c_len>{});
    s_c = base_mma{}(s_a8, s_b8, s_c, static_cast<int>(v_sfa), static_cast<int>(v_sfb),
                     opus::number<M_REPEAT>{}, opus::number<N_REPEAT>{});
    opus::set_slice(v_c, s_c, opus::number<c_offset>{}, opus::number<c_offset + c_len>{});
}

// ===========================================================================
// Kernel
// ===========================================================================
#ifndef BAR_STEP
#define BAR_STEP 8
#endif
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void gemm_a4w4_mxfp4_4wave_kernel(opus_gemm_mxfp4_scale_kargs kargs) {
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp4_t;    // math types of the MMA
    using D_B = opus::fp4_t;
    using D_ACC = opus::fp32_t;
    using D_C = opus::bf16_t;
    using D_AB = opus::u8_t;
    using D_SF = opus::e8m0_t;  // UE8M0 scale, 1 byte (load<16> = 16 bytes)

    const int num_tiles_n = ceil_div_scale(kargs.n, T::B_N);
    const int block_m = block_id_x() / num_tiles_n;
    const int block_n = block_id_x() % num_tiles_n;
    const int loops = kargs.k / T::B_K;
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    const int batch_id = block_id_z();
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    auto g_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a);
    auto g_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row * kargs.stride_sfa, static_cast<unsigned int>((kargs.m - row) * kargs.stride_sfa));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + col * kargs.stride_sfb, static_cast<unsigned int>((kargs.n - col) * kargs.stride_sfb));

    const auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    const auto u_sb = make_layout_sb<T>(wave_id_m, wave_id_n);
    const auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    const auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);
    const auto u_gsfa = make_layout_gsfa<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
    const auto u_ssfa = make_layout_ssfa<T>(wave_id_m, wave_id_n);
    const auto u_gsfb = make_layout_gsfb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfb);
    const auto u_ssfb = make_layout_ssfb<T>(wave_id_m, wave_id_n);
    const auto u_rsfa = make_layout_rsfa<T>(lane_id, wave_id_m);
    const auto u_rsfb = make_layout_rsfb<T>(lane_id, wave_id_n);

    constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave_a + T::smem_padding); 
    constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave_b + T::smem_padding);
    __shared__ char smem_a[smem_a_elem * 4 * sizeof(D_A)];
    __shared__ char smem_b[smem_b_elem * 4 * sizeof(D_B)];
    __shared__ char smem_sfa[2 * T::SFA_PANEL_BYTES * sizeof(D_SF)];
    __shared__ char smem_sfb[2 * T::SFB_PANEL_BYTES * sizeof(D_SF)];
    auto s_a = make_smem(reinterpret_cast<D_AB*>(smem_a));
    auto s_b = make_smem(reinterpret_cast<D_AB*>(smem_b));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, 1>{},          // K: one MFMA (W_K = 128); the E_K = 2 steps are v_a[.][kk]
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    using MMA = decltype(mma);
    static_assert(MMA::mma_a_len == T::W_K_BYTES * T::ELEMS_PER_BYTE / (T::WARP_SIZE / T::W_M));  // 32 fp4 per lane
    typename MMA::vtype_a v_a[2][T::E_K];   // [half][kk]: E_M fp4 operands, 16 B each
    typename MMA::vtype_b v_b[2][T::E_K];   // [half][kk]
    typename MMA::vtype_c v_c[2][2];
    clear(v_c[0][0]); 
    clear(v_c[0][1]); 
    clear(v_c[1][0]); 
    clear(v_c[1][1]);

    using D_SF_PACK = opus::u32_t;
    D_SF_PACK v_sfa[2][T::E_K];
    D_SF_PACK v_sfb[2][T::E_K];
    D_SF_PACK v_sfa_next[2][T::E_K];
    D_SF_PACK v_sfb_next[2][T::E_K];

    // ---- offsets ----
    auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K_BYTES; };
    auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K_BYTES; };
    auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
    auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * T::NUM_KGROUPS; };
    auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin * T::NUM_KGROUPS; };
    // Scale LDS: stage = panel buffer (switches every SCALE_PANEL_TILES K tiles), tile_in_panel = 8-byte slot in the row.
    auto ssfa_offset = [&](int stage, int tile_in_panel) { return stage * T::SFA_PANEL_BYTES + tile_in_panel * T::NUM_KGROUPS; };
    auto ssfb_offset = [&](int stage, int tile_in_panel) { return stage * T::SFB_PANEL_BYTES + tile_in_panel * T::NUM_KGROUPS; };

    // ---- global -> LDS ----
    auto prefetch_a = [&](int stage, int tile, int half_tile_m) {
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage, half_tile_m), ga_offset(half_tile_m, tile));
    };
    auto prefetch_b = [&](int stage, int tile, int half_tile_n) {
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage, half_tile_n), gb_offset(half_tile_n, tile));
    };
    auto prefetch_sfa = [&](int panel_k_begin) {
        async_load<T::VEC_GLOBAL_SCALE>(g_sfa, s_sfa.ptr, u_gsfa, u_ssfa + ssfa_offset((panel_k_begin / T::SCALE_PANEL_TILES) & 1, 0), gsfa_offset(panel_k_begin));
    };
    auto prefetch_sfb = [&](int panel_k_begin) {
        async_load<T::VEC_GLOBAL_SCALE>(g_sfb, s_sfb.ptr, u_gsfb, u_ssfb + ssfb_offset((panel_k_begin / T::SCALE_PANEL_TILES) & 1, 0), gsfb_offset(panel_k_begin));
    };

    // ---- LDS -> VGPR ----
    auto load_a = [&](int stage, int half_tile_m) {
        opus::static_for<T::E_K>([&](auto kk) { v_a[half_tile_m][kk.value] = __builtin_bit_cast(typename MMA::vtype_a, load<T::VEC_A>(s_a, u_ra + sa_offset(stage, half_tile_m) + kk.value * T::W_K_BYTES)); });
    };
    auto load_b = [&](int stage, int half_tile_n) {
        opus::static_for<T::E_K>([&](auto kk) { v_b[half_tile_n][kk.value] = __builtin_bit_cast(typename MMA::vtype_b, load<T::VEC_B>(s_b, u_rb + sb_offset(stage, half_tile_n) + kk.value * T::W_K_BYTES)); });
    };
    opus::vector_t<D_SF, 2 * T::VEC_LDS_SCALE> v_sfa_raw, v_sfb_raw;  
    auto load_sfa = [&](int tile) { v_sfa_raw = load<T::VEC_LDS_SCALE>(s_sfa, u_rsfa + ssfa_offset((tile / T::SCALE_PANEL_TILES) & 1, tile % T::SCALE_PANEL_TILES)); };
    auto load_sfb = [&](int tile) { v_sfb_raw = load<T::VEC_LDS_SCALE>(s_sfb, u_rsfb + ssfb_offset((tile / T::SCALE_PANEL_TILES) & 1, tile % T::SCALE_PANEL_TILES)); };

    //
    // Before (load<8> x 2 halves = 16 e8m0, row-major; lane 16g+r holds row g*32+r and row 128+g*32+r):
    //            dword0 [0][0]        dword1 [0][1]        dword2 [1][0]                dword3 [1][1]
    // lane 0     M0k0  M0k1  M0k2  M0k3   M0k4  M0k5  M0k6  M0k7   M128k0 M128k1 M128k2 M128k3   M128k4 M128k5 M128k6 M128k7
    // lane 16    M32k0 M32k1 M32k2 M32k3  M32k4 M32k5 M32k6 M32k7  M160k0 M160k1 M160k2 M160k3   M160k4 M160k5 M160k6 M160k7
    // lane 32    M64k0 M64k1 M64k2 M64k3  M64k4 M64k5 M64k6 M64k7  M192k0 M192k1 M192k2 M192k3   M192k4 M192k5 M192k6 M192k7
    // lane 48    M96k0 M96k1 M96k2 M96k3  M96k4 M96k5 M96k6 M96k7  M224k0 M224k1 M224k2 M224k3   M224k4 M224k5 M224k6 M224k7
    //
    // After (byte e = repeat e, picked by op_sel; lane group g = K group g of the MFMA):
    //            v_sfa[0][0]           v_sfa[0][1]           v_sfa[1][0]                 v_sfa[1][1]
    // lane 0     M0k0 M32k0 M64k0 M96k0  M0k4 M32k4 M64k4 M96k4  M128k0 M160k0 M192k0 M224k0  M128k4 M160k4 M192k4 M224k4
    // lane 16    M0k1 M32k1 M64k1 M96k1  M0k5 M32k5 M64k5 M96k5  M128k1 M160k1 M192k1 M224k1  M128k5 M160k5 M192k5 M224k5
    // lane 32    M0k2 M32k2 M64k2 M96k2  M0k6 M32k6 M64k6 M96k6  M128k2 M160k2 M192k2 M224k2  M128k6 M160k6 M192k6 M224k6
    // lane 48    M0k3 M32k3 M64k3 M96k3  M0k7 M32k7 M64k7 M96k7  M128k3 M160k3 M192k3 M224k3  M128k7 M160k7 M192k7 M224k7
    //
    using SF_DW = opus::vector_t<D_SF_PACK, 2 * T::E_K>;
    auto transpose_sfa = [&](D_SF_PACK (&sfa)[2][T::E_K]) {
        const auto dw = __builtin_bit_cast(SF_DW, v_sfa_raw);
        opus::static_for<2 * T::E_K>([&](auto i) {
            sfa[i.value / T::E_K][i.value % T::E_K] = transpose_scale_dword_4x4(dw[i.value], lane_id / T::W_M);
        });
    };
    auto transpose_sfb = [&](D_SF_PACK (&sfb)[2][T::E_K]) {
        const auto dw = __builtin_bit_cast(SF_DW, v_sfb_raw);
        opus::static_for<2 * T::E_K>([&](auto i) {
            sfb[i.value / T::E_K][i.value % T::E_K] = transpose_scale_dword_4x4(dw[i.value], lane_id / T::W_N);
        });
    };

    // ===== Prologue =====
    // K tile 0, scale panels 0 / 1 and K tile 1 are all issued up front; only
    // tile 0 and the panels are waited for (tile 1 keeps flying into iteration 0).
    prefetch_a(0, 0, 0);
    prefetch_b(0, 0, 0);
    prefetch_a(0, 0, 1);
    prefetch_b(0, 0, 1);
    prefetch_sfa(0);
    prefetch_sfb(0);
    if (loops > T::SCALE_PANEL_TILES) { prefetch_sfa(T::SCALE_PANEL_TILES); prefetch_sfb(T::SCALE_PANEL_TILES); }
    if (loops > 1) {
        prefetch_a(1, 1, 0);
        prefetch_b(1, 1, 0);
        prefetch_a(1, 1, 1);
        prefetch_b(1, 1, 1);
        s_waitcnt_vmcnt(number<2 * T::a_buffer_load_insts + 2 * T::b_buffer_load_insts>{});
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    // A0 / B0 of tile 0 (A1 / B1 are read at the start of iteration 0).
    load_a(0, 0);
    load_b(0, 0);
    load_sfa(0);
    load_sfb(0);
    transpose_sfa(v_sfa);
    transpose_sfb(v_sfb);
    __builtin_amdgcn_sched_barrier(0);

    // ===== Main loop =====
    // Iteration `tile`: A0 / B0 of K tile `tile` are in registers, A1 / B1 are read
    // from LDS `stage` during the first steps of C00 (C00 does not use them), K tile+1
    // is in LDS next_stage, K tile+2 is copied into `stage` after the barrier.
    // One step = 2 MFMAs (kk = 0, 1).  Each region (between sched_barrier(0)) gives
    // the scheduler a fixed MFMA / DS read / buffer_load / VALU interleave, so that
    // the single wave of each SIMD keeps the MFMA pipe fed (co-execution):
    //   R1  C00 steps 0..BAR-1 : A1, B1 of `tile` (16 ds_read_b128), 1 per MFMA
    //       lgkmcnt(0) + vmcnt(0) + s_barrier  (`stage` fully read, K tile+1 landed)
    //   R2  C00 steps BAR..15  : scales of tile+1 (2 DS), copy A0 of tile+2 (4 VMEM)
    //   R3  C10                : copy B0, A1 of tile+2 (8 VMEM), transpose scales (VALU)
    //   R4  C01                : B0 of tile+1 (8 DS), copy B1 of tile+2 (4 VMEM), scale panel
    //   R5  C11                : A0 of tile+1 (8 DS)
    // HAS_NEXT: tile+1 exists; HAS_LATER: tile+2 exists.
    constexpr int frags = T::E_M * T::E_N;
    auto steps = [&](auto hm_, auto hn_, auto first_, auto last_) {
        constexpr int hm = decltype(hm_)::value, hn = decltype(hn_)::value;
        opus::static_for<decltype(last_)::value - decltype(first_)::value>([&](auto i) {
            constexpr int s = decltype(first_)::value + decltype(i)::value;
            mma_scale_one<T, s / T::E_N, s % T::E_N>(mma, v_a[hm][0], v_b[hn][0], v_c[hm][hn], v_sfa[hm][0], v_sfb[hn][0]);
            mma_scale_one<T, s / T::E_N, s % T::E_N>(mma, v_a[hm][1], v_b[hn][1], v_c[hm][hn], v_sfa[hm][1], v_sfb[hn][1]);
        });
    };
    auto iteration = [&](int tile, auto has_next_, auto has_later_) {
        constexpr bool has_next = decltype(has_next_)::value;
        constexpr bool has_later = decltype(has_later_)::value;
        const int stage = tile & 1;
        const int next_stage = stage ^ 1;
        // Second tile of a panel: copy the panel after next (K tile tile + 3); its LDS
        // buffer last held K tiles tile - 1 / tile, read before this iteration's barrier.
        const int next_panel_k_begin = tile + 1 + T::SCALE_PANEL_TILES;
        const bool do_panel_copy = (tile % T::SCALE_PANEL_TILES) == 1 && next_panel_k_begin < loops;

        // R1: C00 steps 0..BAR-1 + A1 / B1 of this tile.
        load_a(stage, 1);
        load_b(stage, 1);
        steps(number<0>{}, number<0>{}, number<0>{}, number<BAR_STEP>{});
        sched_pairs<2 * BAR_STEP, 1, SCHED_DS_READ, 1>();
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_lgkmcnt(0_I);
        if constexpr (has_next) {
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        __builtin_amdgcn_sched_barrier(0);

        // R2: C00 steps BAR..15.
        if constexpr (has_next) { load_sfa(tile + 1); load_sfb(tile + 1); }
        if constexpr (has_later) prefetch_a(stage, tile + 2, 0);
        steps(number<0>{}, number<0>{}, number<BAR_STEP>{}, number<frags>{});
        sched_pairs<2, 1, SCHED_DS_READ, 1>();
        sched_pairs<T::a_buffer_load_insts, 3, SCHED_VMEM_READ, 1>();
        __builtin_amdgcn_sched_barrier(0);

        // R3: C10.
        if constexpr (has_later) { prefetch_b(stage, tile + 2, 0); prefetch_a(stage, tile + 2, 1); }
        if constexpr (has_next) { transpose_sfa(v_sfa_next); transpose_sfb(v_sfb_next); }
        steps(number<1>{}, number<0>{}, number<0>{}, number<frags>{});
        sched_pairs<2 * T::a_buffer_load_insts, 4, SCHED_VMEM_READ, 1>();
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (has_next) opus::static_for<T::E_K>([&](auto kk) { v_sfb[0][kk.value] = v_sfb_next[0][kk.value]; });

        // Scale panel copy: a runtime branch, kept on a region boundary (a branch
        // inside a region splits its basic block and defeats sched_group_barrier).
        if (do_panel_copy) { prefetch_sfa(next_panel_k_begin); prefetch_sfb(next_panel_k_begin); }
        __builtin_amdgcn_sched_barrier(0);

        // R4: C01.
        if constexpr (has_next) load_b(next_stage, 0);
        if constexpr (has_later) prefetch_b(stage, tile + 2, 1);
        steps(number<0>{}, number<1>{}, number<0>{}, number<frags>{});
        sched_pairs<T::E_N * T::E_K, 2, SCHED_DS_READ, 1>();
        sched_pairs<T::b_buffer_load_insts, 4, SCHED_VMEM_READ, 1>();
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (has_next) opus::static_for<T::E_K>([&](auto kk) { v_sfa[0][kk.value] = v_sfa_next[0][kk.value]; });

        // R5: C11.
        if constexpr (has_next) load_a(next_stage, 0);
        steps(number<1>{}, number<1>{}, number<0>{}, number<frags>{});
        sched_pairs<T::E_M * T::E_K, 2, SCHED_DS_READ, 1>();
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (has_next) {
            opus::static_for<T::E_K>([&](auto kk) {
                v_sfa[1][kk.value] = v_sfa_next[1][kk.value];
                v_sfb[1][kk.value] = v_sfb_next[1][kk.value];
            });
        }
    };

    int tile = 0;
#pragma clang loop unroll(disable)
    for (; tile + 2 < loops; ++tile) iteration(tile, number<true>{}, number<true>{});
    if (tile + 1 < loops) { iteration(tile, number<true>{}, number<false>{}); ++tile; }
    iteration(tile, number<false>{}, number<false>{});

    // ===== Epilogue: C (bf16) through LDS, full cache-line stores =====
    // A wave's MFMA fragments cover 4 columns of a row per lane; written directly they
    // give partial 128 B lines.  Each 128 x 128 quadrant is converted to bf16 and staged
    // in the A or B LDS buffer (alternating, one barrier per quadrant; rows padded to
    // 136 elements = 272 B to stagger banks), then copied out row-contiguously: one
    // buffer_store = 4 rows x 256 B = 8 full lines.
    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    constexpr int c_lds_stride = T::HALF_B_N + 8;
    constexpr int VEC_OUT = 16 / sizeof(D_C);                                            // 8 bf16 = 16 B
    static_assert(T::HALF_B_M * c_lds_stride * sizeof(D_C) <= sizeof(smem_a) && sizeof(smem_a) == sizeof(smem_b));
    auto s_c0 = make_smem(reinterpret_cast<D_C*>(smem_a));
    auto s_c1 = make_smem(reinterpret_cast<D_C*>(smem_b));
    auto u_sc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(opus::number<c_lds_stride>{}, 1_I), p_coord_c);
    constexpr int c_copies = T::HALF_B_M * T::HALF_B_N / (T::BLOCK_SIZE * VEC_OUT);       // 8
    auto copy_out = [&](auto& s_c, int hm, int hn) {
        opus::static_for<c_copies>([&](auto i) {
            const int linear = (decltype(i)::value * T::BLOCK_SIZE + static_cast<int>(thread_id_x())) * VEC_OUT;
            const int r = linear / T::HALF_B_N, c = linear % T::HALF_B_N;
            const auto v = load<VEC_OUT>(s_c, r * c_lds_stride + c);
#ifdef DIAG_NO_STORE
            if (kargs.m < 0)   // diagnostic: never true
#endif
            store<VEC_OUT>(g_c, v, (hm * T::HALF_B_M + r) * kargs.stride_c + hn * T::HALF_B_N + c);
        });
    };
    auto sync_lds = [&]() {
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };
    sync_lds();                                   // main-loop LDS reads done in every wave
    store<T::VEC_C>(s_c0, cast<D_C>(v_c[0][0]), u_sc);
    sync_lds();
    copy_out(s_c0, 0, 0);
    store<T::VEC_C>(s_c1, cast<D_C>(v_c[1][0]), u_sc);
    sync_lds();
    copy_out(s_c1, 1, 0);
    store<T::VEC_C>(s_c0, cast<D_C>(v_c[0][1]), u_sc);
    sync_lds();
    copy_out(s_c0, 0, 1);
    store<T::VEC_C>(s_c1, cast<D_C>(v_c[1][1]), u_sc);
    sync_lds();
    copy_out(s_c1, 1, 1);
}
