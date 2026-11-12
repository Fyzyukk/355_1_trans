#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "gemm_a4w4_mxfp4_4wave_traits.hpp"

using opus::operator""_I;

// ===========================================================================
// A: global -> LDS -> VGPR
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
    // Writer wave (wave_id_n, wave_id_m), lane (e = lane / 16, r = lane % 16) loads row wave_id_n * 128 + e * 32 +
    // wave_id_m * 16 + r: after the 4x4 byte transpose across e, lane (g, r) holds byte e = repeat e,
    // K group g -- the MFMA scale layout of wave (h = wave_id_n, w = wave_id_m).
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p wave_id_n: half (128 rows)
        opus::number<4>{},                                             // p e (32 rows)
        opus::number<2>{},                                             // p wave_id_m (16 rows)
        opus::number<16>{},                                            // p r
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // y 16 bytes (panel)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    static_assert(2 * 4 * 2 * 16 == T::B_M && T::B_M == T::B_N);
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_n, lane_id / 16, wave_id_m, lane_id % 16}));
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
    // Writer wave (wave_id_m, wave_id_n), lane (e = lane / 16, r = lane % 16) loads row wave_id_m * 128 + e * 32 +
    // wave_id_n * 16 + r: after the 4x4 byte transpose across e, lane (g, r) holds byte e = repeat e,
    // K group g -- the MFMA scale layout of wave (h = wave_id_m, w = wave_id_n).
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p wave_id_m: half (128 rows)
        opus::number<4>{},                                             // p e (32 rows)
        opus::number<2>{},                                             // p wave_id_n (16 rows)
        opus::number<16>{},                                            // p r
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // y 16 bytes (panel)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    static_assert(2 * 4 * 2 * 16 == T::B_M && T::B_M == T::B_N);
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, lane_id / 16, wave_id_n, lane_id % 16}));
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
constexpr int SCHED_VMEM_READ = 0x820;   // 0x020 VMEM read | 0x800 LDS DMA (newer LLVM classifies buffer_load ... lds as LDS DMA)
constexpr int SCHED_DS_READ = 0x100;

template<int REPS, int MFMAS, int OTHER_MASK, int OTHERS, int GROUP = 0>
__device__ inline void sched_pairs() {
    opus::static_for<REPS>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, MFMAS, GROUP);
        __builtin_amdgcn_sched_group_barrier(OTHER_MASK, OTHERS, GROUP);
    });
}

// One MFMA, then DS reads and VALU.
template<int DS, int VALU>
__device__ inline void sched_m() {
    __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
    if constexpr (DS > 0) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, DS, 0);
    if constexpr (VALU > 0) __builtin_amdgcn_sched_group_barrier(SCHED_VALU, VALU, 0);
}
// C11 (32 MFMAs): 8 DS (A0 of t+1) + 24 VALU (scale transposes of half 0).
__device__ inline void sched_c11() {
    opus::static_for<8>([&](auto) { sched_m<1, 1>(); sched_m<0, 1>(); sched_m<0, 1>(); sched_m<0, 0>(); });
}
// C00 (32 MFMAs): 16 DS (A1 / B1 of t+1) in the first 16 MFMAs, so none is left at the end of
// the quadrant, + 24 VALU (scale transposes of half 1).
__device__ inline void sched_c00() {
    opus::static_for<16>([&](auto) { sched_m<1, 1>(); });
    opus::static_for<8>([&](auto) { sched_m<0, 1>(); });
    opus::static_for<8>([&](auto) { sched_m<0, 0>(); });
}

template<int REPS, int DS, int VMEM, int VALU>
__device__ inline void sched_x4() {
    static_assert(DS <= 2 && VMEM <= 1);
    opus::static_for<REPS>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (DS >= 1) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (VMEM >= 1) __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (DS >= 2) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (VALU > 0) __builtin_amdgcn_sched_group_barrier(SCHED_VALU, VALU, 0);
    });
}

// One fp4 MFMA operand per lane: 32 fp4 = 16 bytes = 4 registers.  Held as 4 x i32 because
// opus::vector_t<fp4_t, 32> is a bit-packed struct, not a register vector (the MFMA builtin
// and the pin attribute need an int vector).
using fp4x32_t = opus::vector_t<opus::i32_t, 4>;

template<class T, int M_REPEAT, int N_REPEAT, int C_REG, class MMA, class CF>
__device__ inline void mma_scale_one(
    MMA& mma,
    const fp4x32_t& a,
    const fp4x32_t& b,
    CF& c,
    unsigned int v_sfa,
    unsigned int v_sfb) {
    (void)mma;
    using base_mma = typename opus::remove_cvref_t<MMA>::MMA;
    using i32x8_t = opus::vector_t<opus::i32_t, 8>;
    const i32x8_t a8 = __builtin_shufflevector(a, a, 0, 1, 2, 3, -1, -1, -1, -1);
    const i32x8_t b8 = __builtin_shufflevector(b, b, 0, 1, 2, 3, -1, -1, -1, -1);
    if constexpr (C_REG >= 0) {
        [[clang::amdgpu_pin_agpr(C_REG)]] c = base_mma{}(a8, b8, c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<M_REPEAT>{}, opus::number<N_REPEAT>{});
    } else if constexpr (C_REG <= -2) {
        [[clang::amdgpu_pin_vgpr(-C_REG - 2)]] c = base_mma{}(a8, b8, c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<M_REPEAT>{}, opus::number<N_REPEAT>{});
    } else {
        c = base_mma{}(a8, b8, c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<M_REPEAT>{}, opus::number<N_REPEAT>{});
    }
}

template<class T, int HM, int HN, int M, int N>
constexpr int c_reg() {
    constexpr int frag = (M * T::E_N + N) * 4;
    if constexpr (HN == 0) return 128 + HM * 64 + frag;           // AGPR
    else return -(128 + HM * 64 + frag) - 2;                        // VGPR
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
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int tiles_per_batch = ceil_div_scale(kargs.m, T::B_M) * num_tiles_n;
    const int total_tiles = tiles_per_batch * kargs.batch;
    const int grid_blocks = static_cast<int>(__builtin_amdgcn_grid_size_x() / T::BLOCK_SIZE);
    // Tile 0 (+ scale panel 0) of this block's next output tile is copied during the last K
    // tile of the current one (see "2. Last tile"); the next prologue then skips it.
    int prefetched = 0;
    opus::vector_t<opus::e8m0_t, Traits::VEC_GLOBAL_SCALE> v_nsfa0, v_nsfb0;   // next tile's scale panel 0
#pragma clang loop unroll(disable)
    for (int tile_id = block_id_x(); tile_id < total_tiles; tile_id += grid_blocks) {
    const int batch_id = tile_id / tiles_per_batch;
    const int tile_in_batch = tile_id % tiles_per_batch;
    const int block_m = tile_in_batch / num_tiles_n;
    const int block_n = tile_in_batch % num_tiles_n;
    const int loops = kargs.k / T::B_K;
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    // lane id from mbcnt over an opaque all-ones mask: the compiler still knows it is in
    // [0, 64) but cannot hoist the lane-dependent offsets out of the tile loop (kept live
    // through the write-back they spill, and the reloads put vmcnt(0) into the prologue).
    unsigned int lane_mask = ~0u;
    asm volatile("" : "+s"(lane_mask));
    const int lane_id = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lane_mask, __builtin_amdgcn_mbcnt_lo(lane_mask, 0u)));
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
    constexpr int smem_a_bytes = smem_a_elem * 4 * sizeof(D_A);
    constexpr int smem_b_bytes = smem_b_elem * 4 * sizeof(D_B);
    __shared__ __attribute__((aligned(16))) char smem_ab[smem_a_bytes + smem_b_bytes];
    // Stage-major LDS: [A stage 0][B stage 0][A stage 1][B stage 1].  Stage 1 of A and B is
    // then one contiguous 66 KB block, which stages half of C at the original row stride while
    // stage 0 receives the next tile's K tile 0.
    char* const smem_a = smem_ab;
    char* const smem_b = smem_ab + smem_a_bytes / 2;
    __shared__ char smem_sfa[T::SCALE_PANEL_BUFFERS * T::SFA_PANEL_BYTES * sizeof(D_SF)];
    __shared__ char smem_sfb[T::SCALE_PANEL_BUFFERS * T::SFB_PANEL_BYTES * sizeof(D_SF)];
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
    fp4x32_t v_a[2][T::E_K][T::E_M];   // [half][kk]: E_M fp4 operands, 16 B each
    fp4x32_t v_b[2][T::E_K][T::E_N];
    using CFrag = opus::vector_t<D_ACC, MMA::mma_c_len>;   // one 16x16 C fragment, 4 fp32 per lane
    CFrag v_c[2][2][T::E_M * T::E_N];
    opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
        clear(v_c[hm.value][hn.value][f.value]);
    }); }); });

    using D_SF_PACK = opus::u32_t;
    D_SF_PACK v_sfa[2][T::E_K];
    D_SF_PACK v_sfb[2][T::E_K];

    // ---- offsets ----
    auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K_BYTES; };
    auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K_BYTES; };
    constexpr int stage_elems = 2 * smem_a_elem + 2 * smem_b_elem;   // bytes (D_AB = u8)
    auto sa_offset = [&](int stage, int half_tile_m) { return stage * stage_elems + half_tile_m * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return stage * stage_elems + half_tile_n * smem_b_elem; };
    auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * T::NUM_KGROUPS; };
    auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin * T::NUM_KGROUPS; };
    auto ssfa_offset = [&](int stage, int tile_in_panel) { return stage * T::SFA_PANEL_BYTES + tile_in_panel * T::NUM_KGROUPS; };
    auto ssfb_offset = [&](int stage, int tile_in_panel) { return stage * T::SFB_PANEL_BYTES + tile_in_panel * T::NUM_KGROUPS; };

    // ---- global -> LDS ----
    auto prefetch_a = [&](int stage, int tile, int half_tile_m) {
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage, half_tile_m), ga_offset(half_tile_m, tile));
    };
    auto prefetch_b = [&](int stage, int tile, int half_tile_n) {
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage, half_tile_n), gb_offset(half_tile_n, tile));
    };
    // Scale panels go global -> VGPR -> 4x4 byte transpose -> LDS, so LDS holds the MFMA scale
    // layout and the main loop reads it as is (one transpose per panel instead of one per wave
    // and tile).  A panel is loaded in iteration t and written in t + 1 (its load has landed by
    // B_b of t + 1).
    using SF_PANEL = opus::vector_t<D_SF, T::VEC_GLOBAL_SCALE>;
    using SF_DW4 = opus::vector_t<opus::u32_t, 4>;
    SF_PANEL v_sf_pend;
    auto fetch_sfa = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfa, u_gsfa, gsfa_offset(panel_k_begin)); };
    auto fetch_sfb = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfb, u_gsfb, gsfb_offset(panel_k_begin)); };
    auto put_sf = [&](auto& s_sf_, const auto& u_ssf, const SF_PANEL& raw, int buf) {
        const auto dw = __builtin_bit_cast(SF_DW4, raw);
        SF_DW4 t;
        opus::static_for<4>([&](auto d) { t[d.value] = transpose_scale_dword_4x4(dw[d.value], lane_id / 16); });
        store<T::VEC_GLOBAL_SCALE>(s_sf_, __builtin_bit_cast(SF_PANEL, t), u_ssf + (buf * T::SFA_PANEL_BYTES + lane_id * T::VEC_GLOBAL_SCALE));   // the LDS-DMA layout has no lane term
    };

    // ---- LDS -> VGPR ----
    const auto ra_offs = opus::layout_to_offsets<T::VEC_A>(u_ra);   // [repeat]
    const auto rb_offs = opus::layout_to_offsets<T::VEC_B>(u_rb);
    auto load_a = [&](auto& sa_, auto half_) {
        constexpr int half = decltype(half_)::value;
        opus::static_for<T::E_K>([&](auto kk) { opus::static_for<T::E_M>([&](auto e) {
            constexpr int agpr = 0 + half * 32 + (kk.value * T::E_M + e.value) * 4;
            [[clang::amdgpu_pin_agpr(agpr)]] v_a[half][kk.value][e.value] =
                __builtin_bit_cast(fp4x32_t, load<T::VEC_A>(sa_, ra_offs[e.value] + sa_offset(0, half) + kk.value * T::W_K_BYTES));
        }); });
    };
    auto load_b = [&](auto& sb_, auto half_) {
        constexpr int half = decltype(half_)::value;
        opus::static_for<T::E_K>([&](auto kk) { opus::static_for<T::E_N>([&](auto e) {
            constexpr int agpr = 64 + half * 32 + (kk.value * T::E_N + e.value) * 4;
            [[clang::amdgpu_pin_agpr(agpr)]] v_b[half][kk.value][e.value] =
                __builtin_bit_cast(fp4x32_t, load<T::VEC_B>(sb_, rb_offs[e.value] + sb_offset(0, half) + kk.value * T::W_K_BYTES));
        }); });
    };
    // Scale reads: lane reads the 2 dwords (kk = 0, 1) of half h at LDS index (h * 2 + w) * 64 +
    // lane (+ 8 B for the second tile of the panel), early (C01) into v_sf*_next; read_sf moves
    // half h into v_sfa / v_sfb when the previous values are no longer used.
    using SF_KK = opus::vector_t<D_SF_PACK, T::E_K>;
    const int sfa_rd_base = (wave_id_m * T::WARP_SIZE + lane_id) * T::VEC_GLOBAL_SCALE;
    const int sfb_rd_base = (wave_id_n * T::WARP_SIZE + lane_id) * T::VEC_GLOBAL_SCALE;
    constexpr int sf_half_stride = 2 * T::WARP_SIZE * T::VEC_GLOBAL_SCALE;
    SF_KK v_sfa_next[2], v_sfb_next[2];
    int sf_read_off = 0;
    auto load_sfa = [&](auto& rd_sfa_, int buf, int tile_in_panel) {
        sf_read_off = buf * T::SFA_PANEL_BYTES + tile_in_panel * T::NUM_KGROUPS;
        opus::static_for<2>([&](auto h) { v_sfa_next[h.value] = __builtin_bit_cast(SF_KK, load<T::VEC_LDS_SCALE>(rd_sfa_, sfa_rd_base + h.value * sf_half_stride + sf_read_off)); });
    };
    auto load_sfb = [&](auto& rd_sfb_, int, int) {
        opus::static_for<2>([&](auto h) { v_sfb_next[h.value] = __builtin_bit_cast(SF_KK, load<T::VEC_LDS_SCALE>(rd_sfb_, sfb_rd_base + h.value * sf_half_stride + sf_read_off)); });
    };

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
    auto read_sf = [&](auto&, auto&, auto h_) {
        constexpr int h = decltype(h_)::value;
        opus::static_for<T::E_K>([&](auto kk) { v_sfa[h][kk.value] = v_sfa_next[h][kk.value]; v_sfb[h][kk.value] = v_sfb_next[h][kk.value]; });
    };

    // LDS reads go through __restrict__ copies of the views, passed as function parameters
    // (clang ignores __restrict__ on locals): without them the compiler cannot tell the
    // reads from the in-flight LDS-DMA copies and puts a vmcnt(0) before the reads.
    auto views = [&](OPUS_LDS_ADDR char* __restrict__ read_a, OPUS_LDS_ADDR char* __restrict__ read_b,
                     OPUS_LDS_ADDR char* __restrict__ read_sfa, OPUS_LDS_ADDR char* __restrict__ read_sfb, auto&& body) {
        auto rd_a = s_a;     rd_a.ptr = read_a;
        auto rd_b = s_b;     rd_b.ptr = read_b;
        auto rd_sfa = s_sfa; rd_sfa.ptr = read_sfa;
        auto rd_sfb = s_sfb; rd_sfb.ptr = read_sfb;
        body(rd_a, rd_b, rd_sfa, rd_sfb);
    };

    // ===== Prologue =====
    constexpr int frags = T::E_M * T::E_N;   // 16 (m, n) steps per quadrant, 2 MFMAs each
    // Issue tile 0 and scale panel 0, then tile 1 and panel 1 (tile 2, 3); wait only for
    // tile 0 + panel 0 (counted vmcnt), read A0 / B0 and the scales of tile 0, and compute
    // C00 of tile 0 while A1 / B1 of tile 0 are read.  The main loop then starts with
    // "C00 of tile t is done".
    static_assert(T::SFA_PANEL_BYTES == T::SFB_PANEL_BYTES);
    if (!prefetched) {
        prefetch_a(0, 0, 0);
        prefetch_b(0, 0, 0);
        prefetch_a(0, 0, 1);
        prefetch_b(0, 0, 1);
        v_nsfa0 = fetch_sfa(0);
        v_nsfb0 = fetch_sfb(0);
    }
    // Panel 0 and sfb of panel 1 are transposed into LDS here; sfa of panel 1 stays in v_sf_pend
    // and is written in iteration 0.
    SF_PANEL v_sfb1;
    if (loops > T::SCALE_PANEL_TILES) {
        v_sfb1 = fetch_sfb(T::SCALE_PANEL_TILES);
        v_sf_pend = fetch_sfa(T::SCALE_PANEL_TILES);
    }
    if (loops > 1) {
        prefetch_a(1, 1, 0);
        prefetch_b(1, 1, 0);
        prefetch_a(1, 1, 1);
        prefetch_b(1, 1, 1);
        s_waitcnt_vmcnt(number<4 * T::a_buffer_load_insts + 1>{});      // tile 1 + pending sfa may fly
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    views(s_a.ptr, s_b.ptr, s_sfa.ptr, s_sfb.ptr, [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
        put_sf(rd_sfa, u_ssfa, v_nsfa0, 0);
        put_sf(rd_sfb, u_ssfb, v_nsfb0, 0);
        if (loops > T::SCALE_PANEL_TILES) put_sf(rd_sfb, u_ssfb, v_sfb1, 1);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        load_a(rd_a, number<0>{});
        load_b(rd_b, number<0>{});
        load_sfa(rd_sfa, 0, 0);
        load_sfb(rd_sfb, 0, 0);
        read_sf(rd_sfa, rd_sfb, number<0>{});
        read_sf(rd_sfa, rd_sfb, number<1>{});
        __builtin_amdgcn_sched_barrier(0);
        // C00 of tile 0 + A1, B1 of tile 0.
        load_a(rd_a, number<1>{});
        load_b(rd_b, number<1>{});
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][0][m], v_b[0][0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);   // kk = 0
            mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][1][m], v_b[0][1][n], v_c[0][0][m * T::E_N + n], v_sfa[0][1], v_sfb[0][1]);   // kk = 1
        });
        sched_x4<8, 2, 0, 0>();
        __builtin_amdgcn_sched_barrier(0);
    });

    // ===== Main loop =====
    int tile = 0;
    // Scale panel buffers tracked instead of (k / 2) % 3: sf_rd holds the panel of tile t+1,
    // sf_wr = sf_rd + 1 (mod 3) receives the panel copied in iteration t (t+3 for odd t,
    // t+2 for even t -- both one panel after tile t+1's); after an even t, sf_rd advances.
    int sf_rd = 0;
    // Two tiles per trip: stage / next_stage and the panel parity are compile-time in each
    // copy; A / B / C are pinned, so both copies use the same registers.
#pragma clang loop unroll(disable)
    for (; tile + 3 < loops; tile += 2) {
        {   // even tile
            constexpr int stage = 0;
            constexpr int next_stage = 1;
            const int sf_wr = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // B_a: every wave is done reading stage(t) (A1 / B1 of tile t in C00), so
                // tile t+2 may be copied into it.
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C10(t) + A0, B0 of tile t+2 -> stage(t), one buffer_load per 4 MFMAs.
                prefetch_a(stage, tile + 2, 0);
                prefetch_b(stage, tile + 2, 0);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][0][m], v_b[0][0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);   // kk = 0
                    mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][1][m], v_b[0][1][n], v_c[1][0][m * T::E_N + n], v_sfa[1][1], v_sfb[0][1]);   // kk = 1
                });
                sched_x4<8, 0, 1, 0>();
                __builtin_amdgcn_sched_barrier(0);
                // B_b: tile t+1 (and its scale panel) landed; the 8 copies just issued may still fly.
                s_waitcnt_vmcnt(number<2 * T::a_buffer_load_insts>{});
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
                load_sfa(rd_sfa, sf_rd, 1);
                load_sfb(rd_sfb, sf_rd, 1);
                load_b(rd_b, number<0>{});
                prefetch_a(stage, tile + 2, 1);
                prefetch_b(stage, tile + 2, 1);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][0][m], v_b[1][0][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);   // kk = 0
                    mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][1][m], v_b[1][1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][1], v_sfb[1][1]);   // kk = 1
                });
                sched_x4<4, 2, 1, 0>();
                sched_x4<4, 1, 1, 0>();
                __builtin_amdgcn_sched_barrier(0);

                // Scale panel copies, split over two tiles (one panel instruction touches 64 cache
                // lines, ~5x an A/B copy on the TA): odd t -> sfa, even t -> sfb of the panel
                // starting at t + 3 / t + 2.  With 3 panel buffers it overwrites the panel three
                // back (read long ago) and lands before the B_b that reads it.  A runtime branch,
                // so it sits on a region boundary.
                // The two branches end in different (empty) asm, so the compiler cannot merge
                // their tails into one copy that selects between the sfa / sfb layouts (that
                // selection goes through scratch, whose vmcnt wait drains every copy in flight).
                __builtin_amdgcn_sched_barrier(0);

                // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
                load_a(rd_a, number<0>{});
                read_sf(rd_sfa, rd_sfb, number<0>{});
                put_sf(rd_sfa, u_ssfa, v_sf_pend, sf_wr);      // even tile: sfa of the panel starting at tile + 2
                v_sf_pend = fetch_sfb(tile + 4);               //            load sfb of the panel starting at tile + 4
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][0][m], v_b[1][0][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);   // kk = 0
                    mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][1][m], v_b[1][1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][1], v_sfb[1][1]);   // kk = 1
                });
                sched_c11();
                __builtin_amdgcn_sched_barrier(0);

                // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
                load_a(rd_a, number<1>{});
                load_b(rd_b, number<1>{});
                read_sf(rd_sfa, rd_sfb, number<1>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][0][m], v_b[0][0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);   // kk = 0
                    mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][1][m], v_b[0][1][n], v_c[0][0][m * T::E_N + n], v_sfa[0][1], v_sfb[0][1]);   // kk = 1
                });
                sched_c00();
                __builtin_amdgcn_sched_barrier(0);
            });
            sf_rd = sf_wr;                                     // advances after an even tile
        }
        {   // odd tile
            const int tile_odd = tile + 1;
            const int tile = tile_odd;
            constexpr int stage = 1;
            constexpr int next_stage = 0;
            const int sf_wr = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // B_a: every wave is done reading stage(t) (A1 / B1 of tile t in C00), so
                // tile t+2 may be copied into it.
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C10(t) + A0, B0 of tile t+2 -> stage(t), one buffer_load per 4 MFMAs.
                prefetch_a(stage, tile + 2, 0);
                prefetch_b(stage, tile + 2, 0);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][0][m], v_b[0][0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);   // kk = 0
                    mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][1][m], v_b[0][1][n], v_c[1][0][m * T::E_N + n], v_sfa[1][1], v_sfb[0][1]);   // kk = 1
                });
                sched_x4<8, 0, 1, 0>();
                __builtin_amdgcn_sched_barrier(0);
                // B_b: tile t+1 (and its scale panel) landed; the 8 copies just issued may still fly.
                s_waitcnt_vmcnt(number<2 * T::a_buffer_load_insts>{});
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
                load_sfa(rd_sfa, sf_rd, 0);
                load_sfb(rd_sfb, sf_rd, 0);
                load_b(rd_b, number<0>{});
                prefetch_a(stage, tile + 2, 1);
                prefetch_b(stage, tile + 2, 1);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][0][m], v_b[1][0][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);   // kk = 0
                    mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][1][m], v_b[1][1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][1], v_sfb[1][1]);   // kk = 1
                });
                sched_x4<4, 2, 1, 0>();
                sched_x4<4, 1, 1, 0>();
                __builtin_amdgcn_sched_barrier(0);

                // Scale panel copies, split over two tiles (one panel instruction touches 64 cache
                // lines, ~5x an A/B copy on the TA): odd t -> sfa, even t -> sfb of the panel
                // starting at t + 3 / t + 2.  With 3 panel buffers it overwrites the panel three
                // back (read long ago) and lands before the B_b that reads it.  A runtime branch,
                // so it sits on a region boundary.
                // The two branches end in different (empty) asm, so the compiler cannot merge
                // their tails into one copy that selects between the sfa / sfb layouts (that
                // selection goes through scratch, whose vmcnt wait drains every copy in flight).
                // odd tile: sfa of the panel starting at tile + 3; past the last panel it only
                // overwrites the buffer of a consumed panel (harmless), so no branch.
                __builtin_amdgcn_sched_barrier(0);

                // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
                load_a(rd_a, number<0>{});
                read_sf(rd_sfa, rd_sfb, number<0>{});
                put_sf(rd_sfb, u_ssfb, v_sf_pend, sf_wr);      // odd tile: sfb of the panel starting at tile + 3
                v_sf_pend = fetch_sfa(tile + 3);               //           load sfa of the panel starting at tile + 3
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][0][m], v_b[1][0][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);   // kk = 0
                    mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][1][m], v_b[1][1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][1], v_sfb[1][1]);   // kk = 1
                });
                sched_c11();
                __builtin_amdgcn_sched_barrier(0);

                // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
                load_a(rd_a, number<1>{});
                load_b(rd_b, number<1>{});
                read_sf(rd_sfa, rd_sfb, number<1>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][0][m], v_b[0][0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);   // kk = 0
                    mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][1][m], v_b[0][1][n], v_c[0][0][m * T::E_N + n], v_sfa[0][1], v_sfb[0][1]);   // kk = 1
                });
                sched_c00();
                __builtin_amdgcn_sched_barrier(0);
            });
        }
    }
    // At most one more steady iteration (odd number of them).
#pragma clang loop unroll(disable)
    for (; tile + 2 < loops; ++tile) {
        const int stage = tile & 1;
        const int next_stage = stage ^ 1;
        const int sf_wr = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;
        views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
              [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
            // B_a: every wave is done reading stage(t) (A1 / B1 of tile t in C00), so
            // tile t+2 may be copied into it.
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // C10(t) + A0, B0 of tile t+2 -> stage(t), one buffer_load per 4 MFMAs.
            prefetch_a(stage, tile + 2, 0);
            prefetch_b(stage, tile + 2, 0);
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][0][m], v_b[0][0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);   // kk = 0
                mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][1][m], v_b[0][1][n], v_c[1][0][m * T::E_N + n], v_sfa[1][1], v_sfb[0][1]);   // kk = 1
            });
            sched_x4<8, 0, 1, 0>();
            __builtin_amdgcn_sched_barrier(0);
            // B_b: tile t+1 (and its scale panel) landed; the 8 copies just issued may still fly.
            s_waitcnt_vmcnt(number<2 * T::a_buffer_load_insts>{});
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
            load_sfa(rd_sfa, sf_rd, (tile + 1) & 1);
            load_sfb(rd_sfb, sf_rd, (tile + 1) & 1);
            load_b(rd_b, number<0>{});
            prefetch_a(stage, tile + 2, 1);
            prefetch_b(stage, tile + 2, 1);
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][0][m], v_b[1][0][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);   // kk = 0
                mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][1][m], v_b[1][1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][1], v_sfb[1][1]);   // kk = 1
            });
            sched_x4<4, 2, 1, 0>();
            sched_x4<4, 1, 1, 0>();
            __builtin_amdgcn_sched_barrier(0);

            // Scale panel copies, split over two tiles (one panel instruction touches 64 cache
            // lines, ~5x an A/B copy on the TA): odd t -> sfa, even t -> sfb of the panel
            // starting at t + 3 / t + 2.  With 3 panel buffers it overwrites the panel three
            // back (read long ago) and lands before the B_b that reads it.  A runtime branch,
            // so it sits on a region boundary.
            // The two branches end in different (empty) asm, so the compiler cannot merge
            // their tails into one copy that selects between the sfa / sfb layouts (that
            // selection goes through scratch, whose vmcnt wait drains every copy in flight).
            if (tile & 1) {
                put_sf(rd_sfb, u_ssfb, v_sf_pend, sf_wr);
                v_sf_pend = fetch_sfa(tile + 3);
                asm volatile("; panel copy: sfa" ::: "memory");
            } else {
                put_sf(rd_sfa, u_ssfa, v_sf_pend, sf_wr);
                v_sf_pend = fetch_sfb(tile + 4);
                asm volatile("; panel copy: sfb" ::: "memory");
            }
            __builtin_amdgcn_sched_barrier(0);

            // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
            load_a(rd_a, number<0>{});
            read_sf(rd_sfa, rd_sfb, number<0>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][0][m], v_b[1][0][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);   // kk = 0
                mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][1][m], v_b[1][1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][1], v_sfb[1][1]);   // kk = 1
            });
            sched_c11();
            __builtin_amdgcn_sched_barrier(0);

            // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
            load_a(rd_a, number<1>{});
            load_b(rd_b, number<1>{});
            read_sf(rd_sfa, rd_sfb, number<1>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][0][m], v_b[0][0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);   // kk = 0
                mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][1][m], v_b[0][1][n], v_c[0][0][m * T::E_N + n], v_sfa[0][1], v_sfb[0][1]);   // kk = 1
            });
            sched_c00();
            __builtin_amdgcn_sched_barrier(0);
        });
        sf_rd = (tile & 1) ? sf_rd : sf_wr;
    }

    // ===== Epilogue =====
    // 1. Second-to-last tile: C10, C01, C11 of it and C00 of the last tile; no more copies.
    if (tile + 1 < loops) {
        const int next_stage = (tile & 1) ^ 1;
        views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
              [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
            // C10(t).
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][0][m], v_b[0][0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);   // kk = 0
                mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][1][m], v_b[0][1][n], v_c[1][0][m * T::E_N + n], v_sfa[1][1], v_sfb[0][1]);   // kk = 1
            });
            sched_x4<8, 0, 0, 0>();
            __builtin_amdgcn_sched_barrier(0);
            // B_b: tile t+1 (and its scale panel) landed.
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // C01(t) + scales and B0 of tile t+1.
            load_sfa(rd_sfa, sf_rd, (tile + 1) & 1);
            load_sfb(rd_sfb, sf_rd, (tile + 1) & 1);
            load_b(rd_b, number<0>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][0][m], v_b[1][0][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);   // kk = 0
                mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][1][m], v_b[1][1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][1], v_sfb[1][1]);   // kk = 1
            });
            sched_x4<4, 2, 0, 0>();
            sched_x4<4, 1, 0, 0>();
            __builtin_amdgcn_sched_barrier(0);

            // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
            load_a(rd_a, number<0>{});
            read_sf(rd_sfa, rd_sfb, number<0>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][0][m], v_b[1][0][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);   // kk = 0
                mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][1][m], v_b[1][1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][1], v_sfb[1][1]);   // kk = 1
            });
            sched_c11();
            __builtin_amdgcn_sched_barrier(0);

            // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
            load_a(rd_a, number<1>{});
            load_b(rd_b, number<1>{});
            read_sf(rd_sfa, rd_sfb, number<1>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][0][m], v_b[0][0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);   // kk = 0
                mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][1][m], v_b[0][1][n], v_c[0][0][m * T::E_N + n], v_sfa[0][1], v_sfb[0][1]);   // kk = 1
            });
            sched_c00();
            __builtin_amdgcn_sched_barrier(0);
        });
        ++tile;
    }
    // 2. Last tile: C10, C01, C11.  Every LDS read of this output tile is done (after the
    //    barrier), so K tile 0 and scale panel 0 of the block's next output tile are copied
    //    into stage 0 / panel buffer 0 now, one copy per 4 MFMAs.  Without a next tile the
    //    copies use an empty buffer range (no memory traffic, they only zero stage 0, which is
    //    not read again) so the interleaving needs no branch.
    {
        const int next_id = tile_id + grid_blocks;
        const bool has_next = next_id < total_tiles;
        const int nid = has_next ? next_id : tile_id;
        const int n_batch = nid / tiles_per_batch;
        const int n_in_batch = nid % tiles_per_batch;
        const int n_row = (n_in_batch / num_tiles_n) * T::B_M;
        const int n_col = (n_in_batch % num_tiles_n) * T::B_N;
        auto ng_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + n_batch * kargs.stride_a_batch + n_row * kargs.stride_a, has_next ? 0xffffffffu : 0u);
        auto ng_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + n_batch * kargs.stride_b_batch + n_col * kargs.stride_b, has_next ? 0xffffffffu : 0u);
        auto ng_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + n_batch * kargs.stride_sfa_batch + n_row * kargs.stride_sfa, has_next ? static_cast<unsigned int>((kargs.m - n_row) * kargs.stride_sfa) : 0u);
        auto ng_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + n_batch * kargs.stride_sfb_batch + n_col * kargs.stride_sfb, has_next ? static_cast<unsigned int>((kargs.n - n_col) * kargs.stride_sfb) : 0u);
        prefetched = has_next;

        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        // C10(t) + A0, B0 of the next tile's K tile 0.
        async_load<T::VEC_A>(ng_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
        async_load<T::VEC_B>(ng_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][0][m], v_b[0][0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);   // kk = 0
            mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][1][m], v_b[0][1][n], v_c[1][0][m * T::E_N + n], v_sfa[1][1], v_sfb[0][1]);   // kk = 1
        });
        sched_x4<8, 0, 1, 0>();
        __builtin_amdgcn_sched_barrier(0);

        // C01(t) + A1, B1 of the next tile's K tile 0.
        async_load<T::VEC_A>(ng_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
        async_load<T::VEC_B>(ng_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][0][m], v_b[1][0][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);   // kk = 0
            mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][1][m], v_b[1][1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][1], v_sfb[1][1]);   // kk = 1
        });
        sched_x4<8, 0, 1, 0>();
        __builtin_amdgcn_sched_barrier(0);

        // C11(t) + scale panel 0 of the next tile.
        v_nsfa0 = load<T::VEC_GLOBAL_SCALE>(ng_sfa, u_gsfa, gsfa_offset(0));
        v_nsfb0 = load<T::VEC_GLOBAL_SCALE>(ng_sfb, u_gsfb, gsfb_offset(0));
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][0][m], v_b[1][0][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);   // kk = 0
            mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][1][m], v_b[1][1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][1], v_sfb[1][1]);   // kk = 1
        });
        sched_x4<2, 0, 1, 0>();
        sched_x4<6, 0, 0, 0>();
        __builtin_amdgcn_sched_barrier(0);
    }

    // 3. Write back C (bf16) through LDS, as opus_gemm_pipeline_a8w8_blockscale_mxscale_
    //    bpreshuffle_4wave_pin_agpr_gfx950 (PR 6696, Output::lds).  A wave owns only 16
    //    columns of a row per fragment (the other wave_n holds the neighbours), so direct
    //    fragment stores leave every 128-byte line partially written.  The whole 256 x 256
    //    tile is staged as one bf16 image in the idle A+B LDS, rows padded to B_N + 8
    //    elements (the 16 rows of a fragment then step by 16 bytes across the banks);
    //    each quadrant gets its own region, so staging needs no buffer reuse.  Per
    //    quadrant: stage -> sync -> 8 copies of 16 B per thread (one buffer_store = 4
    //    rows x 256 B), non-temporal (aux 2).
    // Staging uses only stage 1 (stage 0 receives the next tile's K tile 0): half of C
    // (128 rows x 256 columns) at a time, rows padded to B_N + 8 elements (528 B, as before),
    // which is exactly the 66 KB of stage 1.  Top half (Q00 / Q01), then bottom half.
    constexpr int c_lds_stride = T::B_N + 8;
    constexpr int VEC_OUT = 16 / sizeof(D_C);                                            // 8 bf16 = 16 B
    static_assert(T::HALF_B_M * c_lds_stride * sizeof(D_C) == stage_elems * sizeof(D_AB));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_ab + stage_elems));
    // Re-derive the thread position here (opaque) so the write-back offsets are computed now
    // instead of being kept live (and spilled) from the prologue.
    // (from the SGPR wave id and mbcnt, so thread_id_x() / v0 need not stay live)
    unsigned int lane_mask_e = ~0u;
    asm volatile("" : "+s"(lane_mask_e));
    const int lane_e = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lane_mask_e, __builtin_amdgcn_mbcnt_lo(lane_mask_e, 0u)));
    const int wave_e = wave_id;
    const int tid_e = wave_e * T::WARP_SIZE + lane_e;
    auto p_coord_c = opus::make_tuple(wave_e % T::T_M, lane_e % mma.grpn_c, wave_e / T::T_M, lane_e / mma.grpn_c);
    auto u_sc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(opus::number<c_lds_stride>{}, 1_I), p_coord_c);
    const int output_thread_id = tid_e;
    // bf16 C: LDS -> global, one half tile.  All LDS reads are issued before the stores: the
    // compiler does not move a ds_read above a buffer_store (it cannot tell them apart), so a
    // read / store loop would expose every read latency.
    constexpr int c_copies = T::HALF_B_M * T::B_N / (T::BLOCK_SIZE * VEC_OUT);   // 16
    auto copy_output_half = [&](int half_tile_m) {
        opus::vector_t<D_C, VEC_OUT> v[c_copies];
        opus::static_for<c_copies>([&](auto i) {
            const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
            v[i.value] = load<VEC_OUT>(s_c, (linear / T::B_N) * c_lds_stride + linear % T::B_N);
        });
        opus::static_for<c_copies>([&](auto i) {
            const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
            const int r = linear / T::B_N + half_tile_m * T::HALF_B_M, c = linear % T::B_N;
            store<VEC_OUT>(g_c, v[i.value], r * kargs.stride_c + c, 0, opus::number<2>{});
        });
    };
    // fp32 accumulators -> bf16 -> LDS (columns hn * 128 of the staged half).
    auto stage_quadrant = [&](const auto& acc, int half_tile_n) {
        typename MMA::vtype_c quad;
        opus::static_for<T::E_M * T::E_N>([&](auto f) {
            opus::set_slice(quad, acc[f.value], opus::number<f.value * MMA::mma_c_len>{}, opus::number<(f.value + 1) * MMA::mma_c_len>{});
        });
        store<T::VEC_C>(s_c, cast<D_C>(quad), u_sc + half_tile_n * T::HALF_B_N);
    };
    auto sync_lds = [&]() {
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    };
    // No vmcnt waits here: the next tile's K tile 0 copies are in flight.  Stage 1 was last
    // read before the barrier at the start of "2. Last tile".
    stage_quadrant(v_c[0][0], 0);
    stage_quadrant(v_c[0][1], 1);
    sync_lds();
    copy_output_half(0);
    sync_lds();
    stage_quadrant(v_c[1][0], 0);
    stage_quadrant(v_c[1][1], 1);
    sync_lds();
    copy_output_half(1);
    // Next tile's prologue copies overwrite the A+B LDS used for staging C: every wave must
    // have finished its staged-C reads.
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    }   // persistent tile loop
}
