#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "gemm_a8w8_mxfp8_scale_common.h"

namespace blockscale_generic {

using opus::operator""_I;

template<class T, int MRepeat, class Mem, class Offsets, class V>
__device__ inline void load_a_mrepeat_scale(
    Mem& mem, const Offsets& offsets, V& dst) {
    static_assert(T::VEC_A == 16 && T::E_M == 4 && T::E_K == 1);
    static_assert(MRepeat >= 0 && MRepeat < T::E_M);
    // One 16x16x128 MFMA A operand has two 16-byte K64 pieces.
    opus::static_for<2>([&](auto j) {
        constexpr int i = MRepeat * 2 + decltype(j)::value;
        const auto value = mem.template load<T::VEC_A>(offsets[i]);
        opus::set_slice(dst, value, opus::number<i * T::VEC_A>{}, opus::number<(i + 1) * T::VEC_A>{});
    });
}

template<class T, int Begin, int End, class Mem, class Offsets, class V>
__device__ inline void load_b_range_scale(
    Mem& mem, const Offsets& offsets, V& dst) {
    opus::static_for<End - Begin>([&](auto j) {
        constexpr int i = Begin + decltype(j)::value;
        auto value = mem.template load<T::VEC_B>(offsets[i]);
        opus::set_slice(dst, value, opus::number<i * T::VEC_B>{}, opus::number<(i + 1) * T::VEC_B>{});
    });
}

__device__ inline void sched_barrier_pairs_scale() {
    __builtin_amdgcn_sched_group_barrier(0x08, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x02, 2, 0);
    __builtin_amdgcn_sched_group_barrier(0x08, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x02, 2, 0);
}

template<class T, int HALF_TILE_M, int M_REPEAT, int N_REPEAT, class MMA>
__device__ inline auto mma_scale_one(
    MMA& mma,
    const typename opus::remove_cvref_t<MMA>::vtype_a& v_a,
    const typename opus::remove_cvref_t<MMA>::vtype_b& v_b,
    typename opus::remove_cvref_t<MMA>::MMA::vtype_c v_c,
    unsigned int v_sfa,
    unsigned int v_sfb)
    -> typename opus::remove_cvref_t<MMA>::MMA::vtype_c {
    (void)mma;

    using tiled_mma = opus::remove_cvref_t<MMA>;
    using base_mma = typename tiled_mma::MMA;

    constexpr int a_len = tiled_mma::mma_a_len;
    constexpr int b_len = tiled_mma::mma_b_len;
    constexpr int a_offset = M_REPEAT * a_len;
    constexpr int b_offset = N_REPEAT * b_len;
    // A 128-row half has four M repeats.  Each half owns one 4-byte SFA pack,
    // so op_sel remains 0..3 instead of incorrectly trying to address bytes
    // 4..7 in one dword.
    constexpr int scale_op_sel_a = M_REPEAT;
    static_assert(HALF_TILE_M == 0 || HALF_TILE_M == 1);
    static_assert(T::E_M == 4);

    auto s_a = opus::slice(v_a, opus::number<a_offset>{}, opus::number<a_offset + a_len>{});
    auto s_b = opus::slice(v_b, opus::number<b_offset>{}, opus::number<b_offset + b_len>{});
    return base_mma{}(s_a, s_b, v_c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<scale_op_sel_a>{}, opus::number<N_REPEAT>{});
}

template<class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_M / threads_m_per_block>{},
        opus::number<T::T_N>{},
        opus::number<threads_m_per_wave>{},
        opus::number<T::T_M>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,lane_id % threads_k}));
}

template<class T>
__device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;

    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::smem_m_rep / num_waves>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template<class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    constexpr int lanes_n = T::W_N; // 16: lane_id % 16 selects N within an N16 group.
    constexpr int lanes_k = T::WARP_SIZE / lanes_n; // 4: lane_id / 16 selects a K16 slice.
    constexpr int k_per_wave_load = lanes_k * T::VEC_B; // 4 * 16 = 64 K elements per N row.
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n); // 128 / (4 * 16) = 2.
    constexpr int k_repeats = T::B_K / k_per_wave_load; // 128 / 64 = 2 wave loads per N16 group.

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<n_repeats>{},
        opus::number<k_repeats>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_B>{});

    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
    constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
    constexpr int k_repeats = T::B_K / k_per_wave_load;

    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<n_repeats>{},
        opus::number<k_repeats>{},
        opus::number<T::VEC_B>{});
    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sb_block_shape,
        opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {

    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M / T::T_M>{},
        opus::number<T::E_K>{},
        opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
        opus::number<T::WARP_SIZE / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id % T::W_M;
    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, lane_id_m / T::T_M, lane_id / T::W_M}));
}

template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
    constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::T_N>{},
        opus::number<T::E_K>{},
        opus::number<k_vectors>{},
        opus::number<T::WARP_SIZE>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}

template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int stride_sfa) {
    // Logical [M, K/128] = [256, 1], stored as [K/128, M].
    // gsfa_offset selects the K column; contiguous M stays the final vector axis.
    constexpr auto gsfa_block_shape = opus::make_tuple(
        opus::number<T::B_K / T::GROUP_K>{}, // 1
        opus::number<T::B_M / T::HALF_B_M>{}, // 2
        opus::number<T::E_M>{}, // 4
        opus::number<T::T_M>{}, // 2
        opus::number<T::W_M>{}); // 16

    constexpr auto gsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_SCALE_A>(gsfa_block_shape,
        opus::unfold_x_stride(gsfa_block_dim, gsfa_block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(gsfa_block_dim, opus::tuple{((lane_id & (T::W_M - 1)) / T::E_M) / T::T_M, (lane_id & (T::W_M - 1)) % T::E_M, ((lane_id & (T::W_M - 1)) / T::E_M) % T::T_M}));
}

template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto ssfa_block_shape = opus::make_tuple(
        opus::number<T::SCALE_PANEL_K_TILES>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::B_M / T::HALF_B_M>{},
        opus::number<T::E_M>{});

    constexpr auto ssfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int wave_id = wave_id_n * T::T_M + wave_id_m;
    const int k_column = wave_id * (T::WARP_SIZE / T::W_M) + (static_cast<unsigned int>(lane_id) / T::W_M);
    const int group = lane_id & 15;
    const int owner_wave = group / 4;
    const int call = group % 4;
    return opus::make_layout<T::VEC_SF>(
        ssfa_block_shape,
        opus::unfold_x_stride(ssfa_block_dim, ssfa_block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
        opus::unfold_p_coord(ssfa_block_dim, opus::tuple{k_column, owner_wave % T::T_M, call, owner_wave / T::T_M}));
}

template<class T, int Vec>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {

    constexpr auto rsfa_block_shape = opus::make_tuple(
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<Vec>{});

    constexpr auto rsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<Vec>(
        rsfa_block_shape,
        opus::unfold_x_stride(rsfa_block_dim, rsfa_block_shape, opus::tuple{opus::number<(T::B_M / T::HALF_B_M) * T::E_M>{}, 1_I}),
        opus::unfold_p_coord(rsfa_block_dim, opus::tuple{wave_id_m, lane_id & 15}));
}

template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id) {
    // Global SFB: one byte per K128 column. The producer's N128 half
    // is selected by the scalar offset on the global load.
    constexpr auto gsfb_block_shape = opus::make_tuple(opus::number<T::SCALE_PANEL_K_TILES>{}, 1_I);
    constexpr auto gsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
        gsfb_block_shape,
        opus::unfold_x_stride(gsfb_block_dim, gsfb_block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(gsfb_block_dim, opus::tuple{lane_id}));
}

template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
    // LDS SFB [K slot, N128 half, four replicated bytes].
    constexpr auto ssfb_block_shape = opus::make_tuple(
        opus::number<T::SCALE_PANEL_K_TILES>{},
        opus::number<T::SCALE_N_HALVES>{}, // 2
        opus::number<T::VEC_SF>{});

    constexpr auto ssfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_SF>(
        ssfb_block_shape,
        opus::unfold_x_stride(ssfb_block_dim, ssfb_block_shape, opus::tuple{opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
        opus::unfold_p_coord(ssfb_block_dim, opus::tuple{lane_id, wave_id}));
}

template<class T, int Vec>
__device__ inline constexpr auto make_layout_rsfb_scale() {
    static_assert(Vec == T::VEC_SF || Vec == T::VEC_SF_PAIR);
    // All lanes read the same one or two adjacent N128 scale dwords.
    // The current K slot and N128 half are selected by the read offset.
    constexpr auto rsfb_block_shape = opus::make_tuple(opus::number<Vec>{});

    constexpr auto rsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<Vec>(
        rsfb_block_shape,
        opus::unfold_x_stride(rsfb_block_dim, rsfb_block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(rsfb_block_dim, opus::tuple{}));
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU) void gemm_a8w8_mxfp8_scale_kernel(opus_gemm_scale_kargs kargs) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = std::conditional_t<T::OUTPUT_BF16, opus::bf16_t, opus::fp32_t>;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned int;
    static_assert(T::VEC_SF * sizeof(D_SF) == sizeof(D_SF_PACK));
    static_assert(T::VEC_SF_PAIR == 2 * T::VEC_SF);

    // N varies along grid X and M along grid Y. The public byte-count
    // bound limits either tile dimension to less than 16384.
    int block_m = block_id_y();
    int block_n = block_id_x();
    // Transpose low tile-coordinate bits only for complete square groups.
    // Other legal shapes retain the original direct two-dimensional mapping.
    if (((kargs.m | kargs.n) & 511) == 0) {
        const int old_m = block_m;
        block_m = (block_m & ~1) | (block_n & 1);
        block_n = (block_n & ~1) | (old_m & 1);
    }
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;

    const int batch_id = block_id_z();
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;

    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * kargs.stride_sfb);

    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);

    auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, kargs.stride_sfa);
    auto u_ssfa = make_layout_ssfa_scale<T>(lane_id, wave_id_m, wave_id_n);
    auto u_rsfa = make_layout_rsfa_scale<T, T::VEC_SF>(lane_id, wave_id_m);
    auto u_rsfa_pair = make_layout_rsfa_scale<T, T::VEC_SF_PAIR>(lane_id, wave_id_m);
    auto u_gsfb = make_layout_gsfb_scale<T>(lane_id);
    auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
    auto u_rsfb = make_layout_rsfb_scale<T, T::VEC_SF>();
    auto u_rsfb_pair = make_layout_rsfb_scale<T, T::VEC_SF_PAIR>();

    constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
    constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
    constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A) + smem_b_elem * 4 * sizeof(D_B);
    alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
    constexpr int c_lds_pitch = T::B_N + 8;

    const bool produces_a = wave_id_n == 0;
    auto g_matrix = produces_a ? g_a : g_b;
    const int matrix_vector_offset = produces_a ? (wave_id_m * T::HALF_B_M + (lane_id / 8) * 2) * kargs.stride_a + (lane_id % 8) * 16 : wave_id_m * T::HALF_B_N * kargs.stride_b + lane_id * 16;
    // Both public launch paths enforce stride_a == stride_b == k.
    const int matrix_pair_stride = 16 * kargs.stride_a;
    const int matrix_odd_stride = produces_a ? kargs.stride_a : 1024;
    const int matrix_k_stride = produces_a ? T::B_K : T::B_K * 16;
    constexpr int matrix_pitch = T::smem_linear_wave + T::smem_padding;
    auto* matrix_lds_base = (produces_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem;
    static_assert(smem_a_elem == smem_b_elem);
    // Cache immutable address deltas in VGPRs; only K advances in SGPRs.
    // Four requests share one m0 base, with IOFFSET 0/1056/2112/3168.
    // Subtract that same immediate from VOFFSET to preserve the input address.
    // Only the lane-dependent addresses are cached in ordinary VGPRs.
    // Later row groups reuse them with nonnegative, wave-uniform SOFFSETs.
    opus::vector_t<int, 4> matrix_addresses;
    opus::static_for<4>([&](auto issue_i) {
        constexpr int issue = decltype(issue_i)::value;
        constexpr int local = issue % 4;
        constexpr int immediate = local * matrix_pitch;
        int address = matrix_vector_offset + (issue / 2) * matrix_pair_stride
            + (issue % 2) * matrix_odd_stride - immediate;
        asm volatile("" : "+v"(address));
        matrix_addresses[issue] = address;
    });
    auto prefetch_matrix_issue = [&](auto issue_i, int matrix_stage, int k_tile) {
        constexpr int issue = decltype(issue_i)::value;
        constexpr int local = issue % 4;
        constexpr int immediate = local * matrix_pitch;
        auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem
            + (issue / 4) * 4 * matrix_pitch;
        g_matrix.template async_load<16>(
            reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
            matrix_addresses[issue % 4],
            __builtin_amdgcn_readfirstlane(k_tile * matrix_k_stride
                + (issue / 4) * 2 * matrix_pair_stride),
            opus::number<immediate>{}, opus::number<0>{});
    };

    opus::i32x4_t matrix_raw_rsrc;
    __builtin_memcpy(&matrix_raw_rsrc, &g_matrix.cached_rsrc, sizeof(matrix_raw_rsrc));
    auto prepare_matrix_group = [&](auto group_i, int matrix_stage) {
        constexpr int group = decltype(group_i)::value;
        auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem
            + group * 4 * matrix_pitch;
        const unsigned int address = static_cast<unsigned int>(reinterpret_cast<__UINTPTR_TYPE__>(dst));
        asm volatile("s_mov_b32 m0, %0" : : "s"(address) : "m0", "memory");
    };
    auto issue_matrix_prepared = [&matrix_addresses, &matrix_raw_rsrc, &matrix_pair_stride](auto issue_i, int scalar_offset) {
        constexpr int issue = decltype(issue_i)::value;
        constexpr int immediate = (issue % 4) * matrix_pitch;
        // The preceding prepare and the load both declare m0 to preserve order.
        // Every matrix publication retains explicit VMEM/LGKM retirement.
        asm volatile("buffer_load_dwordx4 %0, %1, %2 offen offset:%3 lds"
            : : "v"(matrix_addresses[issue % 4]), "s"(matrix_raw_rsrc),
                "s"(__builtin_amdgcn_readfirstlane(scalar_offset
                    + (issue / 4) * 2 * matrix_pair_stride)), "n"(immediate) : "m0", "memory");
    };

    alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
    alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    typename decltype(mma)::vtype_a v_a[2];
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_b v_b_second;
    using AccFragment = typename decltype(mma)::MMA::vtype_c;
    static_assert(decltype(mma)::mma_c_len == 4);

    __attribute__((amdgpu_pin_agpr(0))) AccFragment c00_0 = {};
    __attribute__((amdgpu_pin_agpr(4))) AccFragment c00_1 = {};
    __attribute__((amdgpu_pin_agpr(8))) AccFragment c00_2 = {};
    __attribute__((amdgpu_pin_agpr(12))) AccFragment c00_3 = {};
    __attribute__((amdgpu_pin_agpr(16))) AccFragment c00_4 = {};
    __attribute__((amdgpu_pin_agpr(20))) AccFragment c00_5 = {};
    __attribute__((amdgpu_pin_agpr(24))) AccFragment c00_6 = {};
    __attribute__((amdgpu_pin_agpr(28))) AccFragment c00_7 = {};
    __attribute__((amdgpu_pin_agpr(32))) AccFragment c00_8 = {};
    __attribute__((amdgpu_pin_agpr(36))) AccFragment c00_9 = {};
    __attribute__((amdgpu_pin_agpr(40))) AccFragment c00_10 = {};
    __attribute__((amdgpu_pin_agpr(44))) AccFragment c00_11 = {};
    __attribute__((amdgpu_pin_agpr(48))) AccFragment c00_12 = {};
    __attribute__((amdgpu_pin_agpr(52))) AccFragment c00_13 = {};
    __attribute__((amdgpu_pin_agpr(56))) AccFragment c00_14 = {};
    __attribute__((amdgpu_pin_agpr(60))) AccFragment c00_15 = {};
    __attribute__((amdgpu_pin_agpr(64))) AccFragment c01_0 = {};
    __attribute__((amdgpu_pin_agpr(68))) AccFragment c01_1 = {};
    __attribute__((amdgpu_pin_agpr(72))) AccFragment c01_2 = {};
    __attribute__((amdgpu_pin_agpr(76))) AccFragment c01_3 = {};
    __attribute__((amdgpu_pin_agpr(80))) AccFragment c01_4 = {};
    __attribute__((amdgpu_pin_agpr(84))) AccFragment c01_5 = {};
    __attribute__((amdgpu_pin_agpr(88))) AccFragment c01_6 = {};
    __attribute__((amdgpu_pin_agpr(92))) AccFragment c01_7 = {};
    __attribute__((amdgpu_pin_agpr(96))) AccFragment c01_8 = {};
    __attribute__((amdgpu_pin_agpr(100))) AccFragment c01_9 = {};
    __attribute__((amdgpu_pin_agpr(104))) AccFragment c01_10 = {};
    __attribute__((amdgpu_pin_agpr(108))) AccFragment c01_11 = {};
    __attribute__((amdgpu_pin_agpr(112))) AccFragment c01_12 = {};
    __attribute__((amdgpu_pin_agpr(116))) AccFragment c01_13 = {};
    __attribute__((amdgpu_pin_agpr(120))) AccFragment c01_14 = {};
    __attribute__((amdgpu_pin_agpr(124))) AccFragment c01_15 = {};
    __attribute__((amdgpu_pin_agpr(128))) AccFragment c10_0 = {};
    __attribute__((amdgpu_pin_agpr(132))) AccFragment c10_1 = {};
    __attribute__((amdgpu_pin_agpr(136))) AccFragment c10_2 = {};
    __attribute__((amdgpu_pin_agpr(140))) AccFragment c10_3 = {};
    __attribute__((amdgpu_pin_agpr(144))) AccFragment c10_4 = {};
    __attribute__((amdgpu_pin_agpr(148))) AccFragment c10_5 = {};
    __attribute__((amdgpu_pin_agpr(152))) AccFragment c10_6 = {};
    __attribute__((amdgpu_pin_agpr(156))) AccFragment c10_7 = {};
    __attribute__((amdgpu_pin_agpr(160))) AccFragment c10_8 = {};
    __attribute__((amdgpu_pin_agpr(164))) AccFragment c10_9 = {};
    __attribute__((amdgpu_pin_agpr(168))) AccFragment c10_10 = {};
    __attribute__((amdgpu_pin_agpr(172))) AccFragment c10_11 = {};
    __attribute__((amdgpu_pin_agpr(176))) AccFragment c10_12 = {};
    __attribute__((amdgpu_pin_agpr(180))) AccFragment c10_13 = {};
    __attribute__((amdgpu_pin_agpr(184))) AccFragment c10_14 = {};
    __attribute__((amdgpu_pin_agpr(188))) AccFragment c10_15 = {};
    __attribute__((amdgpu_pin_agpr(192))) AccFragment c11_0 = {};
    __attribute__((amdgpu_pin_agpr(196))) AccFragment c11_1 = {};
    __attribute__((amdgpu_pin_agpr(200))) AccFragment c11_2 = {};
    __attribute__((amdgpu_pin_agpr(204))) AccFragment c11_3 = {};
    __attribute__((amdgpu_pin_agpr(208))) AccFragment c11_4 = {};
    __attribute__((amdgpu_pin_agpr(212))) AccFragment c11_5 = {};
    __attribute__((amdgpu_pin_agpr(216))) AccFragment c11_6 = {};
    __attribute__((amdgpu_pin_agpr(220))) AccFragment c11_7 = {};
    __attribute__((amdgpu_pin_agpr(224))) AccFragment c11_8 = {};
    __attribute__((amdgpu_pin_agpr(228))) AccFragment c11_9 = {};
    __attribute__((amdgpu_pin_agpr(232))) AccFragment c11_10 = {};
    __attribute__((amdgpu_pin_agpr(236))) AccFragment c11_11 = {};
    __attribute__((amdgpu_pin_agpr(240))) AccFragment c11_12 = {};
    __attribute__((amdgpu_pin_agpr(244))) AccFragment c11_13 = {};
    __attribute__((amdgpu_pin_agpr(248))) AccFragment c11_14 = {};
    __attribute__((amdgpu_pin_agpr(252))) AccFragment c11_15 = {};

    D_SF_PACK v_sfa[2];
    D_SF_PACK v_sfb[T::SCALE_N_HALVES];
    opus::vector_t<D_SF_PACK, 2> v_sfb_next;
    opus::vector_t<D_SF_PACK, 2> v_sfa_next;
    auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K; };
    auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K * 16; };
    auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
    // Runtime K controls every matrix iteration. The fixed panel size
    // is a cache capacity; panels are refreshed for arbitrarily long K.
    // Positive K is a multiple of B_K under the public contract.
    const int loops = static_cast<unsigned int>(kargs.k) / T::B_K;
    static_assert(T::SCALE_PANEL_K_TILES == 64);
    constexpr int panel_mask = T::SCALE_PANEL_K_TILES - 1;

    constexpr int sfa_columns_per_pass = T::BLOCK_SIZE / T::W_M;
    constexpr int sfa_row_pitch = (T::B_M / T::HALF_B_M) * T::E_M;
    const int sfa_first_column = thread_id_x() / T::W_M;
    auto gsfa_offset = [&](int panel_begin, int pass) {
        const int column = panel_begin + sfa_first_column + pass * sfa_columns_per_pass;
        const int valid_column = column < loops ? column : loops - 1;
        return valid_column * kargs.stride_sfa;
    };
    auto ssfa_offset = [&](int pass, int word) {
        // Each pass advances 16 K slots; each word advances four M rows.
        return pass * sfa_columns_per_pass * T::SFA_PANEL_PITCH + word * 4 * sfa_row_pitch;
    };
    auto rsfa_offset = [&](int k_tile, int half_tile_m) {
        return (k_tile & panel_mask) * T::SFA_PANEL_PITCH + half_tile_m * T::E_M;
    };
    auto gsfb_offset = [&](int panel_begin) {
        const int column = panel_begin + lane_id;
        const int valid_column = column < loops ? column : loops - 1;
        return valid_column - lane_id;
    };
    auto rsfb_offset = [&](int k_tile, int half_tile_n) {
        return (k_tile & panel_mask) * T::SCALE_N_HALVES * T::VEC_SF + half_tile_n * T::VEC_SF;
    };

    opus::vector_t<D_SF, 16> panel_raw_a[4];
    auto load_sfa_panel = [&](int panel_begin) {
        opus::static_for<4>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            panel_raw_a[pass] = load<16>(g_sfa, u_gsfa + gsfa_offset(panel_begin, pass));
        });
        __builtin_amdgcn_sched_barrier(0);
    };

    auto publish_sfa_panel = [&]() {
        // Raw vectors were issued together before this transpose. Each
        // quad owns four M repeats for one K column and one M128/wave-M
        // pair. Transpose each 4x4 byte block in registers, then publish
        // the final dword directly, with no intermediate LDS transpose.
        const int group = lane_id & 15;
        const int call = group % 4;
        const unsigned int select_pair = (call & 1) ? 0x03070105u : 0x06020400u;
        const unsigned int select_quad = (call & 2) ? 0x03020706u : 0x05040100u;
        opus::static_for<4>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const auto raw = panel_raw_a[pass];
            const auto words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, raw);
            opus::static_for<4>([&](auto word_i) {
                constexpr int word = decltype(word_i)::value;
                const D_SF_PACK x = words[word];
                const D_SF_PACK adjacent = opus::mov_dpp(x, opus::number<0xb1>{});
                const D_SF_PACK pair = __builtin_amdgcn_perm(adjacent, x, select_pair);
                const D_SF_PACK opposite = opus::mov_dpp(pair, opus::number<0x4e>{});
                const D_SF_PACK packed = __builtin_amdgcn_perm(opposite, pair, select_quad);
                store<T::VEC_SF>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfa + ssfa_offset(pass, word));
            });
        });
    };

    auto load_sfa_dword = [&](int k_tile, int half_tile_m) {
        return __builtin_bit_cast(D_SF_PACK, load<T::VEC_SF>(s_sfa, u_rsfa + rsfa_offset(k_tile, half_tile_m)));
    };

    auto load_sfb_dword = [&](int k_tile, int half_tile_n) {
        // The prologue already duplicated the compact byte across the
        // dword. Every lane reads the same address, preserving op_sel.
        return __builtin_bit_cast(D_SF_PACK, load<T::VEC_SF>(s_sfb, u_rsfb + rsfb_offset(k_tile, half_tile_n)));
    };

    auto refill_scale_panel = [&](int next_tile) {
        if ((next_tile & panel_mask) == 0) {
            __builtin_amdgcn_s_setprio(0);
            // Current scales are resident in registers. Retire every reader
            // of the previous panel before overwriting the same allocation.
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            load_sfa_panel(next_tile);
            D_SF_PACK raw_b = 0;
            if (wave_id < T::SCALE_N_HALVES) {
                const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(next_tile), wave_id * kargs.stride_sfb);
                raw_b = static_cast<D_SF_PACK>(raw[0]);
            }
            s_waitcnt_vmcnt(0_I);
            publish_sfa_panel();
            if (wave_id < T::SCALE_N_HALVES) {
                const D_SF_PACK packed = raw_b * 0x01010101u;
                store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed),
                    u_ssfb);
            }
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            __builtin_amdgcn_s_setprio(1);
        }
    };

    // ===== Prologue =====
    async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
    async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
    __builtin_amdgcn_sched_barrier(0);

    // Prologue: preload the first bounded scale panel before the first barrier.
    load_sfa_panel(0);
    // Initialize C while the four SFA requests progress.
    asm volatile("" : : "a"(c00_0));
    asm volatile("" : : "a"(c00_1));
    asm volatile("" : : "a"(c00_2));
    asm volatile("" : : "a"(c00_3));
    asm volatile("" : : "a"(c00_4));
    asm volatile("" : : "a"(c00_5));
    asm volatile("" : : "a"(c00_6));
    asm volatile("" : : "a"(c00_7));
    asm volatile("" : : "a"(c00_8));
    asm volatile("" : : "a"(c00_9));
    asm volatile("" : : "a"(c00_10));
    asm volatile("" : : "a"(c00_11));
    asm volatile("" : : "a"(c00_12));
    asm volatile("" : : "a"(c00_13));
    asm volatile("" : : "a"(c00_14));
    asm volatile("" : : "a"(c00_15));
    asm volatile("" : : "a"(c01_0));
    asm volatile("" : : "a"(c01_1));
    asm volatile("" : : "a"(c01_2));
    asm volatile("" : : "a"(c01_3));
    asm volatile("" : : "a"(c01_4));
    asm volatile("" : : "a"(c01_5));
    asm volatile("" : : "a"(c01_6));
    asm volatile("" : : "a"(c01_7));
    asm volatile("" : : "a"(c01_8));
    asm volatile("" : : "a"(c01_9));
    asm volatile("" : : "a"(c01_10));
    asm volatile("" : : "a"(c01_11));
    asm volatile("" : : "a"(c01_12));
    asm volatile("" : : "a"(c01_13));
    asm volatile("" : : "a"(c01_14));
    asm volatile("" : : "a"(c01_15));

    D_SF_PACK panel_sfb_raw = 0;
    if (wave_id < T::SCALE_N_HALVES) {
        const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(0), wave_id * kargs.stride_sfb);
        panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
    }
    async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
    async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));

    // Finish C initialization while the initial B requests progress.
    asm volatile("" : : "a"(c10_0));
    asm volatile("" : : "a"(c10_1));
    asm volatile("" : : "a"(c10_2));
    asm volatile("" : : "a"(c10_3));
    asm volatile("" : : "a"(c10_4));
    asm volatile("" : : "a"(c10_5));
    asm volatile("" : : "a"(c10_6));
    asm volatile("" : : "a"(c10_7));
    asm volatile("" : : "a"(c10_8));
    asm volatile("" : : "a"(c10_9));
    asm volatile("" : : "a"(c10_10));
    asm volatile("" : : "a"(c10_11));
    asm volatile("" : : "a"(c10_12));
    asm volatile("" : : "a"(c10_13));
    asm volatile("" : : "a"(c10_14));
    asm volatile("" : : "a"(c10_15));
    asm volatile("" : : "a"(c11_0));
    asm volatile("" : : "a"(c11_1));
    asm volatile("" : : "a"(c11_2));
    asm volatile("" : : "a"(c11_3));
    asm volatile("" : : "a"(c11_4));
    asm volatile("" : : "a"(c11_5));
    asm volatile("" : : "a"(c11_6));
    asm volatile("" : : "a"(c11_7));
    asm volatile("" : : "a"(c11_8));
    asm volatile("" : : "a"(c11_9));
    asm volatile("" : : "a"(c11_10));
    asm volatile("" : : "a"(c11_11));
    asm volatile("" : : "a"(c11_12));
    asm volatile("" : : "a"(c11_13));
    asm volatile("" : : "a"(c11_14));
    asm volatile("" : : "a"(c11_15));

    // B matrix requests can progress while the scale transpose publishes.
    publish_sfa_panel();
    s_waitcnt_vmcnt(0_I);

    if (wave_id < T::SCALE_N_HALVES) {
        const D_SF_PACK packed = panel_sfb_raw * 0x01010101u;
        store<T::VEC_SF>(s_sfb,
            __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed),
            u_ssfb);
    }

    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    int stage = 0;
    int tile = 0;

    // A single K128 block has no K1 producer.
    if (loops > 1) {
        // Both matrices preload K1. Main-loop publications release the matrix
        // stage for t+2 prefetches; the penultimate block issues no future request.
        // The main-loop producer mapping also covers this initial matrix stage.
        opus::static_for<8>([&](auto initial_issue) {
            using Issue = opus::number<decltype(initial_issue)::value + 0>;
            prefetch_matrix_issue(Issue{}, 1, 1);
        });

        // This guarded K1 block has B data ready by the first
        // main-loop VMEM drain; scales require no further global traffic.
        // The main-loop producer mapping also covers this initial matrix stage.
        opus::static_for<8>([&](auto initial_issue) {
            using Issue = opus::number<decltype(initial_issue)::value + 8>;
            prefetch_matrix_issue(Issue{}, 1, 1);
        });
        __builtin_amdgcn_sched_barrier(0);

    }

    // Seed all four matrix halves and scale dwords for K0. The main
    // and penultimate blocks prefetch each next-K SFA/SFB pair in one b64
    // read, then install its halves after the last current-K consumers.
    v_sfa[0] = load_sfa_dword(0, 0);
    v_sfa[1] = load_sfa_dword(0, 1);
    v_sfb[0] = load_sfb_dword(0, 0);
    v_sfb[1] = load_sfb_dword(0, 1);
    v_b = load<T::VEC_B>(s_b, u_rb + sb_offset(stage, 0));
    v_a[0] = load<T::VEC_A>(s_a, u_ra + sa_offset(stage, 0));
    // Seed every current-tile matrix half. Subsequent A1/B1 values roll
    // into dead operand slices near the end of the previous K iteration.
    v_a[1] = load<T::VEC_A>(s_a, u_ra + sa_offset(stage, 1));
    v_b_second = load<T::VEC_B>(s_b, u_rb + sb_offset(stage, 1));
    __builtin_amdgcn_sched_barrier(0);

    // ===== Main loop: paired K128 blocks; K512 per full unroll-2 group =====
    // Prefetch while two later K128 blocks exist; peel the penultimate block.
    // Future matrix addresses always use the complete runtime K index.
    // Keep priority 1 across main iterations; scale-panel refill uses 0/1.
    __builtin_amdgcn_s_setprio(1);
    auto run_main_k128 = [&](int k_tile) __attribute__((always_inline)) {
        refill_scale_panel(k_tile + 1);
        const int next_stage = stage ^ 1;
        const int future_tile = k_tile + 2;
        int future_matrix_offset = future_tile * matrix_k_stride;
        asm volatile("" : "+s"(future_matrix_offset));

        // The preceding K iteration prefetched both B scale halves.
        __builtin_amdgcn_sched_barrier(0);
        // Keep stage address expressions before the scheduling boundary;
        // removing them moves the final unrolled stage multiply between MFMAs.
        const auto u_sa_next_0 = u_sa + sa_offset(next_stage, 0);
        const auto u_sa_next_1 = u_sa + sa_offset(next_stage, 1);
        __builtin_amdgcn_sched_barrier(0);

        const auto rb0_next_offsets =
            opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0));

        // All LDS reads here are compiler-visible intrinsics. Let LLVM's
        // waitcnt pass wait at actual operand consumers. The A/B data stage
        // handoff still has vmcnt(0)/lgkmcnt(0)/barrier below.

        // A half 0 x B half 0 -> C[0][0] (128x128 wave quadrant).
        // After publication at MFMA5, all waves issue one t+2 request
        // after MFMA7,9,...,37. Wave-uniform resources select the A or B
        // producer without role branches inside the K loop.
        c00_0 = mma_scale_one<T, 0, 0, 0>(
            mma, v_a[0], v_b, c00_0, v_sfa[0], v_sfb[0]);
        c00_1 = mma_scale_one<T, 0, 0, 1>(
            mma, v_a[0], v_b, c00_1, v_sfa[0], v_sfb[0]);
        sched_barrier_pairs_scale();

        // Prefetch published scales ahead of the operand roll.
        const auto u_rsfa_next = u_rsfa_pair + rsfa_offset(k_tile + 1, 0);
        v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SF_PAIR>(s_sfa, u_rsfa_next));
        __builtin_amdgcn_sched_barrier(0);

        c00_2 = mma_scale_one<T, 0, 0, 2>(
            mma, v_a[0], v_b, c00_2, v_sfa[0], v_sfb[0]);
        c00_3 = mma_scale_one<T, 0, 0, 3>(
            mma, v_a[0], v_b, c00_3, v_sfa[0], v_sfb[0]);
        sched_barrier_pairs_scale();

        c00_4 = mma_scale_one<T, 0, 1, 0>(
            mma, v_a[0], v_b, c00_4, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        // MFMA5: publish t+1 and release stage t between independent MFMAs.
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        c00_5 = mma_scale_one<T, 0, 1, 1>(
            mma, v_a[0], v_b, c00_5, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        prepare_matrix_group(opus::number<0>{}, stage);
        __builtin_amdgcn_sched_barrier(0);

        c00_6 = mma_scale_one<T, 0, 1, 2>(
            mma, v_a[0], v_b, c00_6, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<0>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c00_7 = mma_scale_one<T, 0, 1, 3>(
            mma, v_a[0], v_b, c00_7, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        c00_8 = mma_scale_one<T, 0, 2, 0>(
            mma, v_a[0], v_b, c00_8, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<1>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c00_9 = mma_scale_one<T, 0, 2, 1>(
            mma, v_a[0], v_b, c00_9, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        c00_10 = mma_scale_one<T, 0, 2, 2>(
            mma, v_a[0], v_b, c00_10, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<2>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c00_11 = mma_scale_one<T, 0, 2, 3>(
            mma, v_a[0], v_b, c00_11, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        c00_12 = mma_scale_one<T, 0, 3, 0>(
            mma, v_a[0], v_b, c00_12, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<3>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c00_13 = mma_scale_one<T, 0, 3, 1>(
            mma, v_a[0], v_b, c00_13, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        prepare_matrix_group(opus::number<1>{}, stage);
        __builtin_amdgcn_sched_barrier(0);

        c00_14 = mma_scale_one<T, 0, 3, 2>(
            mma, v_a[0], v_b, c00_14, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<4>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c00_15 = mma_scale_one<T, 0, 3, 3>(
            mma, v_a[0], v_b, c00_15, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        // A half 1 x B half 0 -> C[1][0] (128x128 wave quadrant).
        c10_0 = mma_scale_one<T, 1, 0, 0>(
            mma, v_a[1], v_b, c10_0, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<5>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c10_1 = mma_scale_one<T, 1, 0, 1>(
            mma, v_a[1], v_b, c10_1, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        c10_2 = mma_scale_one<T, 1, 0, 2>(
            mma, v_a[1], v_b, c10_2, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<6>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c10_3 = mma_scale_one<T, 1, 0, 3>(
            mma, v_a[1], v_b, c10_3, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        // Prefetch published scales ahead of the operand roll.
        const auto u_rsfb_next = u_rsfb_pair + rsfb_offset(k_tile + 1, 0);
        v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SF_PAIR>(s_sfb, u_rsfb_next));
        __builtin_amdgcn_sched_barrier(0);

        c10_4 = mma_scale_one<T, 1, 1, 0>(
            mma, v_a[1], v_b, c10_4, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<7>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c10_5 = mma_scale_one<T, 1, 1, 1>(
            mma, v_a[1], v_b, c10_5, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        prepare_matrix_group(opus::number<2>{}, stage);
        __builtin_amdgcn_sched_barrier(0);

        c10_6 = mma_scale_one<T, 1, 1, 2>(
            mma, v_a[1], v_b, c10_6, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<8>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c10_7 = mma_scale_one<T, 1, 1, 3>(
            mma, v_a[1], v_b, c10_7, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        c10_8 = mma_scale_one<T, 1, 2, 0>(
            mma, v_a[1], v_b, c10_8, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<9>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c10_9 = mma_scale_one<T, 1, 2, 1>(
            mma, v_a[1], v_b, c10_9, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        c10_10 = mma_scale_one<T, 1, 2, 2>(
            mma, v_a[1], v_b, c10_10, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<10>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c10_11 = mma_scale_one<T, 1, 2, 3>(
            mma, v_a[1], v_b, c10_11, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        c10_12 = mma_scale_one<T, 1, 3, 0>(
            mma, v_a[1], v_b, c10_12, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<11>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c10_13 = mma_scale_one<T, 1, 3, 1>(
            mma, v_a[1], v_b, c10_13, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        prepare_matrix_group(opus::number<3>{}, stage);
        __builtin_amdgcn_sched_barrier(0);

        // Read next A0/M0 while its current-K operand is still live.
        // The temporary is installed only after the last MFMA36 consumer.
        const auto a0_m0_prefetch_offsets =
            opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0));
        auto a0_m0_next_0 = s_a.template load<16>(a0_m0_prefetch_offsets[0]);
        auto a0_m0_next_1 = s_a.template load<16>(a0_m0_prefetch_offsets[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c10_14 = mma_scale_one<T, 1, 3, 2>(
            mma, v_a[1], v_b, c10_14, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<12>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c10_15 = mma_scale_one<T, 1, 3, 3>(
            mma, v_a[1], v_b, c10_15, v_sfa[1], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        // Spread the dead B0 operand reads after MFMA32.
        load_b_range_scale<T, 0, 2>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        const auto& v_b_n1 = v_b_second;

        // A half 0 x B half 1 -> C[0][1] (128x128 wave quadrant).
        c01_0 = mma_scale_one<T, 0, 0, 0>(
            mma, v_a[0], v_b_n1, c01_0, v_sfa[0], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<13>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c01_1 = mma_scale_one<T, 0, 0, 1>(
            mma, v_a[0], v_b_n1, c01_1, v_sfa[0], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        // Spread the dead B0 operand reads after MFMA34.
        load_b_range_scale<T, 2, 4>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c01_2 = mma_scale_one<T, 0, 0, 2>(
            mma, v_a[0], v_b_n1, c01_2, v_sfa[0], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<14>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c01_3 = mma_scale_one<T, 0, 0, 3>(
            mma, v_a[0], v_b_n1, c01_3, v_sfa[0], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        // Spread the dead B0 operand reads after MFMA36.
        load_b_range_scale<T, 4, 6>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // M0's four c01 consumers are complete, and the early publication
        // barrier has published next_stage. Roll only its two K64 pieces.
        const auto ra0_next_offsets =
            opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0));
        const auto ra1_next_offsets =
            opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1));
        const auto rb1_next_offsets =
            opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1));
        opus::set_slice(v_a[0], a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
        opus::set_slice(v_a[0], a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);

        // Continue C01 while next-k_tile operands roll into dead registers.
        c01_4 = mma_scale_one<T, 0, 1, 0>(
            mma, v_a[0], v_b_n1, c01_4, v_sfa[0], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        issue_matrix_prepared(opus::number<15>{}, future_matrix_offset);
        __builtin_amdgcn_sched_group_barrier(0x20, 1, 0);
        __builtin_amdgcn_sched_barrier(0);
        c01_5 = mma_scale_one<T, 0, 1, 1>(
            mma, v_a[0], v_b_n1, c01_5, v_sfa[0], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        // Spread the dead B0 operand reads after MFMA38.
        load_b_range_scale<T, 6, 8>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // Install the low dword of the SFB pair prefetched after MFMA20.
        // Keep current SFB1 through C11. B0's four operand pairs have now
        // rolled to the next K block at MFMA32/34/36/38.
        v_sfb[0] = v_sfb_next[0];
        __builtin_amdgcn_sched_barrier(0);

        c01_6 = mma_scale_one<T, 0, 1, 2>(
            mma, v_a[0], v_b_n1, c01_6, v_sfa[0], v_sfb[1]);
        c01_7 = mma_scale_one<T, 0, 1, 3>(
            mma, v_a[0], v_b_n1, c01_7, v_sfa[0], v_sfb[1]);
        sched_barrier_pairs_scale();

        // M1 is dead after c01_7; M2/M3 still use their current-k_tile slices.
        load_a_mrepeat_scale<T, 1>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);

        c01_8 = mma_scale_one<T, 0, 2, 0>(
            mma, v_a[0], v_b_n1, c01_8, v_sfa[0], v_sfb[1]);
        c01_9 = mma_scale_one<T, 0, 2, 1>(
            mma, v_a[0], v_b_n1, c01_9, v_sfa[0], v_sfb[1]);
        sched_barrier_pairs_scale();

        c01_10 = mma_scale_one<T, 0, 2, 2>(
            mma, v_a[0], v_b_n1, c01_10, v_sfa[0], v_sfb[1]);
        c01_11 = mma_scale_one<T, 0, 2, 3>(
            mma, v_a[0], v_b_n1, c01_11, v_sfa[0], v_sfb[1]);
        sched_barrier_pairs_scale();

        // M2 is dead after c01_11. Keep the old SFA0 pack through M3.
        load_a_mrepeat_scale<T, 2>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);

        c01_12 = mma_scale_one<T, 0, 3, 0>(
            mma, v_a[0], v_b_n1, c01_12, v_sfa[0], v_sfb[1]);
        c01_13 = mma_scale_one<T, 0, 3, 1>(
            mma, v_a[0], v_b_n1, c01_13, v_sfa[0], v_sfb[1]);
        sched_barrier_pairs_scale();

        c01_14 = mma_scale_one<T, 0, 3, 2>(
            mma, v_a[0], v_b_n1, c01_14, v_sfa[0], v_sfb[1]);
        c01_15 = mma_scale_one<T, 0, 3, 3>(
            mma, v_a[0], v_b_n1, c01_15, v_sfa[0], v_sfb[1]);
        sched_barrier_pairs_scale();

        // C01 has consumed current SFA0. Install the low dword of the
        // SFA pair prefetched after MFMA2; keep current SFA1 through C11.
        __builtin_amdgcn_sched_barrier(0);
        v_sfa[0] = v_sfa_next[0];
        // M3 is dead after c01_15. The earlier three slices already hold
        // k_tile t+1, so these last two K64 pieces complete the A0 roll.
        load_a_mrepeat_scale<T, 3>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // C11: reorder independent accumulators and roll each operand after its last use.
        c11_0 = mma_scale_one<T, 1, 0, 0>(
            mma, v_a[1], v_b_n1, c11_0, v_sfa[1], v_sfb[1]);
        c11_1 = mma_scale_one<T, 1, 0, 1>(
            mma, v_a[1], v_b_n1, c11_1, v_sfa[1], v_sfb[1]);
        sched_barrier_pairs_scale();

        c11_2 = mma_scale_one<T, 1, 0, 2>(
            mma, v_a[1], v_b_n1, c11_2, v_sfa[1], v_sfb[1]);
        c11_3 = mma_scale_one<T, 1, 0, 3>(
            mma, v_a[1], v_b_n1, c11_3, v_sfa[1], v_sfb[1]);
        sched_barrier_pairs_scale();
        // MFMA52: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_a_mrepeat_scale<T, 0>(s_a, ra1_next_offsets, v_a[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_4 = mma_scale_one<T, 1, 1, 0>(
            mma, v_a[1], v_b_n1, c11_4, v_sfa[1], v_sfb[1]);
        c11_5 = mma_scale_one<T, 1, 1, 1>(
            mma, v_a[1], v_b_n1, c11_5, v_sfa[1], v_sfb[1]);
        sched_barrier_pairs_scale();

        // MFMA54: keep the next operand separate until its current value dies.
        auto b1_n3_early_0 = s_b.template load<16>(rb1_next_offsets[6]);
        auto b1_n3_early_1 = s_b.template load<16>(rb1_next_offsets[7]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_6 = mma_scale_one<T, 1, 1, 2>(
            mma, v_a[1], v_b_n1, c11_6, v_sfa[1], v_sfb[1]);
        c11_7 = mma_scale_one<T, 1, 1, 3>(
            mma, v_a[1], v_b_n1, c11_7, v_sfa[1], v_sfb[1]);
        sched_barrier_pairs_scale();
        // MFMA56: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_a_mrepeat_scale<T, 1>(s_a, ra1_next_offsets, v_a[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // MFMA56: preload the final operand slices into independent registers.
        auto a1_m3_next_0 = s_a.template load<16>(ra1_next_offsets[6]);
        auto a1_m3_next_1 = s_a.template load<16>(ra1_next_offsets[7]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_8 = mma_scale_one<T, 1, 2, 0>(
            mma, v_a[1], v_b_n1, c11_8, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        c11_12 = mma_scale_one<T, 1, 3, 0>(
            mma, v_a[1], v_b_n1, c11_12, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA58: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_b_range_scale<T, 0, 2>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_9 = mma_scale_one<T, 1, 2, 1>(
            mma, v_a[1], v_b_n1, c11_9, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        c11_13 = mma_scale_one<T, 1, 3, 1>(
            mma, v_a[1], v_b_n1, c11_13, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA60: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_b_range_scale<T, 2, 4>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_10 = mma_scale_one<T, 1, 2, 2>(
            mma, v_a[1], v_b_n1, c11_10, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        c11_14 = mma_scale_one<T, 1, 3, 2>(
            mma, v_a[1], v_b_n1, c11_14, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA62: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_b_range_scale<T, 4, 6>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_11 = mma_scale_one<T, 1, 2, 3>(
            mma, v_a[1], v_b_n1, c11_11, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA63: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_a_mrepeat_scale<T, 2>(s_a, ra1_next_offsets, v_a[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_15 = mma_scale_one<T, 1, 3, 3>(
            mma, v_a[1], v_b_n1, c11_15, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA64: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        opus::set_slice(v_a[1], a1_m3_next_0, opus::number<96>{}, opus::number<112>{});
        opus::set_slice(v_a[1], a1_m3_next_1, opus::number<112>{}, opus::number<128>{});
        opus::set_slice(v_b_second, b1_n3_early_0, opus::number<96>{}, opus::number<112>{});
        opus::set_slice(v_b_second, b1_n3_early_1, opus::number<112>{}, opus::number<128>{});
        __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
        __builtin_amdgcn_sched_barrier(0);

        v_sfb[1] = v_sfb_next[1];
        v_sfa[1] = v_sfa_next[1];
        stage = next_stage;
    };

#pragma unroll 2
    for (tile = 0; tile + 3 < loops; tile += 2) {
        run_main_k128(tile);
        run_main_k128(tile + 1);
    }

    // One unmatched main K128 block precedes the original drain.
    if (tile + 2 < loops) {
        run_main_k128(tile);
        ++tile;
    }

    // ===== Epilogue: drain the remaining K blocks and finish output stores =====
    // Penultimate runtime K128 block: roll the final block with no
    // unused future matrix request. K128 skips this segment entirely.
    if (loops > 1) {
        refill_scale_panel(tile + 1);
        const int next_stage = stage ^ 1;

        // The preceding K iteration prefetched both B scale halves.
        __builtin_amdgcn_sched_barrier(0);
        // Keep stage address expressions before the scheduling boundary;
        // removing them moves the final unrolled stage multiply between MFMAs.
        const auto u_sa_next_0 = u_sa + sa_offset(next_stage, 0);
        const auto u_sa_next_1 = u_sa + sa_offset(next_stage, 1);
        __builtin_amdgcn_sched_barrier(0);

        const auto rb0_next_offsets =
            opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0));

        // All LDS reads here are compiler-visible intrinsics. Let LLVM's
        // waitcnt pass wait at actual operand consumers. The A/B data stage
        // handoff still has vmcnt(0)/lgkmcnt(0)/barrier below.
        __builtin_amdgcn_s_setprio(1);

        // A half 0 x B half 0 -> C[0][0] (128x128 wave quadrant).
        // Publish the final block at MFMA5, then finish its predecessor
        // and roll its successor without any further global matrix requests.
        c00_0 = mma_scale_one<T, 0, 0, 0>(
            mma, v_a[0], v_b, c00_0, v_sfa[0], v_sfb[0]);
        c00_1 = mma_scale_one<T, 0, 0, 1>(
            mma, v_a[0], v_b, c00_1, v_sfa[0], v_sfb[0]);
        sched_barrier_pairs_scale();

        // Prefetch published scales ahead of the operand roll.
        const auto u_rsfa_next = u_rsfa_pair + rsfa_offset(tile + 1, 0);
        v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SF_PAIR>(s_sfa, u_rsfa_next));
        __builtin_amdgcn_sched_barrier(0);

        c00_2 = mma_scale_one<T, 0, 0, 2>(
            mma, v_a[0], v_b, c00_2, v_sfa[0], v_sfb[0]);
        c00_3 = mma_scale_one<T, 0, 0, 3>(
            mma, v_a[0], v_b, c00_3, v_sfa[0], v_sfb[0]);
        sched_barrier_pairs_scale();

        c00_4 = mma_scale_one<T, 0, 1, 0>(
            mma, v_a[0], v_b, c00_4, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        // MFMA5: publish t+1 and release stage t between independent MFMAs.
        __builtin_amdgcn_s_setprio(0);
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        __builtin_amdgcn_s_setprio(1);

        c00_5 = mma_scale_one<T, 0, 1, 1>(
            mma, v_a[0], v_b, c00_5, v_sfa[0], v_sfb[0]);
        __builtin_amdgcn_sched_barrier(0);

        c00_6 = mma_scale_one<T, 0, 1, 2>(
            mma, v_a[0], v_b, c00_6, v_sfa[0], v_sfb[0]);
        c00_7 = mma_scale_one<T, 0, 1, 3>(
            mma, v_a[0], v_b, c00_7, v_sfa[0], v_sfb[0]);
        sched_barrier_pairs_scale();

        c00_8 = mma_scale_one<T, 0, 2, 0>(
            mma, v_a[0], v_b, c00_8, v_sfa[0], v_sfb[0]);
        c00_9 = mma_scale_one<T, 0, 2, 1>(
            mma, v_a[0], v_b, c00_9, v_sfa[0], v_sfb[0]);
        sched_barrier_pairs_scale();

        c00_10 = mma_scale_one<T, 0, 2, 2>(
            mma, v_a[0], v_b, c00_10, v_sfa[0], v_sfb[0]);
        c00_11 = mma_scale_one<T, 0, 2, 3>(
            mma, v_a[0], v_b, c00_11, v_sfa[0], v_sfb[0]);
        sched_barrier_pairs_scale();

        c00_12 = mma_scale_one<T, 0, 3, 0>(
            mma, v_a[0], v_b, c00_12, v_sfa[0], v_sfb[0]);
        c00_13 = mma_scale_one<T, 0, 3, 1>(
            mma, v_a[0], v_b, c00_13, v_sfa[0], v_sfb[0]);
        sched_barrier_pairs_scale();

        c00_14 = mma_scale_one<T, 0, 3, 2>(
            mma, v_a[0], v_b, c00_14, v_sfa[0], v_sfb[0]);
        c00_15 = mma_scale_one<T, 0, 3, 3>(
            mma, v_a[0], v_b, c00_15, v_sfa[0], v_sfb[0]);
        sched_barrier_pairs_scale();

        // A half 1 x B half 0 -> C[1][0] (128x128 wave quadrant).
        c10_0 = mma_scale_one<T, 1, 0, 0>(
            mma, v_a[1], v_b, c10_0, v_sfa[1], v_sfb[0]);
        c10_1 = mma_scale_one<T, 1, 0, 1>(
            mma, v_a[1], v_b, c10_1, v_sfa[1], v_sfb[0]);
        sched_barrier_pairs_scale();

        c10_2 = mma_scale_one<T, 1, 0, 2>(
            mma, v_a[1], v_b, c10_2, v_sfa[1], v_sfb[0]);
        c10_3 = mma_scale_one<T, 1, 0, 3>(
            mma, v_a[1], v_b, c10_3, v_sfa[1], v_sfb[0]);
        sched_barrier_pairs_scale();

        // Prefetch published scales ahead of the operand roll.
        const auto u_rsfb_next = u_rsfb_pair + rsfb_offset(tile + 1, 0);
        v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SF_PAIR>(s_sfb, u_rsfb_next));
        __builtin_amdgcn_sched_barrier(0);

        c10_4 = mma_scale_one<T, 1, 1, 0>(
            mma, v_a[1], v_b, c10_4, v_sfa[1], v_sfb[0]);
        c10_5 = mma_scale_one<T, 1, 1, 1>(
            mma, v_a[1], v_b, c10_5, v_sfa[1], v_sfb[0]);
        sched_barrier_pairs_scale();

        c10_6 = mma_scale_one<T, 1, 1, 2>(
            mma, v_a[1], v_b, c10_6, v_sfa[1], v_sfb[0]);
        c10_7 = mma_scale_one<T, 1, 1, 3>(
            mma, v_a[1], v_b, c10_7, v_sfa[1], v_sfb[0]);
        sched_barrier_pairs_scale();

        c10_8 = mma_scale_one<T, 1, 2, 0>(
            mma, v_a[1], v_b, c10_8, v_sfa[1], v_sfb[0]);
        c10_9 = mma_scale_one<T, 1, 2, 1>(
            mma, v_a[1], v_b, c10_9, v_sfa[1], v_sfb[0]);
        sched_barrier_pairs_scale();

        c10_10 = mma_scale_one<T, 1, 2, 2>(
            mma, v_a[1], v_b, c10_10, v_sfa[1], v_sfb[0]);
        c10_11 = mma_scale_one<T, 1, 2, 3>(
            mma, v_a[1], v_b, c10_11, v_sfa[1], v_sfb[0]);
        sched_barrier_pairs_scale();

        c10_12 = mma_scale_one<T, 1, 3, 0>(
            mma, v_a[1], v_b, c10_12, v_sfa[1], v_sfb[0]);
        c10_13 = mma_scale_one<T, 1, 3, 1>(
            mma, v_a[1], v_b, c10_13, v_sfa[1], v_sfb[0]);
        sched_barrier_pairs_scale();

        // Read next A0/M0 while its current-K operand is still live.
        // The temporary is installed only after the last MFMA36 consumer.
        const auto a0_m0_prefetch_offsets =
            opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0));
        auto a0_m0_next_0 = s_a.template load<16>(a0_m0_prefetch_offsets[0]);
        auto a0_m0_next_1 = s_a.template load<16>(a0_m0_prefetch_offsets[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c10_14 = mma_scale_one<T, 1, 3, 2>(
            mma, v_a[1], v_b, c10_14, v_sfa[1], v_sfb[0]);
        c10_15 = mma_scale_one<T, 1, 3, 3>(
            mma, v_a[1], v_b, c10_15, v_sfa[1], v_sfb[0]);
        sched_barrier_pairs_scale();

        // Spread the dead B0 operand reads after MFMA32.
        load_b_range_scale<T, 0, 2>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        const auto& v_b_n1 = v_b_second;

        // A half 0 x B half 1 -> C[0][1] (128x128 wave quadrant).
        c01_0 = mma_scale_one<T, 0, 0, 0>(
            mma, v_a[0], v_b_n1, c01_0, v_sfa[0], v_sfb[1]);
        c01_1 = mma_scale_one<T, 0, 0, 1>(
            mma, v_a[0], v_b_n1, c01_1, v_sfa[0], v_sfb[1]);
        sched_barrier_pairs_scale();

        // Spread the dead B0 operand reads after MFMA34.
        load_b_range_scale<T, 2, 4>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c01_2 = mma_scale_one<T, 0, 0, 2>(
            mma, v_a[0], v_b_n1, c01_2, v_sfa[0], v_sfb[1]);
        c01_3 = mma_scale_one<T, 0, 0, 3>(
            mma, v_a[0], v_b_n1, c01_3, v_sfa[0], v_sfb[1]);
        sched_barrier_pairs_scale();

        // Spread the dead B0 operand reads after MFMA36.
        load_b_range_scale<T, 4, 6>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // M0's four c01 consumers are complete, and the early publication
        // barrier has published next_stage. Roll only its two K64 pieces.
        const auto ra0_next_offsets =
            opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0));
        const auto ra1_next_offsets =
            opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1));
        const auto rb1_next_offsets =
            opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1));
        opus::set_slice(v_a[0], a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
        opus::set_slice(v_a[0], a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);

        __builtin_amdgcn_s_setprio(1);

        // Continue C01 while next-tile operands roll into dead registers.
        c01_4 = mma_scale_one<T, 0, 1, 0>(
            mma, v_a[0], v_b_n1, c01_4, v_sfa[0], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        c01_5 = mma_scale_one<T, 0, 1, 1>(
            mma, v_a[0], v_b_n1, c01_5, v_sfa[0], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        // Spread the dead B0 operand reads after MFMA38.
        load_b_range_scale<T, 6, 8>(s_b, rb0_next_offsets, v_b);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // Install the low dword of the SFB pair prefetched after MFMA20.
        // Keep current SFB1 through C11. B0's four operand pairs have now
        // rolled to the next K block at MFMA32/34/36/38.
        v_sfb[0] = v_sfb_next[0];
        __builtin_amdgcn_sched_barrier(0);

        c01_6 = mma_scale_one<T, 0, 1, 2>(
            mma, v_a[0], v_b_n1, c01_6, v_sfa[0], v_sfb[1]);
        c01_7 = mma_scale_one<T, 0, 1, 3>(
            mma, v_a[0], v_b_n1, c01_7, v_sfa[0], v_sfb[1]);
        sched_barrier_pairs_scale();

        // M1 is dead after c01_7; M2/M3 still use their current-tile slices.
        load_a_mrepeat_scale<T, 1>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);

        c01_8 = mma_scale_one<T, 0, 2, 0>(
            mma, v_a[0], v_b_n1, c01_8, v_sfa[0], v_sfb[1]);
        c01_9 = mma_scale_one<T, 0, 2, 1>(
            mma, v_a[0], v_b_n1, c01_9, v_sfa[0], v_sfb[1]);
        sched_barrier_pairs_scale();

        c01_10 = mma_scale_one<T, 0, 2, 2>(
            mma, v_a[0], v_b_n1, c01_10, v_sfa[0], v_sfb[1]);
        c01_11 = mma_scale_one<T, 0, 2, 3>(
            mma, v_a[0], v_b_n1, c01_11, v_sfa[0], v_sfb[1]);
        sched_barrier_pairs_scale();

        // M2 is dead after c01_11. Keep the old SFA0 pack through M3.
        load_a_mrepeat_scale<T, 2>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);

        c01_12 = mma_scale_one<T, 0, 3, 0>(
            mma, v_a[0], v_b_n1, c01_12, v_sfa[0], v_sfb[1]);
        c01_13 = mma_scale_one<T, 0, 3, 1>(
            mma, v_a[0], v_b_n1, c01_13, v_sfa[0], v_sfb[1]);
        sched_barrier_pairs_scale();

        c01_14 = mma_scale_one<T, 0, 3, 2>(
            mma, v_a[0], v_b_n1, c01_14, v_sfa[0], v_sfb[1]);
        c01_15 = mma_scale_one<T, 0, 3, 3>(
            mma, v_a[0], v_b_n1, c01_15, v_sfa[0], v_sfb[1]);
        sched_barrier_pairs_scale();

        // C01 has consumed current SFA0. Install the low dword of the
        // SFA pair prefetched after MFMA2; keep current SFA1 through C11.
        __builtin_amdgcn_sched_barrier(0);
        v_sfa[0] = v_sfa_next[0];
        // M3 is dead after c01_15. The earlier three slices already hold
        // tile t+1, so these last two K64 pieces complete the A0 roll.
        load_a_mrepeat_scale<T, 3>(s_a, ra0_next_offsets, v_a[0]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // C11: reorder independent accumulators and roll each operand after its last use.
        c11_0 = mma_scale_one<T, 1, 0, 0>(
            mma, v_a[1], v_b_n1, c11_0, v_sfa[1], v_sfb[1]);
        c11_1 = mma_scale_one<T, 1, 0, 1>(
            mma, v_a[1], v_b_n1, c11_1, v_sfa[1], v_sfb[1]);
        sched_barrier_pairs_scale();

        c11_2 = mma_scale_one<T, 1, 0, 2>(
            mma, v_a[1], v_b_n1, c11_2, v_sfa[1], v_sfb[1]);
        c11_3 = mma_scale_one<T, 1, 0, 3>(
            mma, v_a[1], v_b_n1, c11_3, v_sfa[1], v_sfb[1]);
        sched_barrier_pairs_scale();
        // MFMA52: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_a_mrepeat_scale<T, 0>(s_a, ra1_next_offsets, v_a[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_4 = mma_scale_one<T, 1, 1, 0>(
            mma, v_a[1], v_b_n1, c11_4, v_sfa[1], v_sfb[1]);
        c11_5 = mma_scale_one<T, 1, 1, 1>(
            mma, v_a[1], v_b_n1, c11_5, v_sfa[1], v_sfb[1]);
        sched_barrier_pairs_scale();

        // MFMA54: keep the next operand separate until its current value dies.
        auto b1_n3_early_0 = s_b.template load<16>(rb1_next_offsets[6]);
        auto b1_n3_early_1 = s_b.template load<16>(rb1_next_offsets[7]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_6 = mma_scale_one<T, 1, 1, 2>(
            mma, v_a[1], v_b_n1, c11_6, v_sfa[1], v_sfb[1]);
        c11_7 = mma_scale_one<T, 1, 1, 3>(
            mma, v_a[1], v_b_n1, c11_7, v_sfa[1], v_sfb[1]);
        sched_barrier_pairs_scale();
        // MFMA56: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_a_mrepeat_scale<T, 1>(s_a, ra1_next_offsets, v_a[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        // MFMA56: preload the final operand slices into independent registers.
        auto a1_m3_next_0 = s_a.template load<16>(ra1_next_offsets[6]);
        auto a1_m3_next_1 = s_a.template load<16>(ra1_next_offsets[7]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_8 = mma_scale_one<T, 1, 2, 0>(
            mma, v_a[1], v_b_n1, c11_8, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        c11_12 = mma_scale_one<T, 1, 3, 0>(
            mma, v_a[1], v_b_n1, c11_12, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA58: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_b_range_scale<T, 0, 2>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_9 = mma_scale_one<T, 1, 2, 1>(
            mma, v_a[1], v_b_n1, c11_9, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        c11_13 = mma_scale_one<T, 1, 3, 1>(
            mma, v_a[1], v_b_n1, c11_13, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA60: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_b_range_scale<T, 2, 4>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_10 = mma_scale_one<T, 1, 2, 2>(
            mma, v_a[1], v_b_n1, c11_10, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);

        c11_14 = mma_scale_one<T, 1, 3, 2>(
            mma, v_a[1], v_b_n1, c11_14, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA62: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_b_range_scale<T, 4, 6>(s_b, rb1_next_offsets, v_b_second);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_11 = mma_scale_one<T, 1, 2, 3>(
            mma, v_a[1], v_b_n1, c11_11, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA63: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        load_a_mrepeat_scale<T, 2>(s_a, ra1_next_offsets, v_a[1]);
        __builtin_amdgcn_sched_group_barrier(0x100, 2, 0);
        __builtin_amdgcn_sched_barrier(0);

        c11_15 = mma_scale_one<T, 1, 3, 3>(
            mma, v_a[1], v_b_n1, c11_15, v_sfa[1], v_sfb[1]);
        __builtin_amdgcn_sched_barrier(0);
        // MFMA64: these operand slices have no remaining current-K consumers.
        __builtin_amdgcn_sched_barrier(0);
        opus::set_slice(v_a[1], a1_m3_next_0, opus::number<96>{}, opus::number<112>{});
        opus::set_slice(v_a[1], a1_m3_next_1, opus::number<112>{}, opus::number<128>{});
        opus::set_slice(v_b_second, b1_n3_early_0, opus::number<96>{}, opus::number<112>{});
        opus::set_slice(v_b_second, b1_n3_early_1, opus::number<112>{}, opus::number<128>{});
        __builtin_amdgcn_sched_group_barrier(0x100, 4, 0);
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(0);
        v_sfb[1] = v_sfb_next[1];
        v_sfa[1] = v_sfa_next[1];
        stage = next_stage;
    }

    // Consume the final resident tile and stage completed BF16 rows in LDS.
    if constexpr (T::OUTPUT_BF16) {
        s_waitcnt_vmcnt(0_I);
    }
    if constexpr (!T::OUTPUT_BF16) {
        v_a[1] = load<T::VEC_A>(s_a, u_ra + sa_offset(stage, 1));
    }

    if constexpr (!T::OUTPUT_BF16) s_waitcnt_lgkmcnt(0_I);
    if constexpr (!T::OUTPUT_BF16) {
        v_sfb[1] = load_sfb_dword(loops - 1, 1);
    }

    __builtin_amdgcn_s_setprio(1);
    auto p_coord_c = opus::make_tuple(
        wave_id_m, lane_id % mma.grpn_c,
        wave_id_n, lane_id / mma.grpn_c);
    auto u_gc = partition_layout_c<T::VEC_C>(
        mma, opus::make_tuple(T::OUTPUT_BF16 ? c_lds_pitch : kargs.stride_c, 1_I), p_coord_c);

    auto c_offset = [&](int half_tile_m, int half_tile_n) {
        return half_tile_m * T::HALF_B_M * (T::OUTPUT_BF16 ? c_lds_pitch : kargs.stride_c) +
               half_tile_n * T::HALF_B_N;
    };

    // Publish each completed 128x128 quadrant after MFMA20/36/52/64.
    // Eight coalesced 16-byte copies per thread cover one quadrant. All
    // four quadrants belong to this workgroup's single 256x256 output tile.
    auto copy_output_quarter = [&](int half_m, int half_n, int copy_index) {
        const int linear = thread_id_x() * 8 + copy_index * T::BLOCK_SIZE * 8;
        const int output_row = linear / T::HALF_B_N + half_m * T::HALF_B_M;
        const int output_col = linear % T::HALF_B_N + half_n * T::HALF_B_N;
        const auto value = s_c.template load<8>(output_row * c_lds_pitch + output_col);
        g_c.template store<8>(value, output_row * kargs.stride_c + output_col,
                             0, opus::number<2>{});
    };

    const auto gc_offsets = opus::layout_to_offsets<T::VEC_C>(u_gc);
    // Output type is selected at compile time inside these helpers.
    auto stage_output_row = [&](auto index_i, int half_m, int half_n,
                                const AccFragment& c0, const AccFragment& c1,
                                const AccFragment& c2, const AccFragment& c3) {
        if constexpr (T::OUTPUT_BF16) {
            constexpr int index = decltype(index_i)::value;
            constexpr int n_repeat_stride = T::T_N * T::W_N;
            // The native C layout supplies one row base; N repeats are uniform.
            const int row_offset = gc_offsets[index] + c_offset(half_m, half_n);
            const auto packed0 = cast<D_C>(c0);
            const auto packed1 = cast<D_C>(c1);
            const auto packed2 = cast<D_C>(c2);
            const auto packed3 = cast<D_C>(c3);
            static_assert(T::VEC_C * sizeof(D_C) == 8);
            static_assert(n_repeat_stride * sizeof(D_C) == 64);
            const unsigned int address = static_cast<unsigned int>(
                reinterpret_cast<__UINTPTR_TYPE__>(s_c.ptr) + row_offset * sizeof(D_C));
            asm volatile("ds_write2_b64 %0, %1, %2 offset1:8"
                         : : "v"(address), "v"(packed0), "v"(packed1) : "memory");
            asm volatile("ds_write2_b64 %0, %1, %2 offset0:16 offset1:24"
                         : : "v"(address), "v"(packed2), "v"(packed3) : "memory");
        }
    };
    auto store_output_quadrant = [&](int half_m, int half_n, const auto&... fragments) {
        if constexpr (!T::OUTPUT_BF16) {
            static_assert(sizeof...(fragments) == 16);
            const int soff = c_offset(half_m, half_n);
            int index = 0;
            (g_c.template store<T::VEC_C>(fragments, gc_offsets[index++], soff,
                                         opus::number<2>{}), ...);
        }
    };
    auto publish_output_quarter = [&](auto half_m_i, auto half_n_i) {
        if constexpr (T::OUTPUT_BF16) {
            constexpr int half_m = decltype(half_m_i)::value;
            constexpr int half_n = decltype(half_n_i)::value;
            __builtin_amdgcn_sched_barrier(0);
            // C10 can publish while the independent C00 global stores finish.
            if constexpr (!(half_m == 1 && half_n == 0)) {
                s_waitcnt_vmcnt(0_I);
            }
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
            opus::static_for<8>([&](auto copy_i) {
                copy_output_quarter(half_m, half_n, decltype(copy_i)::value);
            });
        }
    };

    c00_0 = mma_scale_one<T, 0, 0, 0>(
        mma, v_a[0], v_b, c00_0, v_sfa[0], v_sfb[0]);
    c00_1 = mma_scale_one<T, 0, 0, 1>(
        mma, v_a[0], v_b, c00_1, v_sfa[0], v_sfb[0]);
    sched_barrier_pairs_scale();

    c00_2 = mma_scale_one<T, 0, 0, 2>(
        mma, v_a[0], v_b, c00_2, v_sfa[0], v_sfb[0]);
    c00_3 = mma_scale_one<T, 0, 0, 3>(
        mma, v_a[0], v_b, c00_3, v_sfa[0], v_sfb[0]);
    sched_barrier_pairs_scale();

    c00_4 = mma_scale_one<T, 0, 1, 0>(
        mma, v_a[0], v_b, c00_4, v_sfa[0], v_sfb[0]);
    c00_5 = mma_scale_one<T, 0, 1, 1>(
        mma, v_a[0], v_b, c00_5, v_sfa[0], v_sfb[0]);
    sched_barrier_pairs_scale();

    c00_6 = mma_scale_one<T, 0, 1, 2>(
        mma, v_a[0], v_b, c00_6, v_sfa[0], v_sfb[0]);
    c00_7 = mma_scale_one<T, 0, 1, 3>(
        mma, v_a[0], v_b, c00_7, v_sfa[0], v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    if constexpr (T::OUTPUT_BF16) {
        // Retire all final matrix readers before the first output LDS store.
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }
    stage_output_row(opus::number<0>{}, 0, 0, c00_0, c00_1, c00_2, c00_3);

    c00_8 = mma_scale_one<T, 0, 2, 0>(
        mma, v_a[0], v_b, c00_8, v_sfa[0], v_sfb[0]);
    c00_9 = mma_scale_one<T, 0, 2, 1>(
        mma, v_a[0], v_b, c00_9, v_sfa[0], v_sfb[0]);
    sched_barrier_pairs_scale();

    c00_10 = mma_scale_one<T, 0, 2, 2>(
        mma, v_a[0], v_b, c00_10, v_sfa[0], v_sfb[0]);
    c00_11 = mma_scale_one<T, 0, 2, 3>(
        mma, v_a[0], v_b, c00_11, v_sfa[0], v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<4>{}, 0, 0, c00_4, c00_5, c00_6, c00_7);

    c00_12 = mma_scale_one<T, 0, 3, 0>(
        mma, v_a[0], v_b, c00_12, v_sfa[0], v_sfb[0]);
    c00_13 = mma_scale_one<T, 0, 3, 1>(
        mma, v_a[0], v_b, c00_13, v_sfa[0], v_sfb[0]);
    sched_barrier_pairs_scale();

    c00_14 = mma_scale_one<T, 0, 3, 2>(
        mma, v_a[0], v_b, c00_14, v_sfa[0], v_sfb[0]);
    c00_15 = mma_scale_one<T, 0, 3, 3>(
        mma, v_a[0], v_b, c00_15, v_sfa[0], v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<8>{}, 0, 0, c00_8, c00_9, c00_10, c00_11);

    c10_0 = mma_scale_one<T, 1, 0, 0>(
        mma, v_a[1], v_b, c10_0, v_sfa[1], v_sfb[0]);
    c10_1 = mma_scale_one<T, 1, 0, 1>(
        mma, v_a[1], v_b, c10_1, v_sfa[1], v_sfb[0]);
    sched_barrier_pairs_scale();

    c10_2 = mma_scale_one<T, 1, 0, 2>(
        mma, v_a[1], v_b, c10_2, v_sfa[1], v_sfb[0]);
    c10_3 = mma_scale_one<T, 1, 0, 3>(
        mma, v_a[1], v_b, c10_3, v_sfa[1], v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<12>{}, 0, 0, c00_12, c00_13, c00_14, c00_15);

    publish_output_quarter(opus::number<0>{}, opus::number<0>{});

    c10_4 = mma_scale_one<T, 1, 1, 0>(
        mma, v_a[1], v_b, c10_4, v_sfa[1], v_sfb[0]);
    c10_5 = mma_scale_one<T, 1, 1, 1>(
        mma, v_a[1], v_b, c10_5, v_sfa[1], v_sfb[0]);
    sched_barrier_pairs_scale();

    c10_6 = mma_scale_one<T, 1, 1, 2>(
        mma, v_a[1], v_b, c10_6, v_sfa[1], v_sfb[0]);
    c10_7 = mma_scale_one<T, 1, 1, 3>(
        mma, v_a[1], v_b, c10_7, v_sfa[1], v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<0>{}, 1, 0, c10_0, c10_1, c10_2, c10_3);

    c10_8 = mma_scale_one<T, 1, 2, 0>(
        mma, v_a[1], v_b, c10_8, v_sfa[1], v_sfb[0]);
    c10_9 = mma_scale_one<T, 1, 2, 1>(
        mma, v_a[1], v_b, c10_9, v_sfa[1], v_sfb[0]);
    sched_barrier_pairs_scale();

    c10_10 = mma_scale_one<T, 1, 2, 2>(
        mma, v_a[1], v_b, c10_10, v_sfa[1], v_sfb[0]);
    c10_11 = mma_scale_one<T, 1, 2, 3>(
        mma, v_a[1], v_b, c10_11, v_sfa[1], v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<4>{}, 1, 0, c10_4, c10_5, c10_6, c10_7);

    c10_12 = mma_scale_one<T, 1, 3, 0>(
        mma, v_a[1], v_b, c10_12, v_sfa[1], v_sfb[0]);
    c10_13 = mma_scale_one<T, 1, 3, 1>(
        mma, v_a[1], v_b, c10_13, v_sfa[1], v_sfb[0]);
    sched_barrier_pairs_scale();

    c10_14 = mma_scale_one<T, 1, 3, 2>(
        mma, v_a[1], v_b, c10_14, v_sfa[1], v_sfb[0]);
    c10_15 = mma_scale_one<T, 1, 3, 3>(
        mma, v_a[1], v_b, c10_15, v_sfa[1], v_sfb[0]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<8>{}, 1, 0, c10_8, c10_9, c10_10, c10_11);

    // B-half0 is dead on the final K tile.  Start the direct AGPR stores for
    // both completed C quadrants before the B-half1 MFMAs occupy the XDL pipe.
    store_output_quadrant(0, 0,
        c00_0, c00_1, c00_2, c00_3,
        c00_4, c00_5, c00_6, c00_7,
        c00_8, c00_9, c00_10, c00_11,
        c00_12, c00_13, c00_14, c00_15);
    store_output_quadrant(1, 0,
        c10_0, c10_1, c10_2, c10_3,
        c10_4, c10_5, c10_6, c10_7,
        c10_8, c10_9, c10_10, c10_11,
        c10_12, c10_13, c10_14, c10_15);

    if constexpr (T::OUTPUT_BF16) {
        // Final B1 is resident from the operand seed or the penultimate block.
        v_b = v_b_second;
    } else {
        v_b = load<T::VEC_B>(s_b, u_rb + sb_offset(stage, 1));
    }

    c01_0 = mma_scale_one<T, 0, 0, 0>(
        mma, v_a[0], v_b, c01_0, v_sfa[0], v_sfb[1]);
    c01_1 = mma_scale_one<T, 0, 0, 1>(
        mma, v_a[0], v_b, c01_1, v_sfa[0], v_sfb[1]);
    sched_barrier_pairs_scale();

    c01_2 = mma_scale_one<T, 0, 0, 2>(
        mma, v_a[0], v_b, c01_2, v_sfa[0], v_sfb[1]);
    c01_3 = mma_scale_one<T, 0, 0, 3>(
        mma, v_a[0], v_b, c01_3, v_sfa[0], v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<12>{}, 1, 0, c10_12, c10_13, c10_14, c10_15);

    publish_output_quarter(opus::number<1>{}, opus::number<0>{});

    c01_4 = mma_scale_one<T, 0, 1, 0>(
        mma, v_a[0], v_b, c01_4, v_sfa[0], v_sfb[1]);
    c01_5 = mma_scale_one<T, 0, 1, 1>(
        mma, v_a[0], v_b, c01_5, v_sfa[0], v_sfb[1]);
    sched_barrier_pairs_scale();

    c01_6 = mma_scale_one<T, 0, 1, 2>(
        mma, v_a[0], v_b, c01_6, v_sfa[0], v_sfb[1]);
    c01_7 = mma_scale_one<T, 0, 1, 3>(
        mma, v_a[0], v_b, c01_7, v_sfa[0], v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<0>{}, 0, 1, c01_0, c01_1, c01_2, c01_3);

    c01_8 = mma_scale_one<T, 0, 2, 0>(
        mma, v_a[0], v_b, c01_8, v_sfa[0], v_sfb[1]);
    c01_9 = mma_scale_one<T, 0, 2, 1>(
        mma, v_a[0], v_b, c01_9, v_sfa[0], v_sfb[1]);
    sched_barrier_pairs_scale();

    c01_10 = mma_scale_one<T, 0, 2, 2>(
        mma, v_a[0], v_b, c01_10, v_sfa[0], v_sfb[1]);
    c01_11 = mma_scale_one<T, 0, 2, 3>(
        mma, v_a[0], v_b, c01_11, v_sfa[0], v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<4>{}, 0, 1, c01_4, c01_5, c01_6, c01_7);

    c01_12 = mma_scale_one<T, 0, 3, 0>(
        mma, v_a[0], v_b, c01_12, v_sfa[0], v_sfb[1]);
    c01_13 = mma_scale_one<T, 0, 3, 1>(
        mma, v_a[0], v_b, c01_13, v_sfa[0], v_sfb[1]);
    sched_barrier_pairs_scale();

    c01_14 = mma_scale_one<T, 0, 3, 2>(
        mma, v_a[0], v_b, c01_14, v_sfa[0], v_sfb[1]);
    c01_15 = mma_scale_one<T, 0, 3, 3>(
        mma, v_a[0], v_b, c01_15, v_sfa[0], v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<8>{}, 0, 1, c01_8, c01_9, c01_10, c01_11);

    c11_0 = mma_scale_one<T, 1, 0, 0>(
        mma, v_a[1], v_b, c11_0, v_sfa[1], v_sfb[1]);
    c11_1 = mma_scale_one<T, 1, 0, 1>(
        mma, v_a[1], v_b, c11_1, v_sfa[1], v_sfb[1]);
    sched_barrier_pairs_scale();

    c11_2 = mma_scale_one<T, 1, 0, 2>(
        mma, v_a[1], v_b, c11_2, v_sfa[1], v_sfb[1]);
    c11_3 = mma_scale_one<T, 1, 0, 3>(
        mma, v_a[1], v_b, c11_3, v_sfa[1], v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<12>{}, 0, 1, c01_12, c01_13, c01_14, c01_15);

    publish_output_quarter(opus::number<0>{}, opus::number<1>{});

    c11_4 = mma_scale_one<T, 1, 1, 0>(
        mma, v_a[1], v_b, c11_4, v_sfa[1], v_sfb[1]);
    c11_5 = mma_scale_one<T, 1, 1, 1>(
        mma, v_a[1], v_b, c11_5, v_sfa[1], v_sfb[1]);
    sched_barrier_pairs_scale();

    c11_6 = mma_scale_one<T, 1, 1, 2>(
        mma, v_a[1], v_b, c11_6, v_sfa[1], v_sfb[1]);
    c11_7 = mma_scale_one<T, 1, 1, 3>(
        mma, v_a[1], v_b, c11_7, v_sfa[1], v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<0>{}, 1, 1, c11_0, c11_1, c11_2, c11_3);

    c11_8 = mma_scale_one<T, 1, 2, 0>(
        mma, v_a[1], v_b, c11_8, v_sfa[1], v_sfb[1]);
    c11_9 = mma_scale_one<T, 1, 2, 1>(
        mma, v_a[1], v_b, c11_9, v_sfa[1], v_sfb[1]);
    sched_barrier_pairs_scale();

    c11_10 = mma_scale_one<T, 1, 2, 2>(
        mma, v_a[1], v_b, c11_10, v_sfa[1], v_sfb[1]);
    c11_11 = mma_scale_one<T, 1, 2, 3>(
        mma, v_a[1], v_b, c11_11, v_sfa[1], v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<4>{}, 1, 1, c11_4, c11_5, c11_6, c11_7);

    c11_12 = mma_scale_one<T, 1, 3, 0>(
        mma, v_a[1], v_b, c11_12, v_sfa[1], v_sfb[1]);
    c11_13 = mma_scale_one<T, 1, 3, 1>(
        mma, v_a[1], v_b, c11_13, v_sfa[1], v_sfb[1]);
    sched_barrier_pairs_scale();

    c11_14 = mma_scale_one<T, 1, 3, 2>(
        mma, v_a[1], v_b, c11_14, v_sfa[1], v_sfb[1]);
    c11_15 = mma_scale_one<T, 1, 3, 3>(
        mma, v_a[1], v_b, c11_15, v_sfa[1], v_sfb[1]);
    sched_barrier_pairs_scale();

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<8>{}, 1, 1, c11_8, c11_9, c11_10, c11_11);

    // Stage this completed output row while later independent MFMAs execute.
    stage_output_row(opus::number<12>{}, 1, 1, c11_12, c11_13, c11_14, c11_15);
    __builtin_amdgcn_s_setprio(0);

    store_output_quadrant(0, 1,
        c01_0, c01_1, c01_2, c01_3,
        c01_4, c01_5, c01_6, c01_7,
        c01_8, c01_9, c01_10, c01_11,
        c01_12, c01_13, c01_14, c01_15);
    store_output_quadrant(1, 1,
        c11_0, c11_1, c11_2, c11_3,
        c11_4, c11_5, c11_6, c11_7,
        c11_8, c11_9, c11_10, c11_11,
        c11_12, c11_13, c11_14, c11_15);

    publish_output_quarter(opus::number<1>{}, opus::number<1>{});
}

} // namespace blockscale_generic
