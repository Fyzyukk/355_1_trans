#pragma once

#include <opus/opus.hpp>

#include "gemm_a8w8_mxfp8_common.h"

using opus::operator""_I;

constexpr int MFMA_MASK = 0x08;
constexpr int VALU_MASK = 0x02;

#define SCHED_BARRIER(mask, cnt, group) __builtin_amdgcn_sched_group_barrier(mask, cnt, group)

template<int Pairs, int VALU_CNT, int Group>
__device__ inline void sched_barrier_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(VALU_MASK, VALU_CNT, Group);
    if constexpr (Pairs > 1) {
        sched_barrier_pairs<Pairs - 1, VALU_CNT, Group>();
    }
}

__device__ inline float e8m0_to_f32(unsigned char b) {
    return __builtin_bit_cast(float, static_cast<unsigned>(b) << 23);
}

// Accumulate one K-group's partial product into the C accumulator, applying that
// group's per-row/per-col MX scales BEFORE accumulation (mathematically:
//   C[m][n] = Σ_kg (sa[m][kg]·sb[n][kg]) · Σ_{k∈kg} A[m][k]·B[n][k]  ).
// c_part is the result of mma.step_k<KG>(v_a, v_b, 0) — full vtype_c but only the
// KG-th K-group contributed. Indexing follows C-layout 0 (probe_c_mma):
//   fragment i = m_rep*(E_N*ELEM_C) + n_rep*ELEM_C + pk
//   scale_a depends on (m_rep, kg)      -> idx = m_rep*E_K + kg
//   scale_b depends on (n_rep, pk, kg)  -> idx = (n_rep*ELEM_C + pk)*E_K + kg
// (pk is a distinct C column per lane, so each pk picks a different scale_b.)
template<int E_M, int E_N, int E_K, int ELEM_C, int KG, typename D_ACC>
__device__ inline void scale_and_accumulate(
    const opus::vector_t<D_ACC, E_M * E_N * ELEM_C>& c_part,
    const opus::vector_t<unsigned char, E_M * E_K>& scale_a,
    const opus::vector_t<unsigned char, E_N * ELEM_C * E_K>& scale_b,
    opus::vector_t<D_ACC, E_M * E_N * ELEM_C>& acc) {

    opus::static_for<E_M>([&](auto m_repeat){
        constexpr int m_rep = decltype(m_repeat)::value;
        opus::static_for<E_N>([&](auto n_repeat){
            constexpr int n_rep = decltype(n_repeat)::value;
            opus::static_for<ELEM_C>([&](auto pk_){
                constexpr int pk = decltype(pk_)::value;
                constexpr int i = m_rep * (E_N * ELEM_C) + n_rep * ELEM_C + pk;
                constexpr int scale_a_idx = m_rep * E_K + KG;
                constexpr int scale_b_idx = (n_rep * ELEM_C + pk) * E_K + KG;
                D_ACC scale_c = e8m0_to_f32(opus::get<scale_a_idx>(scale_a)) * e8m0_to_f32(opus::get<scale_b_idx>(scale_b));
                acc[i] += c_part[i] * scale_c;
            });
        });
    });
}

// Create layout for loading A matrix from global memory. (identical to block_scale)
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A_GLOBAL;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_M / threads_m_per_block>{},
        opus::number<T::T_N>{},
        opus::number<threads_m_per_wave>{},
        opus::number<T::T_M>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A_GLOBAL>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A_GLOBAL>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m, lane_id % threads_k}));
}

// Create layout for storing A matrix to shared memory. (identical to block_scale)
template<class T>
__device__ inline auto make_layout_sa(int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;

    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::smem_m_rep / num_waves>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::VEC_A_GLOBAL>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

// Create layout for reading A matrix from shared memory to registers. (identical to block_scale)
template<class T>
__device__ inline auto make_layout_ra(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{}, // 2
        opus::number<T::T_M / T::T_N>{}, // 2
        opus::number<T::T_M>{}, // 4
        opus::number<T::T_N>{}, // 2
        opus::number<T::W_M / T::T_M>{}, // 4
        opus::number<T::E_K>{}, // 4
        opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{}, // 1
        opus::number<T::WARP_SIZE / T::W_M>{}, // 4 
        opus::number<T::VEC_A>{}); // 8

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id % T::W_M;

    return opus::make_layout(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m / T::T_N, lane_id_m % T::T_M, wave_id_m % T::T_N, lane_id_m / T::T_M, lane_id / T::W_M}));
}

// Create layout for loading B matrix from global memory. (identical to block_scale)
template<class T>
__device__ inline auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    constexpr int threads_k = T::B_K / T::VEC_B_GLOBAL;
    constexpr int threads_n_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_n_per_wave = T::WARP_SIZE / threads_k;

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_N / threads_n_per_block>{},
        opus::number<T::T_N>{},
        opus::number<threads_n_per_wave>{},
        opus::number<T::T_M>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_B_GLOBAL>{});

    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B_GLOBAL>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m, lane_id % threads_k}));
}

// Create layout for storing B matrix to shared memory. (identical to block_scale)
template<class T>
__device__ inline auto make_layout_sb(int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;

    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::smem_n_rep / num_waves>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::VEC_B_GLOBAL>{});

    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sb_block_shape,
        opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

// Create layout for reading B matrix from shared memory to registers. (identical to block_scale)
template<class T>
__device__ inline auto make_layout_rb(int lane_id, int wave_id_n) {
    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{}, // 4
        opus::number<T::T_M>{}, // 4
        opus::number<T::T_N>{}, // 2
        opus::number<T::W_N / T::T_M>{}, // 4
        opus::number<T::E_K>{}, // 4
        opus::number<T::W_N * T::W_K / T::WARP_SIZE / T::VEC_B>{}, // 1
        opus::number<T::WARP_SIZE / T::W_N>{}, // 4
        opus::number<T::VEC_B>{}); // 8

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int lane_id_n = lane_id % T::W_N;

    return opus::make_layout(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{lane_id_n % T::T_M, wave_id_n, lane_id_n / T::T_M, lane_id / T::W_N}));
}

// Handwritten C-store layout for the real-machine 16x16x32 swap_ab accumulator.
// Ground truth from probe_c_mma.cu (feeds the exact ra/rb + swap_ab tiled mma and
// reads back a unique C[m][n] fingerprint): the accumulator register (lane, pk) holds
//   m = lane % W_M ,  n = (lane / W_M) * 4 + pk        (4 consecutive COLUMNS per lane)
// Extended to the block tile (fragment i = m_rep*(E_N*4) + n_rep*4 + pk, matching
// v_c / scale_c_group order):
//   m = m_rep*(W_M*T_M) + wave_id_m*W_M + row          , row = lane % W_M
//   n = n_rep*(W_N*T_N) + wave_id_n*W_N + gk*4 + pk     , gk  = lane / W_M
// The 4 pk fragments are contiguous in n (row-major C) -> store with VEC_C=4.
// y-dims (issue space) are ordered m_rep, n_rep, pk with pk innermost so the flat
// fragment index and the vectorized store both line up with the mma output.
template<class T>
__device__ inline auto make_layout_gc(int lane_id, int wave_id_m, int wave_id_n, int stride_c) {
    const int row = lane_id % T::W_M;   // -> m
    const int gk  = lane_id / T::W_M;   // -> n (group of 4 cols)

    auto gc_shape = opus::make_tuple(
        opus::number<T::E_M>{},                 // 0 y: m_rep   (slowest fragment axis)
        opus::number<T::E_N>{},                 // 1 y: n_rep
        opus::number<T::T_M>{},                 // 2 p: wave_id_m
        opus::number<T::T_N>{},                 // 3 p: wave_id_n
        opus::number<T::WARP_SIZE / T::W_M>{},  // 4 p: gk
        opus::number<T::W_M>{},                 // 5 p: row
        opus::number<4>{});                     // 6 y: pk      (fastest, vec dim)

    auto gc_stride = opus::make_tuple(
        (T::W_M * T::T_M) * stride_c,           // m_rep     -> rows
        (T::W_N * T::T_N),                       // n_rep     -> cols
        T::W_M * stride_c,                       // wave_id_m -> rows
        T::W_N,                                   // wave_id_n -> cols
        4,                                        // gk        -> cols (n = gk*4)
        stride_c,                                 // row       -> rows
        1);                                       // pk        -> cols (contiguous)

    auto gc_coord = opus::make_tuple(
        opus::_, opus::_,
        wave_id_m, wave_id_n, gk, row, opus::_);

    return opus::make_layout(gc_shape, gc_stride, gc_coord);
}

// Create layout for loading E8M0 scale factors for A from global memory.
template<class T>
__device__ inline auto make_layout_sfa(int lane_id, int wave_id_m, int stride_sfa) {
    constexpr auto sfa_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},     
        opus::number<T::T_M>{},      
        opus::number<T::W_M>{},      
        opus::number<T::E_K>{},      
        opus::number<1>{});          

    constexpr auto sfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}));

    return opus::make_layout(
        sfa_block_shape,
        opus::unfold_x_stride(sfa_block_dim, sfa_block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(sfa_block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}

// Create layout for loading E8M0 scale factors for B from global memory.
//
// The MFMA transposes lane<->data on the B side: the C output column is
//   n = n_rep*(W_N*T_N) + wave_id_n*W_N + gk*4 + pk ,  gk = lane / W_M
// (ground truth C-layout 0 from probe_c_mma). Each C column has its own B scale,
// and the 4 "pk" columns a lane holds are DISTINCT columns -> each needs its own
// scale_b. So unlike the old code (which copied rb's n = lane % W_N and had no pk
// dim), sfb must index n by gk = lane / W_M and add a pk (=4) y-dim.
//
// Per lane we load E_N * ELEM_C * E_K scale bytes. Fragment (y-dim) order is
// (n_rep, pk, kg) with kg innermost, so scale_and_accumulate reads
//   scale_b_idx = (n_rep*ELEM_C + pk) * E_K + kg .
// Memory is B scale [N, num_groups_k]: n -> rows (*stride_sfb), kg -> cols (*1).
template<class T>
__device__ inline auto make_layout_sfb(int lane_id, int wave_id_n, int stride_sfb) {
    const int gk = lane_id / T::W_M;   // -> n column group of 4 (matches C-layout 0)

    auto sfb_shape = opus::make_tuple(
        opus::number<T::E_N>{},                 // 0 y: n_rep  (slowest)
        opus::number<T::T_N>{},                 // 1 p: wave_id_n
        opus::number<T::WARP_SIZE / T::W_M>{},  // 2 p: gk
        opus::number<T::ELEM_C>{},              // 3 y: pk
        opus::number<T::E_K>{});                // 4 y: kg     (fastest)

    auto sfb_stride = opus::make_tuple(
        (T::W_N * T::T_N) * stride_sfb,         // n_rep     -> n rows
        T::W_N * stride_sfb,                     // wave_id_n -> n rows
        4 * stride_sfb,                          // gk        -> n rows (n = gk*4)
        stride_sfb,                              // pk        -> n rows (contiguous cols of C)
        1);                                       // kg        -> K-group cols

    auto sfb_coord = opus::make_tuple(
        opus::_, wave_id_n, gk, opus::_, opus::_);

    return opus::make_layout(sfb_shape, sfb_stride, sfb_coord);
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 2) void gemm_a8w8_mxfp8_kernel(opus_gemm_kargs kargs) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = opus::fp32_t;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;   

    const int wgid = block_id_x();
    const int num_tiles_n = ceil_div(kargs.n, T::B_N);
    const int row = (wgid / num_tiles_n) * T::B_M;
    const int col = (wgid % num_tiles_n) * T::B_N;

    const int batch_id = block_id_z();
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;

    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row * kargs.stride_sfa);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + col * kargs.stride_sfb);

    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    auto u_sb = make_layout_sb<T>(wave_id_m, wave_id_n);
    auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);
    auto u_sfa = make_layout_sfa<T>(lane_id, wave_id_m, kargs.stride_sfa);
    auto u_sfb = make_layout_sfb<T>(lane_id, wave_id_n, kargs.stride_sfb);

    constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
    constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
    __shared__ char smem_a[smem_a_elem * 4 * sizeof(D_A)];
    __shared__ char smem_b[smem_b_elem * 4 * sizeof(D_B)];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_a));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_b));

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    constexpr int ELEM_C = decltype(mma)::elem_c;

    typename decltype(mma)::vtype_a v_a;
    typename decltype(mma)::vtype_b v_b;
    typename decltype(mma)::vtype_c v_c[2][2];
    typename decltype(mma)::vtype_c zero_c;
    clear(v_c[0][0]);
    clear(v_c[0][1]);
    clear(v_c[1][0]);
    clear(v_c[1][1]);
    clear(zero_c);

    using vtype_sfa = vector_t<unsigned char, T::E_M * T::E_K>;
    using vtype_sfb = vector_t<unsigned char, T::E_N * T::ELEM_C * T::E_K>;
    vtype_sfa v_sfa[2][2];
    vtype_sfb v_sfb[2][2];

    auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K; };
    auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K; };
    auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
    auto sfa_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * (T::HALF_B_M / T::GROUP_M) * kargs.stride_sfa + tile_k * T::NUM_KGROUPS;};
    auto sfb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * (T::HALF_B_N / T::GROUP_N) * kargs.stride_sfb + tile_k * T::NUM_KGROUPS;};

    const int loops = ceil_div(kargs.k, T::B_K);

    // Correctness-first straight-line pipeline (K=256 = 2 K-tiles fully unrolled).
    // step_k requires the A/B operands to be live at scale time, which is
    // incompatible with the old deferred-v_mma software pipeline, so we load all
    // A/B/scale tiles first, then compute. Per subtile we split the 4 K-groups via
    // step_k and apply each group's MX scale before accumulating (method A, §3).
    // Scheduling tuning (setprio / sched_barrier_pairs) is intentionally dropped
    // here and can be re-layered once numerics are verified.

    // Load every scale + async A/B for both stages and both halves.
    opus::static_for<2>([&](auto stage_){
        constexpr int stage = decltype(stage_)::value;
        opus::static_for<2>([&](auto half_){
            constexpr int half = decltype(half_)::value;
            v_sfa[half][stage] = load(g_sfa, u_sfa, sfa_offset(half, stage));
            v_sfb[half][stage] = load(g_sfb, u_sfb, sfb_offset(half, stage));
            async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage, half), ga_offset(half, stage));
            async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage, half), gb_offset(half, stage));
        });
    });

    s_waitcnt_vmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    // Compute: C[hm][hn] += Σ_stage Σ_kg scale(sfa[hm][stage], sfb[hn][stage], kg)
    //                                     · step_k<kg>(ra(stage,hm), rb(stage,hn))
    opus::static_for<2>([&](auto stage_){
        constexpr int stage = decltype(stage_)::value;
        opus::static_for<2>([&](auto hm_){
            constexpr int hm = decltype(hm_)::value;
            v_a = load<T::VEC_A>(s_a, u_ra + sa_offset(stage, hm));
            opus::static_for<2>([&](auto hn_){
                constexpr int hn = decltype(hn_)::value;
                v_b = load<T::VEC_B>(s_b, u_rb + sb_offset(stage, hn));
                s_waitcnt_lgkmcnt(0_I);
                opus::static_for<T::E_K>([&](auto kg_){
                    constexpr int kg = decltype(kg_)::value;
                    auto v_part = mma.step_k(number<kg>{}, v_a, v_b, zero_c);
                    scale_and_accumulate<T::E_M, T::E_N, T::E_K, T::ELEM_C, kg, D_ACC>(
                        v_part, v_sfa[hm][stage], v_sfb[hn][stage], v_c[hm][hn]);
                });
            });
        });
    });

    __builtin_amdgcn_s_barrier();
    
    auto u_gc = make_layout_gc<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_c);

    auto c_offset = [&](int half_tile_m, int half_tile_n) {
        return half_tile_m * T::HALF_B_M * kargs.stride_c + half_tile_n * T::HALF_B_N;
    };

    store<T::VEC_C>(g_c, v_c[0][0], u_gc, c_offset(0, 0));
    store<T::VEC_C>(g_c, v_c[0][1], u_gc, c_offset(0, 1));
    store<T::VEC_C>(g_c, v_c[1][0], u_gc, c_offset(1, 0));
    store<T::VEC_C>(g_c, v_c[1][1], u_gc, c_offset(1, 1));
}
