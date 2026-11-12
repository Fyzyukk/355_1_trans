#pragma once
// MXFP4 GEMM, 1x4 wave layout: wave w owns all 256 rows x columns [64w, 64w + 64) of the output tile.
//   A: global -> LDS (LDS-DMA, as m10), shared by the 4 waves, read per 64-row quadrant (double-buffered AGPR).
//   B: aiter shuffle_weight(16,16) layout, global -> AGPR directly (each wave its own 4 x 16-row blocks),
//      double-buffered over K tiles (loaded one K tile ahead), no LDS.
//   Scales: e8m0_shuffle panels (2 K tiles) by LDS-DMA, 2 buffers.
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "gemm_a4w4_mxfp4_4wave_traits.hpp"

using opus::operator""_I;

template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    // Wave (P0, P1) copies chunks (P0 * 2 + P1) * 4 + i (i = ih * 2 + il): its 4 issues land in
    // adjacent LDS chunks (1056 B apart, an immediate offset), with the same rows per chunk as
    // before: chunk c holds rows (c / 2) * 16 + 2 * (lane / 8) + c % 2.
    constexpr int threads_k = T::B_K_BYTES / T::VEC_A;                 // 8
    static_assert(threads_k == T::B_K / T::VEC_A_ELEMS && T::T_M == 2 && T::T_N == 2);
    static_assert(T::HALF_B_M / (T::BLOCK_SIZE / threads_k) == 4, "4 issues per half");
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p P0
        opus::number<2>{},                                             // p P1
        opus::number<2>{},                                             // y ih
        opus::number<T::WARP_SIZE / threads_k>{},                      // p lane / 8
        opus::number<2>{},                                             // y il
        opus::number<threads_k>{},                                     // p lane % 8
        opus::number<T::VEC_A>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_A>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_n, wave_id_m, lane_id / threads_k, (lane_id % threads_k) ^ ((((lane_id / threads_k) >> 1) & 1) << 1)}));
}

template<class T>
__device__ inline auto make_layout_sa(int wave_id_m, int wave_id_n) {
    // Chunk (P0 * 2 + P1) * 4 + issue (see make_layout_ga).
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p P0
        opus::number<2>{},                                             // p P1
        opus::number<4>{},                                             // y issue
        opus::number<T::VEC_A>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::smem_linear_wave_a + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_n, wave_id_m}));
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
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_m % T::T_M, lane_m / T::T_M, 0_I, (lane_id / T::W_M) ^ (((lane_m >> 2) & 1) << 1)}));
}


template<int VEC, class G, class LG, class LS>
__device__ inline void async_load_imm(G& g, void* smem_base, const LG& u_g, const LS& u_s, int s_os) {
    const auto gofs = opus::layout_to_offsets<VEC>(u_g);
    const auto sofs = opus::layout_to_offsets<VEC>(u_s);
    constexpr auto imm = opus::layout_imm_offsets_v<opus::remove_cvref_t<LS>, VEC>;
    constexpr int n = static_cast<int>(sizeof(imm) / sizeof(imm[0]));
    char* const base = reinterpret_cast<char*>(smem_base) + sofs[0];
    opus::static_for<n>([&](auto i) {
        constexpr int io = imm[decltype(i)::value] - imm[0];
        static_assert(io >= 0 && io <= 4095, "LDS issue stride must fit the 12-bit MUBUF offset");
        g.template async_load<VEC>(base, gofs[decltype(i)::value] - io, s_os, opus::number<io>{});
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


constexpr int SCHED_VALU = 0x002;
constexpr int SCHED_MFMA = 0x008;
constexpr int SCHED_VMEM_READ = 0x820;
constexpr int SCHED_DS_READ = 0x100;
constexpr int SCHED_DS_WRITE = 0x200;
constexpr int SCHED_VMEM_WRITE = 0x040;

#define SGB(mask, n) __builtin_amdgcn_sched_group_barrier((mask), (n), 0)

// q0: 8 A ds_reads (next quadrant) + 8 B direct loads (next K tile) over 32 MFMAs; scale gather VALU first.
__device__ inline void sched_q0() {
    SGB(SCHED_VALU, 6);
    opus::static_for<8>([&](auto r) { SGB(SCHED_MFMA, 1); SGB(SCHED_DS_READ, 1); SGB(SCHED_VALU, 2); SGB(SCHED_MFMA, 1);
        if constexpr (decltype(r)::value % 4 == 1) SGB(SCHED_VMEM_READ, 1);
        SGB(SCHED_MFMA, 1); SGB(SCHED_MFMA, 1); });
}
// q1 / q2: 8 A ds_reads front-loaded.
template<int NV>
__device__ inline void sched_q12() {
    opus::static_for<8>([&](auto r) { SGB(SCHED_MFMA, 1); SGB(SCHED_DS_READ, 1); SGB(SCHED_MFMA, 1);
        if constexpr (decltype(r)::value % (8 / NV) == 1) SGB(SCHED_VMEM_READ, 1); });
    opus::static_for<16>([&](auto) { SGB(SCHED_MFMA, 1); });
}
// q3: 8 A DMA (+ 2 panel DMA) + 8 A ds_reads (next K tile q0) + 10 scale ds_reads.
__device__ inline void sched_q3() {
    opus::static_for<8>([&](auto r) { SGB(SCHED_MFMA, 1); if constexpr (decltype(r)::value < 6) SGB(SCHED_VMEM_READ, 1); SGB(SCHED_MFMA, 1); SGB(SCHED_DS_READ, 2); });
    opus::static_for<16>([&](auto) { SGB(SCHED_MFMA, 1); });
}

// q3 with the scale gather of the next K tile in its second half (14 VALU).
__device__ inline void sched_q3g() {
    opus::static_for<8>([&](auto r) { SGB(SCHED_MFMA, 1); if constexpr (decltype(r)::value < 6) SGB(SCHED_VMEM_READ, 1); SGB(SCHED_MFMA, 1); SGB(SCHED_DS_READ, 2); });
    opus::static_for<16>([&](auto) { SGB(SCHED_MFMA, 1); SGB(SCHED_VALU, 1); });
}
__device__ inline void sched_q0b() {   // V7: 8 B loads over the first 8 MFMAs, 8 DS over the first 16
    opus::static_for<8>([&](auto) { SGB(SCHED_MFMA, 1); SGB(SCHED_VMEM_READ, 1); SGB(SCHED_DS_READ, 1); SGB(SCHED_VALU, 1); SGB(SCHED_MFMA, 1); });
    opus::static_for<16>([&](auto) { SGB(SCHED_MFMA, 1); });
}
__device__ inline void sched_q0n() {   // V3_GATHER: no gather VALU in front
    opus::static_for<8>([&](auto r) { SGB(SCHED_MFMA, 1); SGB(SCHED_DS_READ, 1); SGB(SCHED_VALU, 1); SGB(SCHED_MFMA, 1);
        if constexpr (decltype(r)::value % 4 == 1) SGB(SCHED_VMEM_READ, 1);
        SGB(SCHED_MFMA, 1); SGB(SCHED_MFMA, 1); });
}
template<int Q, int M, int N>
constexpr int c_reg_q() {
    constexpr int frag = (M * 4 + N) * 4;
    if constexpr (Q < 2) return 128 + Q * 64 + frag;      // AGPR
    else return -(128 + (Q - 2) * 64 + frag) - 2;          // VGPR
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void gemm_a4w4_mxfp4_4wave_kernel(opus_gemm_mxfp4_scale_kargs kargs) {
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp4_t;
    using D_B = opus::fp4_t;
    using D_ACC = opus::fp32_t;
    using D_C = opus::bf16_t;
    using D_AB = opus::u8_t;
    using D_SF = opus::e8m0_t;
    static_assert(T::E_M == 4 && T::E_N == 4 && T::E_K == 2 && T::NUM_WAVES == 4);

    const int num_tiles_n = ceil_div_scale(kargs.n, T::B_N);
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int tiles_per_batch = ceil_div_scale(kargs.m, T::B_M) * num_tiles_n;
    const int total_tiles = tiles_per_batch * kargs.batch;
    const int grid_blocks = kargs.grid_blocks;
    const int loops = kargs.k / T::B_K;   // even (K multiple of 512)

    constexpr int CHUNK = T::smem_linear_wave_a + T::smem_padding;        // 1088
    constexpr int smem_a_half = T::smem_m_rep * CHUNK;                    // 17408
    constexpr int stage_bytes = 2 * smem_a_half;                          // 34816
    constexpr int C_STRIDE = T::B_N + 8;                                  // bf16 elements per staged row
    constexpr int SF_BUFS = 2;
    constexpr int PANEL = T::SFA_PANEL_BYTES;                             // 4096
    static_assert(T::SFA_PANEL_BYTES == T::SFB_PANEL_BYTES);
    __shared__ __attribute__((aligned(16))) char smem_a[2 * stage_bytes];
    __shared__ __attribute__((aligned(16))) char smem_c[T::HALF_B_M * C_STRIDE * sizeof(D_C)];
    __shared__ __attribute__((aligned(16))) char smem_sfa[SF_BUFS * PANEL];
    __shared__ __attribute__((aligned(16))) char smem_sfb[SF_BUFS * PANEL];
    auto s_a = make_smem(reinterpret_cast<D_AB*>(smem_a));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_c));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
    auto s_sfa32 = make_smem(reinterpret_cast<opus::u32_t*>(smem_sfa));
    auto s_sfb32 = make_smem(reinterpret_cast<opus::u32_t*>(smem_sfb));

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, 1>{}, seq<T::T_M, T::T_N, T::T_K>{}, seq<T::W_M, T::W_N, T::W_K>{}, mfma_adaptor_swap_ab{});
    using MMA = decltype(mma);
    using CFrag = opus::vector_t<D_ACC, MMA::mma_c_len>;

    fp4x32_t v_a[2][2][4];   // [buf][kk][e]: A quadrant operands (pinned a0..a63)
    fp4x32_t v_b[2][2][4];   // [K-tile parity][kk][e]: B operands (pinned a64..a127)
    // scales live across output tiles too (V4: the last q3 of a tile reads / gathers the next tile's)
    using SFRAW_A = opus::vector_t<opus::u32_t, 8>;
    using SFRAW_B = opus::vector_t<opus::u32_t, 2>;
    SFRAW_A sfa_raw; SFRAW_B sfb_raw;
    unsigned int v_sfa[4][2], v_sfb[2], v_sfbn[2];
    int pbase = 0;           // V4: panel buffer parity of this tile's panel 0
    int pend = 0;            // V4: s_c holds the previous tile's bottom half, stored in this tile's first trip
    D_C* pend_ptr = reinterpret_cast<D_C*>(kargs.ptr_c);
    int pend_soff = 0;
    auto gA = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a), static_cast<unsigned int>(kargs.batch * kargs.stride_a_batch));
    auto gB = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b), static_cast<unsigned int>(kargs.batch * kargs.stride_b_batch));
    auto gSFA = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa), static_cast<unsigned int>(kargs.batch * kargs.stride_sfa_batch));
    auto gSFB = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb), static_cast<unsigned int>(kargs.batch * kargs.stride_sfb_batch));
    auto gC = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c));
    opus::vector_t<D_C, 8> v_pc[8];
    int prefetched = 0;      // V3_PREF: this tile's head (A(0), panel 0, B(0), A(1) half 0) was issued by the previous tile

#pragma clang loop unroll(disable)
    for (int tile_id = block_id_x(); tile_id < total_tiles; tile_id += grid_blocks) {
    const int batch_id = tile_id / tiles_per_batch;
    const int tile_in_batch = tile_id % tiles_per_batch;
    const int row = (tile_in_batch / num_tiles_n) * T::B_M;
    const int col = (tile_in_batch % num_tiles_n) * T::B_N;
    unsigned int lane_mask = ~0u;
    asm volatile("" : "+s"(lane_mask));
    const int lane_id = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lane_mask, __builtin_amdgcn_mbcnt_lo(lane_mask, 0u)));
    const int wave_id_m = wave_id % T::T_M;   // only used to split the A copies
    const int wave_id_n = wave_id / T::T_M;
    const int rows_a = min(T::B_M, kargs.m - row), rows_b = min(T::B_N, kargs.n - col);

    auto g_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a, static_cast<unsigned int>(rows_a * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b, static_cast<unsigned int>(rows_b * kargs.stride_b));
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + (row / 32) * kargs.stride_sfa, static_cast<unsigned int>(ceil_div_scale(kargs.m - row, 32) * kargs.stride_sfa));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / 32) * kargs.stride_sfb, static_cast<unsigned int>(ceil_div_scale(kargs.n - col, 32) * kargs.stride_sfb));

    const auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    const auto ra_offs = opus::layout_to_offsets<T::VEC_A>(make_layout_ra<T>(lane_id, 0));   // wave_m 0; wave_m 1 = + 2 chunks

    CFrag v_c[4][16];
#if !defined(V4) || defined(V5_NOFIRST)
    opus::static_for<4>([&](auto q) { opus::static_for<16>([&](auto f) { clear(v_c[q.value][f.value]); }); });
#endif
    const int tid0 = wave_id * T::WARP_SIZE + lane_id;

    auto ga_offset = [&](int half, int kt) { return half * T::HALF_B_M * kargs.stride_a + kt * T::B_K_BYTES; };
    auto sa_offset = [&](int stage, int half) { return stage * stage_bytes + half * smem_a_half; };
    auto prefetch_a = [&](int stage, int kt) {
        async_load_imm<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage, 0), ga_offset(0, kt));
        async_load_imm<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage, 1), ga_offset(1, kt));
    };
    // A quadrant Q (rows (Q/2)*128 + (2e + Q%2)*16 + r, e = 0..3) of the K tile in `stage` -> v_a[BUF].
    auto load_aq = [&](auto q_, auto buf_, int stage) {
        constexpr int Q = decltype(q_)::value, BUF = decltype(buf_)::value;
        opus::static_for<2>([&](auto kk) { opus::static_for<4>([&](auto e) {
            constexpr int agpr = BUF * 32 + (kk.value * 4 + e.value) * 4;
            [[clang::amdgpu_pin_agpr(agpr)]] v_a[BUF][kk.value][e.value] = __builtin_bit_cast(fp4x32_t,
                load<T::VEC_A>(s_a, ra_offs[e.value] + (Q % 2) * 2 * CHUNK + sa_offset(stage, Q / 2) + kk.value * T::W_K_BYTES));
        }); });
    };
    // B of K tile kt for this wave (16-row blocks 4 * wave + e) -> v_b[BB]: contiguous 1 KB per (e, kk), lane * 16.
    const int b_lane = lane_id * T::VEC_B;
    // per-fragment lane offsets in VGPRs (SGPR copies of these got hoisted out of the tile loop and spilled,
    // and every scratch reload waited vmcnt(0)); the K tile goes into soffset
    int b_voff[4];
    opus::static_for<4>([&](auto e) { b_voff[e.value] = b_lane + (4 * wave_id + e.value) * T::W_N * kargs.stride_b; });
    auto load_b_eg = [&](auto& gb_, auto bb_, auto e0_, auto ne_, int kt) {   // fragments e0 .. e0 + ne - 1
        constexpr int BB = decltype(bb_)::value, E0 = decltype(e0_)::value, NE = decltype(ne_)::value;
        opus::static_for<NE>([&](auto i) { opus::static_for<2>([&](auto kk) {
            constexpr int e = E0 + decltype(i)::value;
            constexpr int agpr = 64 + BB * 32 + (kk.value * 4 + e) * 4;
            [[clang::amdgpu_pin_agpr(agpr)]] v_b[BB][kk.value][e] = __builtin_bit_cast(fp4x32_t,
                load<T::VEC_B>(gb_, b_voff[e] + kk.value * 1024, kt * 2048));
        }); });
    };
    auto load_b_e = [&](auto bb_, auto e0_, auto ne_, int kt) {   // fragments e0 .. e0 + ne - 1
        constexpr int BB = decltype(bb_)::value, E0 = decltype(e0_)::value, NE = decltype(ne_)::value;
        opus::static_for<NE>([&](auto i) { opus::static_for<2>([&](auto kk) {
            constexpr int e = E0 + decltype(i)::value;
            constexpr int agpr = 64 + BB * 32 + (kk.value * 4 + e) * 4;
            [[clang::amdgpu_pin_agpr(agpr)]] v_b[BB][kk.value][e] = __builtin_bit_cast(fp4x32_t,
                load<T::VEC_B>(g_b, b_voff[e] + kk.value * 1024, kt * 2048));
        }); });
    };
    auto load_b = [&](auto bb_, int kt) { load_b_e(bb_, number<0>{}, number<4>{}, kt); };
    // Scale panel P (K tiles 2P, 2P + 1), lane-linear image of e8m0_shuffle: [32-row block j][tile 2][256 B].
    const int sfa_g_lane = (2 * wave_id + lane_id / 32) * kargs.stride_sfa + (lane_id % 32) * T::VEC_GLOBAL_SCALE;
    const int sfb_g_lane = (2 * wave_id + lane_id / 32) * kargs.stride_sfb + (lane_id % 32) * T::VEC_GLOBAL_SCALE;
    auto dma_panel = [&](int p) {
        const int buf = p & 1;
        g_sfa.template async_load<T::VEC_GLOBAL_SCALE>(smem_sfa + buf * PANEL + wave_id * 1024, sfa_g_lane + p * 512);
        g_sfb.template async_load<T::VEC_GLOBAL_SCALE>(smem_sfb + buf * PANEL + wave_id * 1024, sfb_g_lane + p * 512);
    };
    auto read_sf = [&](int kt) {
        const int base = ((kt >> 1) & 1) * (PANEL / 4) + (kt & 1) * 64 + lane_id;   // dwords
        opus::static_for<8>([&](auto j) { sfa_raw[j.value] = load<1>(s_sfa32, base + j.value * 128)[0]; });
        opus::static_for<2>([&](auto j) { sfb_raw[j.value] = load<1>(s_sfb32, base + (2 * wave_id + j.value) * 128)[0]; });
    };
    // dword D (32-row block) = [kk0 w0, kk0 w1, kk1 w0, kk1 w1]; quadrant Q needs w = Q % 2 of blocks (Q/2)*4 + e.
    auto gather_a = [&](auto q) {
        constexpr int Q = decltype(q)::value;
        constexpr unsigned int sel = 0x06020400u + (Q % 2) * 0x01010101u;
        const unsigned int t01 = __builtin_amdgcn_perm(sfa_raw[(Q / 2) * 4 + 1], sfa_raw[(Q / 2) * 4 + 0], sel);
        const unsigned int t23 = __builtin_amdgcn_perm(sfa_raw[(Q / 2) * 4 + 3], sfa_raw[(Q / 2) * 4 + 2], sel);
        v_sfa[Q][0] = __builtin_amdgcn_perm(t23, t01, 0x05040100u);
        v_sfa[Q][1] = __builtin_amdgcn_perm(t23, t01, 0x07060302u);
    };
    auto gather_b = [&]() {
        v_sfbn[0] = __builtin_amdgcn_perm(sfb_raw[1], sfb_raw[0], 0x05040100u);
        v_sfbn[1] = __builtin_amdgcn_perm(sfb_raw[1], sfb_raw[0], 0x07060302u);
    };
    auto gather_sf = [&]() {
        opus::static_for<4>([&](auto q) { gather_a(q); });
        gather_b(); v_sfb[0] = v_sfbn[0]; v_sfb[1] = v_sfbn[1];
    };
    auto mma_q = [&](auto q_, auto buf_, auto bb_) {
        constexpr int Q = decltype(q_)::value, BUF = decltype(buf_)::value, BB = decltype(bb_)::value;
        opus::static_for<16>([&](auto i) {
            constexpr int m = decltype(i)::value / 4, n = decltype(i)::value % 4;
            mma_scale_one<T, m, n, c_reg_q<Q, m, n>()>(mma, v_a[BUF][0][m], v_b[BB][0][n], v_c[Q][m * 4 + n], v_sfa[Q][0], v_sfb[0]);
            mma_scale_one<T, m, n, c_reg_q<Q, m, n>()>(mma, v_a[BUF][1][m], v_b[BB][1][n], v_c[Q][m * 4 + n], v_sfa[Q][1], v_sfb[1]);
        });
    };

#ifdef V4
    // ===== V5: tensor-level descriptors + per-tile SGPR bases; K tiles >= loops are the next tile's (its head is
    // copied in this tile's pipeline slots), so a prefetched tile needs no prologue =====
    const int nid0 = tile_id + grid_blocks;
    const bool has_next = nid0 < total_tiles;
    const int nid = has_next ? nid0 : tile_id;
    const int nbt = nid / tiles_per_batch, nin = nid % tiles_per_batch;
    const int nrow = (nin / num_tiles_n) * T::B_M, ncol = (nin % num_tiles_n) * T::B_N;
    const int ab0 = batch_id * kargs.stride_a_batch + row * kargs.stride_a, ab1 = nbt * kargs.stride_a_batch + nrow * kargs.stride_a;
    const int bb0 = batch_id * kargs.stride_b_batch + col * kargs.stride_b, bb1 = nbt * kargs.stride_b_batch + ncol * kargs.stride_b;
    const int sa0 = batch_id * kargs.stride_sfa_batch + (row / 32) * kargs.stride_sfa, sa1 = nbt * kargs.stride_sfa_batch + (nrow / 32) * kargs.stride_sfa;
    const int sb0 = batch_id * kargs.stride_sfb_batch + (col / 32) * kargs.stride_sfb, sb1 = nbt * kargs.stride_sfb_batch + (ncol / 32) * kargs.stride_sfb;
    const int half_loops = loops >> 1;
    // kt / pp are relative to this tile; NX: the copy belongs to the next tile (kt - loops / pp - half_loops)
    auto copy_a = [&](int stage, int half, int kt, auto nx_) {
        constexpr bool NX = decltype(nx_)::value;
        const int so = NX ? ab1 + (kt - loops) * T::B_K_BYTES : ab0 + kt * T::B_K_BYTES;
        async_load_imm<T::VEC_A>(gA, s_a.ptr, u_ga, u_sa + sa_offset(stage, half), so + half * T::HALF_B_M * kargs.stride_a);
    };
    auto copy_b = [&](auto bb_, auto e0_, auto ne_, int kt, auto nx_) {
        constexpr int BB = decltype(bb_)::value, E0 = decltype(e0_)::value, NE = decltype(ne_)::value;
        constexpr bool NX = decltype(nx_)::value;
        const int so = NX ? bb1 + (kt - loops) * 2048 : bb0 + kt * 2048;
        opus::static_for<NE>([&](auto i) { opus::static_for<2>([&](auto kk) {
            constexpr int e = E0 + decltype(i)::value;
            constexpr int agpr = 64 + BB * 32 + (kk.value * 4 + e) * 4;
            [[clang::amdgpu_pin_agpr(agpr)]] v_b[BB][kk.value][e] = __builtin_bit_cast(fp4x32_t,
                load<T::VEC_B>(gB, b_voff[e] + kk.value * 1024, so));
        }); });
    };
    auto copy_panel = [&](int pp, auto nx_) {   // panel pp (K tiles 2pp, 2pp + 1) -> buffer (pp + pbase) & 1
        constexpr bool NX = decltype(nx_)::value;
        const int buf = (pp + pbase) & 1;
        gSFA.template async_load<T::VEC_GLOBAL_SCALE>(smem_sfa + buf * PANEL + wave_id * 1024, sfa_g_lane, NX ? sa1 + (pp - half_loops) * 512 : sa0 + pp * 512);
        gSFB.template async_load<T::VEC_GLOBAL_SCALE>(smem_sfb + buf * PANEL + wave_id * 1024, sfb_g_lane, NX ? sb1 + (pp - half_loops) * 512 : sb0 + pp * 512);
    };
    auto read_sf_k = [&](int kt) {   // scales of K tile kt (kt >= loops: the next tile's, same buffer rule)
        const int base = ((((kt >> 1) + pbase) & 1) * (PANEL / 4)) + (kt & 1) * 64 + lane_id;
        opus::static_for<8>([&](auto j) { sfa_raw[j.value] = load<1>(s_sfa32, base + j.value * 128)[0]; });
        opus::static_for<2>([&](auto j) { sfb_raw[j.value] = load<1>(s_sfb32, base + (2 * wave_id + j.value) * 128)[0]; });
    };
    if (!prefetched) {
        pbase = 0;
        copy_a(0, 0, 0, number<false>{}); copy_a(0, 1, 0, number<false>{});
        copy_panel(0, number<false>{});
        __builtin_amdgcn_sched_barrier(0);
        copy_b(number<0>{}, number<0>{}, number<4>{}, 0, number<false>{});
        __builtin_amdgcn_sched_barrier(0);
        copy_a(1, 0, 1, number<false>{});
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 8>{});   // B(0) + A(1) half 0 may fly
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        read_sf_k(0);
        load_aq(number<0>{}, number<0>{}, 0);
        gather_a(number<0>{}); gather_a(number<1>{}); gather_a(number<2>{}); gather_b();
        __builtin_amdgcn_sched_barrier(0);
    }
#ifdef DBG_TW
    else { s_waitcnt_vmcnt(0_I); s_waitcnt_lgkmcnt(0_I); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0); }
#endif

    // 4 of the 16 bottom-half copies of the previous tile (part 0..3): LDS -> VGPR -> global
    // copy ii of a staged half: row tid0 / 32 + 8 ii, column (tid0 % 32) * 8 -> one VGPR base each side,
    // the per-copy row step in the SGPR offset (global) / the immediate (LDS)
    const int cpy_lds = (tid0 / 32) * C_STRIDE + (tid0 % 32) * 8;
    const int cpy_g = (tid0 / 32) * kargs.stride_c + (tid0 % 32) * 8;
    auto pend_part = [&](auto part_) {
        constexpr int PART = decltype(part_)::value;
        auto g_cp = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c), pend ? 0xffffffffu : 0u);
        opus::static_for<4>([&](auto i) {
            constexpr int ii = PART * 4 + decltype(i)::value;
            const auto v = load<8>(s_c, cpy_lds + ii * 8 * C_STRIDE);
            store<8>(g_cp, v, cpy_g, pend_soff + (T::HALF_B_M + ii * 8) * kargs.stride_c, opus::number<2>{});
        });
    };
    auto ktile5 = [&](auto p_, auto first_, int t, auto last_) {
        constexpr int P = decltype(p_)::value;
        constexpr bool FIRST = decltype(first_)::value;
        constexpr bool LAST = decltype(last_)::value;      // last trip: K tiles >= loops are the next tile's
        constexpr auto NX1 = number<LAST && P == 1>{};     // K tile t + 1
        constexpr auto NX2 = number<LAST>{};               // K tile t + 2 (and the panel copied in P 0)
#ifdef V5_NOZC
        constexpr bool ZC = false;
#else
        constexpr bool ZC = FIRST && P == 0;
#endif
        auto mma5 = [&](auto q_, auto buf_) {
            constexpr int Q = decltype(q_)::value, BUF = decltype(buf_)::value;
#ifdef V8_NMAJOR
            constexpr bool NMAJ = Q == 0;   // q0: B fragment n by n (B(t) chunks arrive in this order)
#else
            constexpr bool NMAJ = false;
#endif
            opus::static_for<16>([&](auto i) {
                constexpr int m = NMAJ ? decltype(i)::value % 4 : decltype(i)::value / 4;
                constexpr int n = NMAJ ? decltype(i)::value / 4 : decltype(i)::value % 4;
                if constexpr (ZC) {
                    using base_mma = typename opus::remove_cvref_t<decltype(mma)>::MMA;
                    using i32x8_t = opus::vector_t<opus::i32_t, 8>;
                    const fp4x32_t& a = v_a[BUF][0][m]; const fp4x32_t& b = v_b[P][0][n];
                    const i32x8_t a8 = __builtin_shufflevector(a, a, 0, 1, 2, 3, -1, -1, -1, -1);
                    const i32x8_t b8 = __builtin_shufflevector(b, b, 0, 1, 2, 3, -1, -1, -1, -1);
                    constexpr int CR = c_reg_q<Q, m, n>();
                    if constexpr (CR >= 0) { [[clang::amdgpu_pin_agpr(CR)]] v_c[Q][m * 4 + n] = base_mma{}(a8, b8, static_cast<int>(v_sfa[Q][0]), static_cast<int>(v_sfb[0]), opus::number<m>{}, opus::number<n>{}); }
                    else { [[clang::amdgpu_pin_vgpr(-CR - 2)]] v_c[Q][m * 4 + n] = base_mma{}(a8, b8, static_cast<int>(v_sfa[Q][0]), static_cast<int>(v_sfb[0]), opus::number<m>{}, opus::number<n>{}); }
                } else {
                    mma_scale_one<T, m, n, c_reg_q<Q, m, n>()>(mma, v_a[BUF][0][m], v_b[P][0][n], v_c[Q][m * 4 + n], v_sfa[Q][0], v_sfb[0]);
                }
                mma_scale_one<T, m, n, c_reg_q<Q, m, n>()>(mma, v_a[BUF][1][m], v_b[P][1][n], v_c[Q][m * 4 + n], v_sfa[Q][1], v_sfb[1]);
            });
        };
        copy_a(1 - P, 1, t + 1, NX1);   // A(t+1) half 1
        __builtin_amdgcn_sched_barrier(0);
        v_sfb[0] = v_sfbn[0]; v_sfb[1] = v_sfbn[1];
        gather_a(number<3>{});
#ifdef V7_BEARLY
        copy_b(number<1 - P>{}, number<0>{}, number<4>{}, t + 1, NX1);   // all of B(t+1): a full K tile of latency
        load_aq(number<1>{}, number<1>{}, P);
        mma5(number<0>{}, number<0>{});
        sched_q0b();
#else
        copy_b(number<1 - P>{}, number<0>{}, number<1>{}, t + 1, NX1);
        load_aq(number<1>{}, number<1>{}, P);
        mma5(number<0>{}, number<0>{});
        sched_q0n();
#endif
        __builtin_amdgcn_sched_barrier(0);
#ifndef V7_BEARLY
        copy_b(number<1 - P>{}, number<1>{}, number<2>{}, t + 1, NX1);
#endif
        load_aq(number<2>{}, number<0>{}, P);
#ifndef V5_NOPEND
        if constexpr (FIRST) pend_part(number<2 * P>{});
#endif
        mma5(number<1>{}, number<1>{});
#ifdef V7_BEARLY
        sched_q12<1>();
#else
        sched_q12<4>();
#endif
        __builtin_amdgcn_sched_barrier(0);
#ifndef V7_BEARLY
        copy_b(number<1 - P>{}, number<3>{}, number<1>{}, t + 1, NX1);
#endif
        load_aq(number<3>{}, number<1>{}, P);
        mma5(number<2>{}, number<0>{});
#ifdef V7_BEARLY
        sched_q12<1>();
#else
        sched_q12<2>();
#endif
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_lgkmcnt(0_I);
        s_waitcnt_vmcnt(number<8>{});   // A(t+1) landed (only B(t+1) is younger)
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (P == 0) copy_panel((t >> 1) + 1, NX2);
        copy_a(P, 0, t + 2, NX2);       // A(t+2) half 0
#ifndef V5_NOPEND
        if constexpr (FIRST) pend_part(number<2 * P + 1>{});
#endif
        read_sf_k(t + 1);
        load_aq(number<0>{}, number<0>{}, 1 - P);
        mma5(number<3>{}, number<1>{});
        gather_a(number<0>{}); gather_a(number<1>{}); gather_a(number<2>{}); gather_b();
        sched_q3g();
        __builtin_amdgcn_sched_barrier(0);
    };
    // trips: first (zero-C, pending C stores), middle, last (next tile's head); K = 512: first = last
    // K = 512 (one trip): no cross-tile head (the first trip's K tiles >= loops are harmless reads of this tile)
    ktile5(number<0>{}, number<true>{}, 0, number<false>{});
    ktile5(number<1>{}, number<true>{}, 1, number<false>{});
#pragma clang loop unroll(disable)
    for (int t = 2; t < loops - 2; t += 2) {
        ktile5(number<0>{}, number<false>{}, t, number<false>{});
        ktile5(number<1>{}, number<false>{}, t + 1, number<false>{});
    }
    if (loops > 2) {
        ktile5(number<0>{}, number<false>{}, loops - 2, number<true>{});
        ktile5(number<1>{}, number<false>{}, loops - 1, number<true>{});
    }
    // ===== write back: top half now, bottom half staged and stored in the next tile's first trip =====
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    const int c_r = lane_id % 16, c_c = wave_id * 64 + 4 * (lane_id / 16);
    auto stage_q = [&](auto q_) {
        constexpr int Q = decltype(q_)::value;
        opus::static_for<16>([&](auto i) {
            constexpr int m = decltype(i)::value / 4, n = decltype(i)::value % 4;
            const int r = (2 * m + Q % 2) * 16 + c_r;
            store<4>(s_c, cast<D_C>(v_c[Q][m * 4 + n]), r * C_STRIDE + c_c + n * 16);
        });
    };
    auto sync_lds = [&]() { s_waitcnt_lgkmcnt(0_I); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0); };
    const int c_soff = batch_id * kargs.stride_c_batch + row * kargs.stride_c + col;
    stage_q(number<0>{}); stage_q(number<1>{});
    sync_lds();
    opus::static_for<2>([&](auto h) {   // 2 x 8 copies (VGPR peak: the next tile's scales are live here)
        opus::vector_t<D_C, 8> v[8];
        opus::static_for<8>([&](auto i) {
            constexpr int ii = h.value * 8 + decltype(i)::value;
            v[i.value] = load<8>(s_c, cpy_lds + ii * 8 * C_STRIDE);
        });
        opus::static_for<8>([&](auto i) {
            constexpr int ii = h.value * 8 + decltype(i)::value;
            store<8>(gC, v[i.value], cpy_g, c_soff + ii * 8 * kargs.stride_c, opus::number<2>{});
        });
        __builtin_amdgcn_sched_barrier(0);
    });
    sync_lds();
    stage_q(number<2>{}); stage_q(number<3>{});
    sync_lds();
    pend = 1;
    pend_soff = c_soff;
    prefetched = has_next && loops > 2;
    pbase = (pbase + half_loops) & 1;
#else
    // ===== Prologue: A(0) + panel 0, A(1), B(0); wait for A(0) + panel 0 =====
    // Issue order matters for the counted wait below: sched barriers keep it.
    // Issue order matters for the counted wait below: sched barriers keep it.  B(0) goes before A(1) so that
    // the waitcnt state entering the main loop looks like the steady state (B(t) followed by >= 8 copies).
    // head of a tile from its gmem views: A(0) (both halves) + panel 0, B(0), A(1) half 0 (half 1 in K tile 0's q0)
    auto issue_head = [&](auto& ga_, auto& gb_, auto& gsfa_, auto& gsfb_) {
        async_load_imm<T::VEC_A>(ga_, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
        async_load_imm<T::VEC_A>(ga_, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
        gsfa_.template async_load<T::VEC_GLOBAL_SCALE>(smem_sfa + wave_id * 1024, sfa_g_lane);
        gsfb_.template async_load<T::VEC_GLOBAL_SCALE>(smem_sfb + wave_id * 1024, sfb_g_lane);
        __builtin_amdgcn_sched_barrier(0);
        opus::static_for<4>([&](auto e) { opus::static_for<2>([&](auto kk) {
            constexpr int agpr = 64 + (kk.value * 4 + e.value) * 4;
            [[clang::amdgpu_pin_agpr(agpr)]] v_b[0][kk.value][e.value] = __builtin_bit_cast(fp4x32_t,
                load<T::VEC_B>(gb_, b_voff[e.value] + kk.value * 1024, 0));
        }); });
        __builtin_amdgcn_sched_barrier(0);
        async_load_imm<T::VEC_A>(ga_, s_a.ptr, u_ga, u_sa + sa_offset(1, 0), ga_offset(0, 1));
        __builtin_amdgcn_sched_barrier(0);
    };
#ifdef V3_PREF
    if (!prefetched) issue_head(g_a, g_b, g_sfa, g_sfb);
    __builtin_amdgcn_sched_barrier(0);
    if (prefetched) s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 8 + 32>{});   // + the previous tile's 32 C stores
    else            s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 8>{});
#else
    issue_head(g_a, g_b, g_sfa, g_sfb);
#ifdef DBG_VM0
    s_waitcnt_vmcnt(0_I);
#else
    s_waitcnt_vmcnt(number<T::a_buffer_load_insts + 8>{});   // B(0) + A(1) half 0 may fly
#endif
#endif
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    read_sf(0);
    load_aq(number<0>{}, number<0>{}, 0);
#ifdef V3_GATHER
    gather_a(number<0>{}); gather_a(number<1>{}); gather_a(number<2>{}); gather_b();
#endif
    __builtin_amdgcn_sched_barrier(0);

    // ===== Main loop: K tile t uses stage t % 2, B buffer t % 2 =====
    auto ktile = [&](auto p_, int t) {
        constexpr int P = decltype(p_)::value;
        // second half of A(t+1)'s copy (into stage 1 - P), before every B(t+1) load (counted wait at the boundary)
        async_load_imm<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(1 - P, 1), ga_offset(1, t + 1));
        __builtin_amdgcn_sched_barrier(0);
#ifdef V3_GATHER
        v_sfb[0] = v_sfbn[0]; v_sfb[1] = v_sfbn[1];   // gathered in the previous q3
        gather_a(number<3>{});                        // only needed in q3
#else
        gather_sf();
#endif
        // q0 + B(t+1) e=0 + A q1
        load_b_e(number<1 - P>{}, number<0>{}, number<1>{}, t + 1);
        load_aq(number<1>{}, number<1>{}, P);
        mma_q(number<0>{}, number<0>{}, number<P>{});
#ifdef V3_GATHER
        sched_q0n();
#else
        sched_q0();
#endif
        __builtin_amdgcn_sched_barrier(0);
#ifdef DBG_QW
        s_waitcnt_lgkmcnt(0_I); s_waitcnt_vmcnt(0_I);
        __builtin_amdgcn_sched_barrier(0);
#endif
        // q1 + A q2 + B(t+1) e=1,2
        load_b_e(number<1 - P>{}, number<1>{}, number<2>{}, t + 1);
        load_aq(number<2>{}, number<0>{}, P);
        mma_q(number<1>{}, number<1>{}, number<P>{});
#ifdef V7_BEARLY
        sched_q12<1>();
#else
        sched_q12<4>();
#endif
        __builtin_amdgcn_sched_barrier(0);
        // q2 + A q3 + B(t+1) e=3
        load_b_e(number<1 - P>{}, number<3>{}, number<1>{}, t + 1);
        load_aq(number<3>{}, number<1>{}, P);
        mma_q(number<2>{}, number<0>{}, number<P>{});
#ifdef V7_BEARLY
        sched_q12<1>();
#else
        sched_q12<2>();
#endif
        __builtin_amdgcn_sched_barrier(0);
        // stage P fully read by every wave; A(t+1) (+ its panel) landed (only B(t+1) is younger)
        s_waitcnt_lgkmcnt(0_I);
#ifdef DBG_VM0
        s_waitcnt_vmcnt(0_I);
#else
        s_waitcnt_vmcnt(number<8>{});
#endif
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        // q3 + A(t+2) -> stage P (+ panel of t+2, t+3) + scales and A q0 of t+1
        if constexpr (P == 0) dma_panel((t >> 1) + 1);
        async_load_imm<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(P, 0), ga_offset(0, t + 2));   // first half of A(t+2)
        read_sf(t + 1);
        load_aq(number<0>{}, number<0>{}, 1 - P);
        mma_q(number<3>{}, number<1>{}, number<P>{});
#ifdef V3_GATHER
        gather_a(number<0>{}); gather_a(number<1>{}); gather_a(number<2>{}); gather_b();   // for K tile t + 1
        sched_q3g();
#else
        sched_q3();
#endif
        __builtin_amdgcn_sched_barrier(0);
    };
#pragma clang loop unroll(disable)
    for (int t = 0; t < loops; t += 2) {
        ktile(number<0>{}, t);
        ktile(number<1>{}, t + 1);
    }

    // ===== Write back C (bf16) through LDS, one half (128 rows) at a time =====
#ifdef V3_PREF
    {   // the next tile's head goes out first (stage 0 / stage 1 half 0 / panel 0 / v_b[0] are free:
        // every wave passed the last K tile's barrier; the garbage copies of tile `loops` land before these)
        const int nid = tile_id + grid_blocks;
        const bool has_next = nid < total_tiles;
        const int nb = nid / tiles_per_batch, nin = nid % tiles_per_batch;
        const int nrow = (nin / num_tiles_n) * T::B_M, ncol = (nin % num_tiles_n) * T::B_N;
        const int nra = min(T::B_M, kargs.m - nrow), nrb = min(T::B_N, kargs.n - ncol);
        auto ng_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + nb * kargs.stride_a_batch + nrow * kargs.stride_a, has_next ? static_cast<unsigned int>(nra * kargs.stride_a) : 0u);
        auto ng_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + nb * kargs.stride_b_batch + ncol * kargs.stride_b, has_next ? static_cast<unsigned int>(nrb * kargs.stride_b) : 0u);
        auto ng_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + nb * kargs.stride_sfa_batch + (nrow / 32) * kargs.stride_sfa, has_next ? static_cast<unsigned int>(ceil_div_scale(kargs.m - nrow, 32) * kargs.stride_sfa) : 0u);
        auto ng_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + nb * kargs.stride_sfb_batch + (ncol / 32) * kargs.stride_sfb, has_next ? static_cast<unsigned int>(ceil_div_scale(kargs.n - ncol, 32) * kargs.stride_sfb) : 0u);
        __builtin_amdgcn_sched_barrier(0);
        issue_head(ng_a, ng_b, ng_sfa, ng_sfb);
        prefetched = has_next;
    }
#else
    s_waitcnt_vmcnt(0_I);
#endif
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    const int c_r = lane_id % 16, c_c = wave_id * 64 + 4 * (lane_id / 16);
    auto stage_q = [&](auto q_) {
        constexpr int Q = decltype(q_)::value;
        opus::static_for<16>([&](auto i) {
            constexpr int m = decltype(i)::value / 4, n = decltype(i)::value % 4;
            const int r = (2 * m + Q % 2) * 16 + c_r;
            store<4>(s_c, cast<D_C>(v_c[Q][m * 4 + n]), r * C_STRIDE + c_c + n * 16);
        });
    };
    constexpr int VEC_OUT = 8;
    constexpr int c_copies = T::HALF_B_M * T::B_N / (T::BLOCK_SIZE * VEC_OUT);   // 16
    const int tid = wave_id * T::WARP_SIZE + lane_id;
    auto copy_out = [&](int half) {
        opus::vector_t<D_C, VEC_OUT> v[c_copies];
        opus::static_for<c_copies>([&](auto i) {
            const int linear = tid * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
            v[i.value] = load<VEC_OUT>(s_c, (linear / T::B_N) * C_STRIDE + linear % T::B_N);
        });
        opus::static_for<c_copies>([&](auto i) {
            const int linear = tid * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
            const int r = linear / T::B_N + half * T::HALF_B_M, c = linear % T::B_N;
            store<VEC_OUT>(g_c, v[i.value], r * kargs.stride_c + c, 0, opus::number<2>{});
        });
    };
    auto sync_lds = [&]() { s_waitcnt_lgkmcnt(0_I); __builtin_amdgcn_s_barrier(); __builtin_amdgcn_sched_barrier(0); };
    stage_q(number<0>{}); stage_q(number<1>{});
    sync_lds();
    copy_out(0);
    sync_lds();
    stage_q(number<2>{}); stage_q(number<3>{});
    sync_lds();
    copy_out(1);
    sync_lds();
#endif
    }   // persistent tile loop
#ifdef V4
    if (pend) {   // the last tile's bottom half
        s_waitcnt_lgkmcnt(0_I);
        const int tid0 = wave_id * T::WARP_SIZE + static_cast<int>(__builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u)));
        opus::static_for<16>([&](auto i) {
            const int linear = tid0 * 8 + decltype(i)::value * T::BLOCK_SIZE * 8;
            const auto v = load<8>(s_c, (linear / T::B_N) * C_STRIDE + linear % T::B_N);
            store<8>(gC, v, (linear / T::B_N + T::HALF_B_M) * kargs.stride_c + linear % T::B_N, pend_soff, opus::number<2>{});
        });
    }
#endif
}
