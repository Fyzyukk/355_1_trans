#pragma once

#include "gemm_a8w8_mxfp8_scale_common.h"

// Runtime-K compact blockscale GEMM: one 256x256 output tile per workgroup.
// Four Wave64s use a0:a255 for C and two LDS stages for K128 matrix blocks.
template<bool OutputBF16 = false>
struct FourWaveTraits : BlockscaleCommon {
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;

    static constexpr int B_M = 256;
    static constexpr int B_N = 256;
    static constexpr int B_K = 128;

    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;

    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;

    static_assert(NUM_WAVES == 4);
    static_assert(NUM_WAVES == T_M * T_N * T_K);
    static_assert(HALF_B_M % (W_M * T_M) == 0);
    static_assert(HALF_B_N % (W_N * T_N) == 0);
    static_assert(B_K % (W_K * T_K) == 0);

    static constexpr int E_M = HALF_B_M / (W_M * T_M);
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);

    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr bool OUTPUT_BF16 = OutputBF16;
    static constexpr int OUTPUT_TILES_PER_WG = 1;
    static constexpr int VEC_SCALE_A = 16;

    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;

    // Compact input groups and the four hardware K32 lane groups are
    // different: each input K128 scale is broadcast to all four lane groups.
    static constexpr int MFMA_SCALE_GROUP_K = 32;
    static constexpr int SCALE_KGROUPS_PER_MFMA = W_K / MFMA_SCALE_GROUP_K;
    static constexpr int NUM_KGROUPS = B_K / MFMA_SCALE_GROUP_K;

    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = HALF_B_M / smem_sub;
    static constexpr int smem_n_rep = HALF_B_N / smem_sub;
    static constexpr int smem_padding = 32;

    static constexpr int SCALE_N_HALVES = B_N / HALF_B_N;

    static_assert(NUM_KGROUPS == 4);

    static constexpr int a_buffer_load_insts =
        HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts =
        HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int a_ds_read_insts =
        E_M * W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int b_ds_read_insts =
        E_N * W_N * W_K / (WARP_SIZE * VEC_B);

    static constexpr int SMEM_A_ELEMS = smem_m_rep * (smem_linear_wave + smem_padding) * 4;
    static constexpr int SMEM_B_ELEMS = smem_n_rep * (smem_linear_wave + smem_padding) * 4;
    // Scale-panel capacity is independent of the runtime GEMM K extent.
    static constexpr int SCALE_PANEL_K_TILES = 64;
    static constexpr int SFA_PANEL_PITCH = B_M;
    static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_TILES;
    static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * SCALE_PANEL_K_TILES * VEC_SF;
    static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;

    static_assert(E_M == 4 && E_N == 4 && E_K == 1);
    static_assert(a_ds_read_insts == 8 && b_ds_read_insts == 8);
    static_assert(SFA_PANEL_BYTES == 16384 && SFB_PANEL_BYTES == 512);
    static_assert(LDS_BYTES == 152064 && LDS_BYTES <= 163840);
};
