# FP8 Block-Scale GEMM Kernel for AMD GPU (B preshuffle)

A batched FP8 × FP8 → FP32 block-scale GEMM kernel built on the [OPUS](https://github.com/ROCm/aiter/tree/main/csrc/include/opus) template library, targeting AMD gfx950 (MI355X).

This is the **B-preshuffle** variant of `block_scale_gemm`: the `B` weight is
repacked on the host into K-panels so the kernel reads it panel-contiguously.

## What the kernel does

The kernel computes a batched matrix multiply `C = A · B^T` per batch, where:

- `A` is `fp8 e4m3` of shape `[batch, M, K]`
- `B` is `fp8 e4m3` of shape `[batch, N, K]`, **preshuffled** into K-panels before upload
- `C` is `fp32` of shape `[batch, M, N]`

The K-reduction is accumulated in `fp32` on MFMA, and every K-group's partial sum is multiplied by the product of the corresponding `SFA` / `SFB` fp32 scale factors before being added to the final accumulator.

### B preshuffle layout

The host repacks `B[N, K]` (row-major) into K-panels

```
B'[batch][kt][n][ki] = B[batch][n][kt*BLOCK_K + ki]        # shape [batch, K/BLOCK_K, N, BLOCK_K]
```

so each `BLOCK_K`-wide K-panel of a given `N` row is contiguous, and consecutive
`N` rows within a panel are `BLOCK_K` apart. The kernel then loads `B` with
`stride_b = BLOCK_K` and jumps a whole panel (`N*BLOCK_K` elements) between K
iterations. This is done in `shuffle_b()` (host) + a one-line change to
`gb_offset` (kernel); the LDS staging, `ds_read`, MFMA schedule, and scale path
are **identical to the base kernel**. `SFB` does **not** need reshuffling — scale
factors are addressed by `(n_group, k_group)`, independent of B's physical layout.

The CPU reference consumes the original (un-shuffled) `B`, so verification catches
any shuffle/addressing mismatch directly.

### Block-scale scheme

Scale factors are shared across fixed-size groups along `(M, N, K)`:

| Scale tensor | Shape (per batch) | Group granularity | Meaning |
|---|---|---|---|
| `SFA` (A scale) | `[num_groups_k, num_groups_m]` | `GROUP_M × GROUP_K = 1 × 128` | one fp32 scale per `1 × 128` tile of `A` |
| `SFB` (B scale) | `[num_groups_n, num_groups_k]` | `GROUP_N × GROUP_K = 128 × 128` | one fp32 scale per `128 × 128` tile of `B` |

- `GROUP_M = 1`, `GROUP_N = 128`, `GROUP_K = 128`
- `num_groups_m = M / GROUP_M`, `num_groups_n = N / GROUP_N`, `num_groups_k = K / GROUP_K`
- All scale factors are stored as `fp32`

#### `SFA` is transposed for cache locality

### Kernel configuration

Default traits (`gemm_a8w8_blockscale_traits<>`):

| Parameter | Value |
|---|---|
| BLOCK_M × BLOCK_N × BLOCK_K | 256 × 256 × 128 |
| GROUP_M × GROUP_N × GROUP_K | 1 × 128 × 128 |
| Warps per block / block size | 8 / 512 |
| MFMA tile (W_M × W_N × W_K) | 16 × 16 × 128 |
| Warp tiling (T_M × T_N × T_K) | 4 × 2 × 1 |
| A/B global-load vector (fp8) | 16 elems |
| C global-store vector (fp32) | 4 elems |

## Files

```
block_scale_gemm_bpreshuffle/
├── Makefile
├── rebuild.sh
├── gemm_a8w8_blockscale_bpreshuffle_common.h              # kargs + traits
├── gemm_a8w8_blockscale_bpreshuffle_kernel_template.hpp   # kernel body
├── gemm_a8w8_blockscale_bpreshuffle_kernel.cc             # device-only TU + host stub
└── gemm_a8w8_blockscale_bpreshuffle_host.cc               # host launcher / shuffle_b / benchmark / CPU reference
```

## Prerequisites

- ROCm with hipcc
- gfx950 GPU target (e.g. MI355X)
- OPUS headers from [aiter](https://github.com/ROCm/aiter): set `OPUS_INCLUDE_DIR` to `<aiter_root>/csrc/include`
- OpenMP (for CPU reference and random init)

## Build

```bash
cd opus_gemm/block_scale_gemm_bpreshuffle
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
make -j
```

Or use the convenience script (build + a default run):

```bash
./rebuild.sh
```

## Run

```bash
./build/gemm_a8w8_blockscale_bpreshuffle.exe                         # defaults: b=8, m=256, n=512, k=256
./build/gemm_a8w8_blockscale_bpreshuffle.exe -b 1 -m 4096 -n 4096 -k 4096
./build/gemm_a8w8_blockscale_bpreshuffle.exe -b 1 -m 1024 -n 1024 -k 1024 -v 1   # validate vs CPU
```

### Command-line options

| Flag | Description | Default |
|---|---|---|
| `-b`, `--b` | Batch size | 8 |
| `-m`, `--m` | M dimension | 256 |
| `-n`, `--n` | N dimension | 512 |
| `-k`, `--k` | K dimension | 256 |
| `-v`, `--verify` | CPU reference verification (0=off, 1=on) | 0 |

All flags accept both `-m 4096` and `-m=4096` syntax. `M / N / K` must be multiples of `GROUP_M / GROUP_N / GROUP_K` respectively.

## Kernel resource usage

Reported by `-Rpass-analysis=kernel-resource-usage` on gfx950:

| VGPR | SGPR | Wave | Occupancy | LDS (bytes) |
|:---:|:---:|:---:|:---:|:---:|
| 256 | 71 | 8 | 2 | 135168 |

## Performance

Measured on this machine, `batch = 1`, square problem size `M = N = K`
(bpreshuffle vs the base `block_scale_gemm` kernel, same machine/run):

| M=N=K | base TFlops | bpreshuffle TFlops |
|---:|---:|---:|
| 1024 |   79.5 |   79.4 |
| 2048 |  291.2 |  296.0 |
| 4096 | 1310.3 | 1315.1 |
| 8192 | 1069.5 | 1108.0 |

Preshuffle here is on par with the base kernel and slightly ahead on large
shapes. Because `B[N, K]` already has its per-row K contiguous, the win comes
from packing consecutive `N` rows of each K-panel together (better cache-line
utilization), not from removing a transpose. A more aggressive variant that
preshuffles into the exact MFMA fragment order to bypass the LDS round-trip is
possible but not obviously a win for this pipeline, since B in LDS is reused
across the 4 M-waves — see the note in the authoring skill.
