# MXFP8 Block-Scale GEMM Kernel for AMD GPU

A batched MXFP8 × MXFP8 → FP32 GEMM kernel built on the [OPUS](https://github.com/ROCm/aiter/tree/main/csrc/include/opus) template library, targeting AMD gfx950 (MI355X). It uses the hardware scaled Matrix Core instruction `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` (opus `mfma<>` scaled overload) with OCP E8M0 microscaling.

## What the kernel does

Computes a batched matrix multiply `C = A · B^T` per batch, where:

- `A` is `fp8 e4m3` of shape `[batch, M, K]`
- `B` is `fp8 e4m3` of shape `[batch, N, K]`
- `C` is `fp32` of shape `[batch, M, N]`

The K-reduction is accumulated in `fp32` inside the scaled MFMA; each 32-element K group is scaled by its `SFA` / `SFB` E8M0 exponent directly in hardware.

### MXFP8 microscaling scheme (OCP)

| Scale tensor | Shape (per batch) | Group granularity | Type |
|---|---|---|---|
| `SFA` | `[M, num_groups_k]` | `1 × GROUP_K = 1 × 32` | E8M0 (1 byte) |
| `SFB` | `[N, num_groups_k]` | `1 × GROUP_K = 1 × 32` | E8M0 (1 byte) |

- `GROUP_K = 32` (OCP standard); one E8M0 scale per 32 contiguous K elements, per row.
- E8M0: unsigned 8-bit exponent, bias 127; `value = 2^(byte - 127)`, `0x7F = 1.0`.
- The `16x16x128` scaled MFMA covers `K = 128 = 4 × GROUP_K`; the 4 per-group E8M0 bytes are packed into one `int32` (byte *i* → K-block *i*).

### Kernel configuration (v1: correctness-first)

Default traits (`gemm_a8w8_mxfp8_traits<>`):

| Parameter | Value |
|---|---|
| BLOCK_M × BLOCK_N × BLOCK_K | 16 × 16 × 128 |
| Warps per block / block size | 1 / 64 |
| MFMA tile (W_M × W_N × W_K) | 16 × 16 × 128 |
| GROUP_K | 32 |

> **v1 is intentionally simple**: one wave computes one 16×16 output tile via a single scaled-MFMA shape, looping over K. No LDS staging / software pipeline yet — those come after correctness is confirmed.

## Files

```
mxfp8_gemm/
├── Makefile
├── rebuild.sh
├── gemm_a8w8_mxfp8_common.h              # kargs + traits
├── gemm_a8w8_mxfp8_kernel_template.hpp   # kernel body
├── gemm_a8w8_mxfp8_kernel.cc             # device-only TU + host stub
└── gemm_a8w8_mxfp8_host.cc               # host launcher / benchmark / CPU reference
```

## Prerequisites

- ROCm with hipcc
- gfx950 GPU target (e.g. MI355X)
- OPUS headers from [aiter](https://github.com/ROCm/aiter): set `OPUS_INCLUDE_DIR` to `<aiter_root>/csrc/include`
- OpenMP (for CPU reference and random init)

## Build

```bash
cd opus_gemm/mxfp8_gemm
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include   # default: /root/workspace/aiter/csrc/include
make -j
```

Or use the convenience script (build + a default validated run):

```bash
./rebuild.sh
```

## Run

```bash
./build/gemm_a8w8_mxfp8.exe                       # defaults: b=1, m=16, n=16, k=128
./build/gemm_a8w8_mxfp8.exe -m 16 -n 16 -k 256 -v 1   # validate vs CPU
```

### Command-line options

| Flag | Description | Default |
|---|---|---|
| `-b`, `--b` | Batch size | 1 |
| `-m`, `--m` | M dimension | 16 |
| `-n`, `--n` | N dimension | 16 |
| `-k`, `--k` | K dimension | 128 |
| `-v`, `--verify` | CPU reference verification (0=off, 1=on) | 0 |

`M / N / K` must be multiples of `BLOCK_M / BLOCK_N / BLOCK_K` (16 / 16 / 128) in v1.
