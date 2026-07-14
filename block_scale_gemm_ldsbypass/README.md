# FP8 Block-Scale GEMM Kernel for AMD GPU (aggressive LDS-bypass on B)

A batched FP8 × FP8 → FP32 block-scale GEMM kernel built on the [OPUS](https://github.com/ROCm/aiter/tree/main/csrc/include/opus) template library, targeting AMD gfx950 (MI355X).

This is the **LDS-bypass** variant of `block_scale_gemm_bpreshuffle`: the `B`
weight is repacked into the exact per-lane MFMA operand fragment order, so the
main kernel reads `B` **straight from global into registers and feeds MFMA**,
skipping B's `global→LDS→ds_read→register` round-trip entirely. `A` is
unchanged — it keeps its LDS staging and cross-N-wave reuse.

> **Result up front (this machine, gfx950):** the variant is *slower than the
> `bpreshuffle` baseline on every shape tested*, including small M. It is kept
> as a measured control, not a replacement. See [Performance](#performance) and
> [Why it loses](#why-it-loses) below. The main kernel's LDS usage does drop 4×
> (135168 → 33792 bytes/block) — the LDS-bypass mechanism works; it just isn't
> the bottleneck for this compute-bound pipeline.

## What the kernel does

Computes a batched `C = A · B^T` per batch, where:

- `A` is `fp8 e4m3` of shape `[batch, M, K]`
- `B` is `fp8 e4m3` of shape `[batch, N, K]`, repacked twice (see below)
- `C` is `fp32` of shape `[batch, M, N]`

K-reduction accumulates in `fp32` on MFMA; each K-group's partial sum is scaled
by the product of the corresponding `SFA`/`SFB` fp32 scale factors before being
added to the accumulator.

## Two variants of B'' generation

The main gemm kernel is **identical** in both variants (reads `B''` via
`bpp_frag_offset`, no LDS for B). They differ only in *who* produces `B''`:

- **GPU-repack** (`..._host.cc`, default): a one-shot device kernel builds `B''`.
- **host-pack** (`..._hostpack_host.cc`): the CPU builds `B''` — **no LDS anywhere
  in B'' generation**. See [Host-pack variant](#host-pack-variant-no-lds-at-all).

## Two-stage B repack (GPU-repack variant)

1. **Host K-panel preshuffle** (`shuffle_b`, same as `bpreshuffle`): `B[N,K]` →
   `B'[batch][kt][n][ki]` so each `BLOCK_K`-wide K-panel is contiguous per N-row.
2. **One-shot GPU fragment repack** (`gemm_a8w8_blockscale_ldsbypass_repack_kernel`):
   stages `B'` into LDS with the **same** `u_gb`/`u_sb` the main kernel would use,
   reads out the per-lane register fragment with the **same** `u_rb`, then writes
   those bytes **contiguously** into a compact buffer `B''`. Runs once, off the
   timed path (offline weight preprocessing).

### Why a GPU repack instead of a host permutation

The hard part of LDS-bypass is guaranteeing the byte order in `B''` matches the
per-lane MFMA operand fragment order (fixed by `u_rb`, hardware-dictated). A
hand-derived host permutation risks *silent numerical drift* if the 8-D
y/p-swizzle is reversed incorrectly.

The repack kernel avoids all hand-derivation: store and load both address `B''`
through the single `bpp_frag_offset` formula, and the fragment is never
reinterpreted between store and load. By construction the MMA sees inputs
**byte-for-byte identical** to the base (LDS) kernel — so any addressing mistake
surfaces immediately as a `-v 1` failure, not as quiet corruption.

### B'' layout (per batch)

```
B''[col_tile][k_tile][half_n(=2)][wave_n(=T_N)][lane(=64)][FRAG_B_ELEMS]
FRAG_B_ELEMS = HALF_B_N * B_K / (T_N * WARP_SIZE) = E_N * E_K * elem_b = 128 (fp8)
```

Each lane's 128 fp8 are contiguous; the main kernel reads them as
`b_frag_load_insts = 128/VEC_B = 8` back-to-back `VEC_B` loads. The fragment is
per-`(wave_n, lane)` and **shared across the T_M M-waves** — so `B''` totals the
same bytes as `B` (each N-row appears once per col-tile).

## Host-pack variant (no LDS at all)

`gemm_a8w8_blockscale_ldsbypass_hostpack_host.cc` drops the GPU repack kernel and
builds `B''` **entirely on the CPU**, so B'' generation never touches LDS. The main
gemm kernel is byte-for-byte the same.

The trick to stay "zero hand-derivation": the host **replays the exact opus layouts**
the GPU repack used. opus's layout *evaluation* chain (`make_layout`,
`layout_to_offsets`, `coord_to_linear`, ...) is already `__host__ __device__`; the
only device-only pieces are `unfold_x_stride` / `unfold_p_coord` (opus.hpp:2703/2711),
whose bodies are pure constexpr type arithmetic. The host file re-tags verbatim copies
of those two (namespace `hostlayout`) and rebuilds `u_gb`/`u_sb`/`u_rb` identically to
`make_layout_gb/sb/rb`. Then `host_pack_b()` mirrors the repack kernel step for step:

1. stage `B'` into a CPU array standing in for LDS via `u_gb → u_sb`,
2. read it back via `u_rb`, write the per-lane fragment into `B''` at `bpp_frag_offset`.

**One subtlety that does not appear in any layout:** CDNA `buffer_load_lds` scatters a
wave's 64 lanes into consecutive LDS slots — `u_sb` gives only the per-issue *wave*
base (no lane term); the hardware adds `lane * VEC_B`. The host must add that lane
stride explicitly when filling its LDS array (`u_rb`, an ordinary `ds_read`, *does*
carry the lane term, so read-out needs no fix-up).

Correctness is proven directly: a memcmp of host-pack `B''` against GPU-repack `B''`
is **0 differing bytes** (`N=K=256`), and all `-v 1` shapes are VALID. Since both
variants share the main kernel, gemm throughput is unchanged; the only difference is
the one-off preprocessing moving from GPU to CPU (slower wall-clock, but offline and
untimed — the point is proving the LDS-free packing is bit-exact).

## Main kernel loop

- **A**: unchanged — `global→LDS` (`buffer_load_lds`, tracked by `vmcnt` on
  CDNA), single-buffered across two half-tiles, then `ds_read` to registers.
- **B**: `v_b = load<VEC_B>(g_bpp, bpp_frag_offset(...))` — pure contiguous
  global load, no LDS, no reshuffle.
- **waitcnt**: correctness-first. The single CDNA `vmcnt` tracks A's
  `buffer_load_lds` then B's global loads in issue order; `s_waitcnt_vmcnt(0)`
  before each B use is conservative but correct. B needs no barrier (it never
  enters LDS); A keeps its `s_barrier`.
- Loop is `hn`-outer so only one B'' fragment (32 VGPR) is live at a time → zero
  spill despite `v_c` alone costing 128 VGPR.

## Block-scale scheme

Identical to `bpreshuffle`. `SFB` is **not** reshuffled — scale factors are
addressed by `(n_group, k_group)`, independent of B's physical layout.

| Scale tensor | Group granularity | Meaning |
|---|---|---|
| `SFA` | `GROUP_M × GROUP_K = 1 × 128` | one fp32 scale per `1 × 128` tile of `A` |
| `SFB` | `GROUP_N × GROUP_K = 128 × 128` | one fp32 scale per `128 × 128` tile of `B` |

## Kernel configuration

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
block_scale_gemm_ldsbypass/
├── Makefile
├── rebuild.sh
├── gemm_a8w8_blockscale_ldsbypass_common.h              # kargs + traits (adds B'' fragment consts)
├── gemm_a8w8_blockscale_ldsbypass_kernel_template.hpp   # repack kernel + main kernel
├── gemm_a8w8_blockscale_ldsbypass_kernel.cc             # device-only TU + host stub
├── gemm_a8w8_blockscale_ldsbypass_host.cc               # GPU-repack variant: launcher / shuffle_b / repack launch / benchmark / CPU ref
└── gemm_a8w8_blockscale_ldsbypass_hostpack_host.cc      # host-pack variant: CPU builds B'' (no LDS), same main kernel
```

## Build & run

```bash
cd opus_gemm/block_scale_gemm_ldsbypass
export OPUS_INCLUDE_DIR=/path/to/aiter/csrc/include
make -j                                                                          # builds both variants
./build/gemm_a8w8_blockscale_ldsbypass.exe -b 1 -m 1024 -n 1024 -k 1024 -v 1          # GPU-repack, validate vs CPU
./build/gemm_a8w8_blockscale_ldsbypass_hostpack.exe -b 1 -m 1024 -n 1024 -k 1024 -v 1 # host-pack, validate vs CPU
./build/gemm_a8w8_blockscale_ldsbypass.exe -b 1 -m 4096 -n 4096 -k 4096               # benchmark
```

`make hostpack` builds only the host-pack variant.

`M / N / K` must be multiples of `GROUP_M / GROUP_N / GROUP_K`. **`M ≥ BLOCK_M`
(256) is required** — smaller M writes out of a partial C tile and faults (this
is a pre-existing limitation shared with the `bpreshuffle` baseline, not new to
this variant). Verified VALID: `1024³`, `2048³`, `256×512×256 -b2`,
`512×768×512 -b3`, `1024×1024×512`, `512×512×1024`.

## Kernel resource usage

Reported by `-Rpass-analysis=kernel-resource-usage` on gfx950:

| Kernel | VGPR | SGPR | Occupancy | LDS (bytes) | Spill |
|---|:---:|:---:|:---:|:---:|:---:|
| main (`..._kernel`) | 240 | 42 | 2 | **33792** | 0 |
| repack (`..._repack_kernel`) | 38 | 28 | 8 | 16896 | 0 |

The main kernel's LDS is 33792 vs the baseline's 135168 — a **4× drop** from
removing B (and B's double buffer) from LDS. The mechanism works as intended.

## Performance

Measured on this machine, `batch = 1`, vs the `block_scale_gemm_bpreshuffle`
baseline (same machine/run):

| M=N=K | bpreshuffle TFlops | ldsbypass TFlops |
|---:|---:|---:|
| 1024 |   54.2 |   22.7 |
| 2048 |  156.0 |  163.3 |
| 4096 | 1156.9 |  696.4 |
| 8192 | 1019.9 |  761.4 |

Small-M (the variant's core hypothesis — `n = k = 4096/8192`):

| shape | bpreshuffle TFlops | ldsbypass TFlops |
|---|---:|---:|
| 256×4096×4096 |  68.2 | 38.3 |
| 256×8192×8192 | 144.3 | 84.5 |
| 512×4096×4096 | 136.9 | 90.8 |

## Why it loses

The pipeline is **compute-bound**, not LDS-bound (rocprofv3 on the 4096³
baseline: `SQ_INSTS_VALU` = 11.78M dominates; `SQ_INSTS_LDS` = 1.57M, of which
B's `ds_read` is only ~1/3). Removing B's LDS traffic therefore removes
non-bottleneck work while making B's global/L2 reads **×T_M larger**: with
`BLOCK_M = 256` and `T_M = 4`, four M-waves share one B fragment in the LDS
version but each re-fetch it from global here.

Crucially the small-M hypothesis *cannot* be tested at this tile config:
`BLOCK_M/T_M` are fixed, so even at `M = 256` there are still 4 M-waves sharing
B and the ×4 amplification applies. Confirming "small M / low reuse wins" would
require a dedicated **`T_M = 1` specialization** (one M-wave, no B reuse to give
up) — left as future work.

This matches aiter's flatmm choice to keep even a pre-transposed B in LDS.
