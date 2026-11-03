---
name: mxfp8-layout-optimization
description: A repeatable optimization workflow for MI350/MI355 gfx950 MXFP8 GEMM kernels across NT, NN, TT, and TN layouts. Use when implementing, validating, profiling, or tuning layout-specific global-load, pre-shuffle, LDS, ds_read/ds_tr, scale ownership, wave mapping, buffering, and MFMA scheduling paths.
---

# MXFP8 Layout Optimization Skill

## Purpose

Use this skill to optimize MXFP8 GEMM kernels for the four physical input-layout combinations:

- NT
- NN
- TT
- TN

The workflow assumes a common target similar to:

- Architecture: MI350/MI355, gfx950 / CDNA4
- Compute tile: `256 x 256 x 128`
- Workgroup: 4 waves
- Initial wave grid: `2 x 2`
- Pipeline: 3-buffer
- Operand precision: MXFP8 / FP8 data with block scales
- Baseline: a working NT kernel

The central design rule is:

> Normalize each physical input layout into one canonical MFMA operand layout as early as practical, then share the same MFMA mainloop, accumulator layout, wave ownership, and epilogue.

Do not equate GEMM `N/T` with `ds_tr`. `N/T` describes physical/logical matrix orientation. `ds_read_tr` is an LDS-to-register data-rearrangement mechanism.

---

# 1. Required Inputs

Before tuning, collect:

1. Logical GEMM definition:
   - `C[M,N] = A[M,K] x B[K,N]`
2. Physical A shape and strides.
3. Physical B shape and strides.
4. A/B scale shapes, block size, packing, and strides.
5. Workgroup tile and per-wave output tile.
6. MFMA instruction shape and expected operand ownership.
7. Global-load instruction counts and requested bytes.
8. LDS allocation, store layout, read layout, and buffer count.
9. VGPR, AGPR, SGPR, occupancy, and workgroup count.
10. Runtime plus ATT/PMC evidence for the existing NT baseline.

If any item is unknown, record it as an explicit assumption. Do not silently infer it.

---

# 2. Establish the Roofline Baseline

## 2.1 Compute cycles

For a tile `(Mtile, Ntile, Ktile)`:

```text
ops = 2 * Mtile * Ntile * Ktile
compute_cycles = ops / peak_ops_per_cycle_per_CU
```

Current working model for MI355 FP8:

```text
peak_ops_per_cycle_per_CU = 8192
```

For `256 x 256 x 128`:

```text
ops = 2 * 256 * 256 * 128 = 16,777,216
compute_cycles = 16,777,216 / 8192 = 2048 cycles
```

Cross-check using the MFMA decomposition:

- 4 waves in a `2 x 2` grid
- each wave computes `128 x 128`
- `16 x 16 x 128` MFMA produces one `16 x 16` output block
- instructions per wave: `(128 / 16) * (128 / 16) = 64`
- if the instruction occupies 32 cycles, each wave requires `64 * 32 = 2048` cycles
- four waves execute concurrently on the four SIMD paths, so wall-clock compute remains about 2048 cycles

Treat the 32-cycle MFMA value as a hardware-model input that must be verified against the applicable ISA/internal source or ATT evidence.

## 2.2 Global/L1 data demand

For FP8 data, use 1 byte per element before adding scale traffic:

```text
A bytes = Mtile * Ktile
B bytes = Ntile * Ktile
unique input bytes = A bytes + B bytes
BW_global_demand = unique input bytes / compute_cycles
```

For `256 x 256 x 128`:

```text
A = 32 KiB
B = 32 KiB
A+B = 64 KiB
BW_global_demand = 64 KiB / 2048 = 32 B/cycle
```

Compare demand with the validated CU-to-L1 supply model. Under a 64 B/cycle supply assumption:

```text
32 B/cycle demand < 64 B/cycle supply
```

This indicates theoretical bandwidth headroom. It does not prove that latency is hidden or that the effective bandwidth reaches the roof.

## 2.3 LDS-to-VGPR demand

Count actual LDS-read bytes, not unique global bytes.

For a `2 x 2` wave grid:

- each A slab is consumed by two waves along N
- each B slab is consumed by two waves along M

A first-order estimate is:

```text
LDS read bytes ~= 2 * unique A+B bytes
               ~= 128 KiB per tile
BW_LDS_demand ~= 128 KiB / 2048
               ~= 64 B/cycle
```

Compare against the validated CDNA4 LDS read roof. The working roof is 256 B/cycle/CU.

A low average utilization does not rule out bank conflicts, segment conflicts, retries, issue bursts, or dependency stalls.

---

# 3. Canonical Layout Strategy

Define one canonical MFMA-facing representation for each operand:

```text
physical input layout
  -> global-load mapping
  -> optional offline pre-shuffle
  -> LDS store layout
  -> ds_read or ds_read_tr
  -> canonical lane/register ownership
  -> shared MFMA mainloop
```

Layout-dependent code should ideally stop before MFMA issue.

Keep these common across NT/NN/TT/TN where possible:

- workgroup tile
- wave grid
- per-wave output tile
- MFMA shape
- accumulator ownership
- K-loop structure
- scale application semantics
- epilogue

---

# 4. Pre-shuffle Procedure

## 4.1 Goal

Pre-shuffle is not a logical matrix transpose. It converts a standard logical layout into an MFMA-native packed physical layout.

For row-major `B[N,K]`, K is already contiguous. Packing is still useful because MFMA expects particular K fragments from particular rows to land in particular lanes and VGPRs.

## 4.2 Current packing concept

Use the implementation's exact reshape, but the conceptual decomposition is:

- group N rows by 16
- group K by 32
- decompose each K32 group into smaller K fragments
- permute dimensions so the fragments consumed together are physically adjacent

The current implementation may use a permutation such as:

```text
(0, 3, 1, 4, 2, 5)
```

Do not validate this permutation by shape names alone. Validate the complete mapping:

```text
logical (n,k)
  -> blocked indices
  -> packed byte offset
  -> global-load lane
  -> LDS byte offset
  -> ds_read/ds_read_tr destination
  -> final lane + VGPR
```

## 4.3 Weight versus activation

- Weight: prefer offline packing when the packed buffer can be reused.
- Activation: include packing in end-to-end cost.
- If quantization already occurs online, evaluate fused quantize-plus-pack to avoid a separate pass.
- Pre-shuffle data and scales together.

---

# 5. Layout-by-Layout Optimization

## 5.1 NT: reference kernel

Use NT to establish the optimal/common mainloop.

Freeze and record:

- tile and wave grid
- buffer depth
- MFMA schedule
- accumulator mapping
- B pre-shuffle
- scale path
- global-load and LDS-read instruction counts
- wait placement
- VGPR/AGPR/SGPR/LDS usage
- occupancy
- ATT timeline

The NT ATT trace defines the target pattern for continuous MFMA and overlapped data preparation.

## 5.2 NN: isolate the B-path change

Keep A, MFMA, accumulators, workgroup tile, and epilogue unchanged.

Tune only:

```text
B physical layout
  -> B global-load mapping
  -> B LDS store layout
  -> B ds_read/ds_read_tr path
  -> canonical MFMA.B ownership
```

Inspect:

- coalescing and transaction efficiency
- extra requested/returned bytes
- ds_tr instruction count
- ds_tr-to-MFMA dependency distance
- `vmcnt` and `lgkmcnt` waits
- LDS bank/segment conflicts
- prefetch lead distance
- MFMA bubbles

## 5.3 TT: isolate the A-path change

Reuse the already validated B path from NT when applicable.

Tune only:

```text
A physical layout
  -> A global-load mapping
  -> A LDS store layout
  -> A ds_read/ds_read_tr path
  -> canonical MFMA.A ownership
```

Inspect the same metrics as NN, plus any increase in VGPR/LDS pressure.

## 5.4 TN: combine both transformed paths

Tune TN after the isolated A and B paths are understood.

Compute an expected additive penalty:

```text
expected_TN_penalty ~= isolated_A_penalty + isolated_B_penalty
```

If measured TN penalty is materially larger, investigate shared-resource interaction:

- concurrent LDS reads map to the same bank/segment
- A/B ds_tr bursts overlap poorly
- TXA/LDS arbitration
- waitcnt placement serializes paths
- register pressure lowers occupancy
- prefetch lead distance shrinks
- A/B scale paths contend or extend dependencies

---

# 6. Wave Layout Sweep

Start with `2 x 2`:

```text
wave0 -> C[0:128,   0:128]
wave1 -> C[0:128, 128:256]
wave2 -> C[128:256, 0:128]
wave3 -> C[128:256,128:256]
```

This gives:

- A reuse across N-direction waves
- B reuse across M-direction waves

After correctness and baseline performance are stable, compare:

- `2 x 2`
- `1 x 4`
- `4 x 1`

Do not select only by runtime. Also compare:

- MFMA continuity
- ds_read/ds_tr issue pattern
- A/B read balance
- bank/segment conflicts
- VGPR/AGPR allocation
- occupancy
- effective prefetch lead

---

# 7. Triple-Buffer Validation

Triple buffering solves latency hiding, not bandwidth capacity.

Two independent conditions must hold:

## 7.1 Bandwidth condition

```text
average demand <= sustainable supply
```

## 7.2 Lead-time condition

```text
prefetch lead time >= data-ready latency
```

Measure in ATT:

1. first relevant global-load issue
2. last wait that gates LDS/use
3. data-ready point
4. first dependent LDS operation
5. first dependent MFMA

Check whether the future stage is ready before the consumer reaches it.

Do not assume total bytes equal in-flight bytes. A 32 KiB or 64 KiB stage may have a much smaller peak outstanding footprint if requests are bursty or waits are early.

---

# 8. LDS Optimization

Average LDS bandwidth is only a roofline check.

For each layout:

1. Record actual `ds_read`, `ds_read_tr`, and LDS-write instruction counts.
2. Derive bank indices from physical LDS addresses.
3. Compare concurrent A/B reads.
4. Prefer layouts that map simultaneous A/B reads to different physical banks/segments.
5. Check whether transpose reads create short dependencies into MFMA.
6. Check `lgkmcnt`, barriers, and retries/stalls.
7. Spread reads over the compute window when possible instead of issuing a sharp burst.

A layout can be correctness-equivalent and still be slower because of physical LDS mapping.

---

# 9. MXFP8 Scale Validation

For every data-layout change, validate scale ownership independently.

For selected fragments:

1. choose a lane and operand VGPR
2. derive the logical element coordinates
3. derive the MX scale-block coordinates
4. derive the packed scale offset
5. identify the scale register/owner
6. confirm the data fragment and scale fragment correspond

Use both:

- full numerical comparison
- small deterministic mapping cases with recognizable values

This catches cases where data ownership is correct but scale ownership is shifted.

---

# 10. Prologue, Persistent Scheduling, and Epilogue

If the kernel is persistent:

- overlap next-task address, tile-id, and scale/descriptor preparation with the tail of the current mainloop
- monitor SGPR live-range growth and occupancy

For output stores:

- prioritize the LDS space required for the mainloop input pipeline
- compare LDS-staged output against keeping output in VGPR and using buffer stores
- reject an epilogue optimization if it reduces occupancy or breaks triple buffering enough to hurt total runtime

Do not directly copy MI450-only mechanisms such as TDM pipes, TDM tag pools, cluster multicast, or TENSORcnt into an MI355 implementation. Transfer only the general principles: balanced request generation, early preparation, sufficient in-flight work, and conflict-free physical mapping.

---

# 11. Experimental Workflow

## Phase A: correctness

For each layout:

1. verify logical input shapes and strides
2. verify packed data mapping
3. verify packed scale mapping
4. verify LDS store/read mapping
5. verify lane/VGPR ownership
6. run deterministic small cases
7. run randomized numerical comparison

## Phase B: isolated performance deltas

1. measure NT baseline
2. change only B path for NN
3. change only A path for TT
4. combine both paths for TN
5. compare TN against the sum of isolated penalties

## Phase C: scheduling

Sweep:

- prefetch position
- waitcnt thresholds
- ds_read/ds_tr placement
- A/B read interleaving
- MFMA grouping
- barrier placement

## Phase D: resource trade-offs

Track every variant:

- VGPR
- AGPR
- SGPR
- LDS bytes
- occupancy
- waves/CU
- workgroups/CU
- runtime

## Phase E: tile and wave sweep

Only after the layout path is stable, compare alternate tiles and wave grids.

---

# 12. Measurement Template

Record one row per kernel variant:

```text
layout:
tile_m, tile_n, tile_k:
wave_grid:
buffer_count:
pre_shuffle_A:
pre_shuffle_B:
A_read_path:
B_read_path:
MFMA_shape:
MFMA_count_per_wave:
compute_cycles_model:
global_requested_bytes:
global_returned_bytes:
LDS_read_bytes:
VMEM_instruction_count:
DS_instruction_count:
MFMA_instruction_count:
VGPR:
AGPR:
SGPR:
LDS_allocation:
occupancy:
runtime_us:
MFMA_active_ratio:
VMEM_wait_cycles:
LDS_wait_cycles:
barrier_wait_cycles:
prefetch_lead_cycles:
notes:
```

---

# 13. Decision Rules

Use these rules when selecting a variant:

1. Reject incorrect data/scale ownership immediately.
2. Prefer a variant that preserves the canonical MFMA mainloop.
3. Prefer coalesced global access even if LDS performs a controlled transpose.
4. Prefer lower wall-clock critical-path cost, not merely fewer instructions.
5. Prefer stable MFMA issue over maximized instantaneous memory bandwidth.
6. Treat average bandwidth headroom and latency hiding as separate questions.
7. Investigate non-additive TN regression as contention, not as an unavoidable layout tax.
8. Do not accept lower occupancy without evidence that the larger tile/deeper pipeline wins end to end.
9. Evaluate activation packing end to end, not GEMM-only.
10. Keep all hardware throughput values explicitly sourced or marked as assumptions.

---

# 14. Completion Criteria

A layout is considered optimized enough for comparison when:

- correctness passes for data and scales
- the MFMA mainloop is shared or structurally equivalent to NT
- global traffic has no unexplained amplification
- prefetch covers the consumer deadline in steady state
- waitcnt/barrier bubbles are understood
- LDS conflicts are measured or ruled out
- resource usage and occupancy are recorded
- performance delta versus NT has a specific, evidence-backed explanation

The final target is not zero layout-conversion instructions. The target is to keep layout conversion off the wall-clock critical path so that MFMA remains continuously supplied.
