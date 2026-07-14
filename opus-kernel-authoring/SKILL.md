---
name: opus-kernel-authoring
description: Authoring and reviewing OPUS HIP/C++ kernels — prefer opus.hpp wrappers over raw AMDGCN builtins, add concise explanatory comments at key logic, and apply the two replacement modes. Use when generating an OPUS kernel pipeline for a given shape/requirement, or when reviewing an existing OPUS kernel to swap raw builtins for opus.hpp equivalents.
argument-hint: [file or shape/requirement]
---

# OPUS Kernel Authoring

Guidance for writing and reviewing OPUS kernels (`csrc/include/opus/opus.hpp`). Three rules: **prefer opus wrappers over builtins**, **comment the key logic concisely**, **replace in the right mode**.

This is about API idiom / coverage — NOT compile time. For build-time optimization see `/opus-kernel-best-practice` and `/opus-module-build-optimization`.

## Rule 1 — Prefer `opus.hpp` wrappers, fall back to builtin only if none exists

When emitting any device operation (index, sync, load/store, MMA, shuffle, cast, layout, container), **first look for an existing wrapper in `opus.hpp`**. Use it if present. Only drop to a raw `__builtin_amdgcn_*` / hand-written intrinsic when `opus.hpp` has no equivalent.

**Always verify against the current header — do not trust the table below from memory.** The header evolves; symbols get added/renamed. Before relying on any wrapper, grep it:

```bash
grep -nE 'shfl|thread_id_x|make_gmem|make_mfma|sync_threads|<symbol>' csrc/include/opus/opus.hpp
```

Reference map (verify before use):

| Operation | opus.hpp wrapper | Raw builtin (replace with wrapper) |
|---|---|---|
| thread / block / grid index | `opus::thread_id_x/y/z()`, `block_id_*()`, `block_size_*()`, `grid_size_*()` | `__builtin_amdgcn_workitem_id_*`, `workgroup_id_*`, `workgroup_size_*`, `grid_size_*` |
| barrier / sync | `opus::sync_threads()`, `opus::s_waitcnt*`, `opus::warp_all()` | `__builtin_amdgcn_s_barrier()`, `__ockl_wfall_i32` |
| warp shuffle | `opus::shfl()` | `__builtin_amdgcn_ds_bpermute` |
| global / shared / buffer mem | `opus::make_gmem/make_smem/make_buffer_rsrc` + `load/store/async_load/tr_load` (+ `_if`) | raw `reinterpret_cast` deref, hand-written buffer intrinsics |
| matrix core | `opus::make_mfma/make_wmma/make_tiled_mma` + `operator()` | `__builtin_amdgcn_mfma_*` / `wmma_*` directly |
| index / layout | `opus::make_layout/make_layout_packed`, adaptor helpers | hand-written offset multiply-add |
| containers / constants | `opus::make_tuple/make_array/make_vector`, `seq<>`, `number<>` | raw arrays, magic numbers |
| type / conversion | `opus::cast<>`, `opus::numeric_limits<>`, `fp32_to_fp8` etc. | `__builtin_convertvector`, hand-rolled bit ops |

If a needed op has no wrapper, keep the builtin and note briefly why (see Rule 2).

## Rule 2 — Comment the key logic, concisely (2-3 lines max)

Key logic in an OPUS kernel **must** carry a short explanatory comment. This overrides the repo's default "write no comments" stance for OPUS kernels specifically — because shape/tiling/instruction choices are non-obvious and reviewers need the *why*.

Comment these, each in **≤ 2-3 lines**, explaining the *why* not the *what*:
- **shape / tile partitioning** — how the tensor is split across threads/waves (x/y/p dims), why these tile sizes
- **instruction choice** — why this MFMA/WMMA variant, this vector width, this layout
- **optimization** — the mechanism behind a non-obvious optimization (e.g. why async_load + double buffer here)
- **builtin fallback** — when a raw builtin is kept, one line on why no wrapper fits

Do NOT write multi-line paragraphs or docstrings. If it can't be said in 2-3 lines, it's too much.

```cpp
// 64x32 tile, 4 threads/row × 16 rows, each thread 8 contiguous elems (matrix-core A layout)
auto v_a = g_a.load<8>(u);

// swap A/B so 32x32x8 MFMA reads B row-major from LDS without transpose
auto mma = make_mfma<fp16_t, fp16_t, fp32_t>(32_I, 32_I, 8_I, mfma_adaptor_swap_ab{});
```

## Rule 3 — Two replacement modes

### Mode A — during generation: replace inline, no confirmation
While writing the pipeline, if you spot that an op has an `opus.hpp` wrapper, **use the wrapper directly**. Do not pause to ask. Must not affect the runtime behavior of the finished pipeline — the wrapper is a semantic equivalent, so verify the wrapper's signature/semantics match before substituting.

### Mode B — post-generation review: list all, wait for confirmation
When reviewing a complete existing kernel, scan it start to finish, then **present ALL replaceable sites at once as a single checklist** — do not replace anything yet. Each item: file:line, the current builtin, the proposed `opus::` wrapper, and a one-line reason. Then **wait for the user's confirmation**; apply only after they approve (they may accept a subset).

```
Found N replaceable sites:
1. opus_gemm.cu:142  __builtin_amdgcn_s_barrier()  → opus::sync_threads()   (direct equiv)
2. opus_gemm.cu:210  raw fp16x4 reinterpret_cast load → opus::make_gmem + load<4>  (vectorized dispatch)
...
Confirm which to apply (all / e.g. 1,3 / none).
```

## Quick checklist

- [ ] Every device op checked against `opus.hpp` (grep-verified) before falling back to a builtin
- [ ] Key logic (shape, instruction choice, optimization, builtin fallback) commented in ≤ 2-3 lines
- [ ] Generation-time swaps applied inline; review-time swaps listed together and applied only after confirmation
