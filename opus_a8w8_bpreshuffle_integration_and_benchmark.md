# Opus a8w8 block-scale bpreshuffle GEMM 集成、tune 与性能总结

> 更新日期：2026-07-14
> 目标架构：gfx950 / MI355X
> 状态：已完成 Opus 接入、正确性验证；本轮针对**更正后的目标 shape 集合**重新 tune，并与 CK/CKTile 对比。

## 1. 结论摘要

本次把 Opus 的 `a8w8 block-scale bpreshuffle GEMM` 接入到 aiter，新增 Python API
`gemm_a8w8_bpreshuffle_opus`，支持 fp8 x fp8 输入、fp32/bf16 输出，B 权重可以由 wrapper
内部转换成 K-panel bpreshuffle 布局，也可以传入已经预重排的 B。

本轮只 tune Opus，不 tune CK/CKTile。CK 和 CKTile 在测试中只作为强制调用的对比后端。

### 1.1 本轮更正的目标 shape 集合

上一轮用错了 shape，本轮改用正确的目标 shape，按来源分三组（`(N, K) -> [M ...]`）：

- **CK 来源组**
  - `N=768, K=7168`  -> M=[384, 544, 672, 832, 1088, 1440, 1600, 1696, 1856, 32512, 32768]
  - `N=2048, K=7168` -> M=[124, 136, 180, 200, 212, 232, 236, 248, 252, 256, 3840, 4096]
  - `N=7168, K=384`  -> M=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 68, 192, 256, 384, 544, 672, 832, 992, 1088, 1440, 1600, 1696, 1856, 1888, 1984, 2016, 2048, 32512, 32768]
  - `N=7168, K=7168` -> M=[4, 8, 12, 16, 20, 24, 32, 48, 64, 68, 80, 84, 96, 104, 124, 496, 544, 720, 800, 848, 928, 944, 992, 1008, 1024, 3840, 4096]
  - `N=8192, K=1536` -> M=[3840, 4096]
  - `N=65536, K=1536`-> M=[4, 8, 12, 16, 20, 24, 48, 84, 104, 124, 200, 212, 232, 236, 248, 252, 256, 3840, 4096]
- **CKTile 来源组**
  - `N=7168, K=7168` -> M=[15360, 16384]
  - `N=7168, K=16384`-> M=[3840, 4096]
  - `N=65536, K=1536`-> M=[136, 180]
- **ASM 来源组**
  - `N=768, K=7168`  -> M=[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 68, 192, 256, 992, 1888, 1984, 2016, 2048]
  - `N=2048, K=7168` -> M=[4, 8, 12, 16, 20, 24, 48, 68, 84, 104]
  - `N=7168, K=7168` -> M=[136, 180, 192, 200, 212, 232, 236, 248, 252, 256, 272, 336, 416]
  - `N=7168, K=16384`-> M=[4, 8, 12, 16, 20, 24, 48, 68, 84, 104, 124, 136, 180, 200, 212, 232, 236, 248, 252, 256]
  - `N=8192, K=1536` -> M=[256]
  - `N=65536, K=1536`-> M=[68]

去重后共 **169** 个 unique `(M, N, K)`。当前 Opus 两个 tile（`kid=3` 256x256x128、
`kid=9002` 128x256x128）的 shape guard 要求 `N%256==0`、`K%128==0`、`K/128` 为偶数、且
`M` 是 128 或 256 的倍数，因此本集合里只有 **23** 个 shape 能被安全覆盖，其余 **146** 个被自动跳过
（主要是小 M（不是 128/256 的倍数）和 `K=384`（K/128=3 为奇数）两类）。

### 1.2 Tune 结果

Opus tune 候选为 `kid=3` 和 `kid=9002`。`kid=3` 是最初添加的 Opus bpreshuffle kernel，
tile 为 `256x256x128`；`kid=9002` 是 tune 变体，tile 为 `128x256x128`。
在 23 个可支持的目标 shape 上，winner **全部为 `kid=9002`**，结果已落到：

```text
aiter/configs/opus_a8w8_bpreshuffle_tuned_gemm.csv
```

性能结论按 shape 来源分三组（仅统计可支持 shape）：

- **CK 来源组（14 个支持）**：Opus 仍明显慢于 CK，总耗时为 CK 的 `1.72x`；对 CKTile 总体仅
  `1.11x`（省约 10.2%），且在小 M shape（如 `256x2048x7168`、`384x768x7168`）上 CKTile 反而更快。
  只有大 M 的 `32512x768x7168` / `32768x768x7168` 上 Opus 接近 CK（`1.06x` / `1.08x`）。
- **CKTile 来源组（4 个支持）**：本轮主要收益点。Opus 对 CKTile 总体 `1.50x` 加速，耗时降低约
  `33.2%`；但仍慢于 CK，总耗时为 CK 的 `1.42x`。
- **ASM 来源组（5 个支持，全部是 M=256 或 2048 的小 M）**：Opus 远慢于 CK（`3.41x`）和 CKTile
  （Opus 耗时约为 CKTile 的 1.78x），完全不适合替代 ASM/small-M 路径。

下文表格中：

- `us` 越低越好。
- `Opus/CK` 是耗时比，`>1` 表示 Opus 比 CK 慢。
- `CKTile/Opus` 是 `CKTile_us / Opus_us`，`>1` 表示 Opus 比 CKTile 快。

## 2. Opus GEMM 如何接入 aiter

### 2.1 Kernel 语义

该 kernel 是 aiter 现有 Opus `a8w8_scale` 的 B-preshuffle 变体，核心 GEMM 仍是：

```text
A: fp8 e4m3, [batch, M, K]
B: fp8 e4m3, [batch, N, K]
x_scale/SFA: fp32, [batch, M, K/128]
w_scale/SFB: fp32, [batch, N/128, K/128]
Y: fp32 或 bf16, [batch, M, N]
GROUP: 1x128x128
```

B 的物理布局从普通 `[N, K]` 改成 K-panel bpreshuffle：

```text
B_panel[b, kt, n, ki] = B[b, n, kt * 128 + ki]
```

因此 device 端的关键差异是：

- `b_offset` 按 K-panel 跳转。
- `stride_b` 使用 `B_K=128`，而不是原始 K。
- SFA/SFB scale 布局沿用 aiter 的 `a8w8_scale` 约定，不使用 standalone kernel 中的转置 SFA 布局。

### 2.2 主要改动文件

| 模块 | 文件 | 说明 |
|---|---|---|
| device pipeline | `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_scale_bpreshuffle_gfx950.cuh` | 新增 gfx950 bpreshuffle pipeline，复用 a8w8_scale traits/kargs，改 B 寻址 |
| kernel registry | `csrc/opus_gemm/opus_gemm_common.py` | 新增 `a8w8_scale_bpreshuffle_kernels_list`，注册 `kid=3` 和 `kid=9002` |
| gfx950 codegen | `csrc/opus_gemm/codegen/gen_instances_gfx950.py` | 新增 `a8w8_scale_bpreshuffle` tag、pipeline header、kernel func、traits/kargs map 和 emit |
| generic codegen | `csrc/opus_gemm/gen_instances.py` | 新增 tag 到 `INPUT_DTYPE_MAP`/`TAG_TO_LIST`，并加入 subset compile 集合 |
| C++ dispatch | `csrc/opus_gemm/opus_gemm.cu` | 新增普通 dispatch 和 tune dispatch，支持 fp32/bf16 输出 |
| C++ header | `csrc/opus_gemm/include/opus_gemm.h` | 新增 `opus_gemm_a8w8_bpreshuffle` 与 tune 入口声明 |
| pybind | `csrc/include/rocm_ops.hpp`, `csrc/pybind/opus_gemm_pybind.cu` | 新增 pybind 宏和注册 |
| Python wrapper | `aiter/ops/opus/gemm_op_a8w8_bpreshuffle.py` | 新增用户侧 API、B shuffle helper、shape guard、tuned CSV lookup |
| Python export | `aiter/ops/opus/__init__.py` | 导出 `gemm_a8w8_bpreshuffle_opus` |
| tuned CSV lookup | `aiter/ops/opus/common.py` | 默认读取 Opus bpreshuffle tuned CSV，并按 `libtype=opus` 过滤 |
| tuned result | `aiter/configs/opus_a8w8_bpreshuffle_tuned_gemm.csv` | 23 个可支持目标 shape 的 Opus winner |
| tests | `op_tests/test_opus_a8w8_bpreshuffle_gemm.py`, `op_tests/test_opus_vs_ck_a8w8_bpreshuffle.py` | 正确性测试和三后端性能对比 |

### 2.3 Python API

```python
from aiter.ops.opus import gemm_a8w8_bpreshuffle_opus

Y = gemm_a8w8_bpreshuffle_opus(
    A,
    B,
    x_scale,
    w_scale,
    out_dtype=torch.bfloat16,
)
```

参数约束：

- `A`: `[M, K]` 或 `[batch, M, K]`，fp8，contiguous。
- `B`: `[N, K]` 或 `[batch, N, K]`，fp8。默认由 wrapper 内部做 K-panel bpreshuffle。
- `x_scale`: `[batch, M, K/128]`，fp32，contiguous。
- `w_scale`: `[batch, N/128, K/128]`，fp32，contiguous。
- `out_dtype`: `torch.float32` 或 `torch.bfloat16`。
- `kernelId`: 可选。显式传入时用于 tune/调试；不传时优先查 tuned CSV，再使用安全 fallback。
- `preshuffled=True`: 表示传入的 B 已经是 K-panel 布局，wrapper 不再重排。

## 3. KernelId 与 shape 约束

### 3.1 当前保留的 Opus 候选

| kernelId | tile MxNxK | 输出 | symbol | 说明 |
|---:|---|---|---|---|
| 3 | `256x256x128` | fp32/bf16 | `opus_gemm_bpreshuffle_512x256x256x128_4x2_16x16x128_1x128x128` | 最初添加的 Opus bpreshuffle kernel |
| 9002 | `128x256x128` | fp32/bf16 | `opus_gemm_bpreshuffle_128x256_512x128x256x128_4x2_16x16x128_1x128x128` | 本轮 tune 新增变体，所有 supported target shape 的 winner |

`kid=9002` 不是 CK 或 CKTile 的 ID，它是 Opus bpreshuffle 的新增 tune candidate。测试脚本通过
`opus_gemm_a8w8_bpreshuffle_tune(..., kernelId)` 强制指定不同 Opus kernelId 做对比。

曾尝试过更窄 N tile 的候选，但当前 pipeline 的 scale-load 逻辑要求 `B_N >= 256`，更窄 N 变体会触发编译期 zero vector size 问题。因此当前只保留 `kid=3` 和 `kid=9002`。

### 3.2 当前 shape guard

当前 Opus bpreshuffle wrapper 会在 Python 侧做 shape guard，避免不安全 shape 进入 GPU kernel：

```text
N % 256 == 0
K % 128 == 0
(K / 128) 必须为偶数
M 必须能被候选 kernel 的 tile_M 整除，即 256 或 128
```

本轮 169 个 unique shape 中，因此有 146 个不支持，主要是两类（举例）：

| Shape MxNxK | 原因 |
|---|---|
| `2048x7168x384`、`256x7168x384` 等所有 `K=384` shape | `K/128=3`，K loop 为奇数 |
| `4x768x7168`、`124x2048x7168`、`180x65536x1536` 等 | M 不是 128 或 256 的倍数 |

即所有 `N=7168, K=384` 组整组被跳过，以及各组中 M 不是 128/256 倍数的小 M shape。
剩余 23 个 `M % 256 == 0`（或部分 `M=384` 等 128 倍数）的 shape 可被 `kid=9002` 覆盖，
但小 M（M=256/384）下性能仍很差，只是能安全跑通。

## 4. Tuning 闭环

### 4.1 正确性 smoke

首次接入或修改 kernel 后，先触发 JIT rebuild 并跑一个小 shape 正确性：

```bash
cd /root/workspace/aiter
PYTHONPATH=/root/workspace/aiter AITER_USE_SYSTEM_TRITON=1 AITER_REBUILD=2 GPU_ARCHS=gfx950 \
python3 op_tests/test_opus_a8w8_bpreshuffle_gemm.py \
  -b 1 -m 256 -n 256 -k 256 --out-dtype bf16
```

也可以显式验证 `kid=9002`：

```bash
cd /root/workspace/aiter
PYTHONPATH=/root/workspace/aiter AITER_USE_SYSTEM_TRITON=1 AITER_REBUILD=2 GPU_ARCHS=gfx950 \
python3 op_tests/test_opus_a8w8_bpreshuffle_gemm.py \
  -b 1 -m 128 -n 256 -k 256 --out-dtype bf16 --kernel-id 9002
```

### 4.2 对目标 shape tune Opus

测试脚本 `op_tests/test_opus_vs_ck_a8w8_bpreshuffle.py` 做两件事：

- `--tune-opus` 时，逐个测试 Opus candidate kids，选最小 `us` 的 Opus winner。
- 同时强制调用 CK 和 CKTile 后端做性能对比，但不会修改 CK/CKTile 的 tuned config。

运行命令：

```bash
cd /root/workspace/aiter
PYTHONPATH=/root/workspace/aiter AITER_USE_SYSTEM_TRITON=1 GPU_ARCHS=gfx950 \
python3 op_tests/test_opus_vs_ck_a8w8_bpreshuffle.py \
  --tune-opus \
  --opus-kids 3,9002 \
  --opus-out-dtype bf16 \
  --csv-out /tmp/opus_a8w8_bpreshuffle_target_tuned_v2.csv
```

目标 shape 集合直接编码在脚本的 `_SHAPE_GROUPS`（按 ck/cktile/asm 来源），由 `_expand_shape_groups()`
去重展开成 169 个 shape；tune 模式下不支持的 shape 会被 `bpreshuffle_kernel_shape_ok()` 自动跳过。
本轮 23 个 supported shape 全部选择了 `kid=9002`。随后将结果覆盖到：

```text
aiter/configs/opus_a8w8_bpreshuffle_tuned_gemm.csv
```

CSV schema 与 aiter/gradlib GEMM tuner 输出保持一致：

```text
gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,
libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw
```

关键字段：

- `libtype=opus`：Opus runtime lookup 只读取 Opus row。
- `scaleAB=True`、`bpreshuffle=True`：区分普通 a8w8 / scale / bpreshuffle。
- `solidx=9002`：全部 23 个 supported target shape 的 tuned winner。
- `outdtype=torch.bfloat16`：本轮性能对比和落库使用 bf16 输出。

### 4.3 Runtime 如何使用 tuned CSV

`gemm_a8w8_bpreshuffle_opus(..., kernelId=None)` 的选择顺序：

1. 调用 `aiter/ops/opus/common.py::lookup_tuned()`。
2. lookup key 包含 `cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle`。
3. 默认 glob 会读取：

```text
aiter/configs/bf16_tuned_gemm.csv
aiter/configs/model_configs/*_bf16_tuned_gemm.csv
aiter/configs/opus_a8w8_bpreshuffle_tuned_gemm.csv
aiter/configs/model_configs/*opus_a8w8_bpreshuffle_tuned_gemm.csv
```

4. 只保留 `libtype == "opus"` 的 row；重复 key 选择 `us` 最小的 row。
5. 如果没有 CSV winner，则走 fallback：优先安全的 `kid=3`，否则尝试 `kid=9002`。
6. 如果传了显式 `kernelId`，则跳过 CSV lookup，用于 tuning 和 debug。

## 5. 性能测试结果

测试数据来自同一轮 `op_tests/test_opus_vs_ck_a8w8_bpreshuffle.py --tune-opus --opus-kids 3,9002 --opus-out-dtype bf16`
（更正后的 shape 集合）。Opus/CK/CKTile 都先对同一 torch reference 做 correctness check（全部 `err=0`），然后计时。
下表只列出可支持的 shape；同组中不支持的 shape 全部被跳过，不计入汇总。

### 5.1 CK 来源组（14 个支持）

| Shape MxNxK | Opus kid | Opus us | CK us | CKTile us | Opus/CK | CKTile/Opus | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `384x768x7168` | 9002 | 279.5 | 90.4 | 206.9 | 3.09x | 0.74x | CK/CKTile 都快于 Opus |
| `32512x768x7168` | 9002 | 440.7 | 415.9 | 793.0 | 1.06x | 1.80x | CK 略快，Opus 明显快于 CKTile |
| `32768x768x7168` | 9002 | 438.9 | 405.8 | 782.7 | 1.08x | 1.78x | CK 略快，Opus 明显快于 CKTile |
| `256x2048x7168` | 9002 | 296.2 | 117.0 | 194.8 | 2.53x | 0.66x | CK/CKTile 都快于 Opus |
| `3840x2048x7168` | 9002 | 352.7 | 176.3 | 328.4 | 2.00x | 0.93x | CK 最快，CKTile 也略快于 Opus |
| `4096x2048x7168` | 9002 | 350.8 | 197.2 | 360.6 | 1.78x | 1.03x | CK 最快，Opus 约等于 CKTile |
| `1024x7168x7168` | 9002 | 460.2 | 174.2 | 309.4 | 2.64x | 0.67x | CK/CKTile 都快于 Opus |
| `3840x7168x7168` | 9002 | 783.0 | 388.7 | 854.4 | 2.01x | 1.09x | CK 最快，Opus 略快于 CKTile |
| `4096x7168x7168` | 9002 | 693.4 | 427.1 | 882.0 | 1.62x | 1.27x | CK 最快，Opus 快于 CKTile |
| `3840x8192x1536` | 9002 | 368.8 | 178.7 | 288.4 | 2.06x | 0.78x | CK/CKTile 都快于 Opus |
| `4096x8192x1536` | 9002 | 343.2 | 184.1 | 290.4 | 1.86x | 0.85x | CK/CKTile 都快于 Opus |
| `256x65536x1536` | 9002 | 405.2 | 128.6 | 206.7 | 3.15x | 0.51x | CK/CKTile 都快于 Opus |
| `3840x65536x1536` | 9002 | 1404.0 | 808.3 | 1696.8 | 1.74x | 1.21x | CK 最快，Opus 快于 CKTile |
| `4096x65536x1536` | 9002 | 1455.5 | 992.1 | 1796.0 | 1.47x | 1.23x | CK 最快，Opus 快于 CKTile |

14 个 supported shape 汇总：

| 后端 | 总耗时 us | 相对 Opus |
|---|---:|---:|
| CK | 4684.4 | CK 比 Opus 快，Opus/CK = 1.72x |
| Opus | 8072.1 | baseline |
| CKTile | 8990.5 | Opus 对 CKTile 仅 1.11x 加速，耗时降低约 10.2% |

结论：这组以中小 M 为主，Opus 明显落后 CK（1.72x）；对 CKTile 也只是整体略优，且在多数小 M shape
（`256x2048x7168`、`384x768x7168`、`1024x7168x7168` 等）上 CKTile 反而更快。只有大 M 的
`32512x768x7168` / `32768x768x7168` 上 Opus 接近 CK。

### 5.2 CKTile 来源组（4 个支持）

| Shape MxNxK | Opus kid | Opus us | CK us | CKTile us | Opus/CK | CKTile/Opus | Opus 对 CKTile 耗时降低 | 结论 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `3840x7168x16384` | 9002 | 1397.1 | 866.3 | 1772.2 | 1.61x | 1.27x | 21.2% | CK 最快，Opus 快于 CKTile |
| `4096x7168x16384` | 9002 | 1400.7 | 944.3 | 1799.4 | 1.48x | 1.28x | 22.2% | CK 最快，Opus 快于 CKTile |
| `15360x7168x7168` | 9002 | 1854.0 | 1386.3 | 3106.6 | 1.34x | 1.68x | 40.3% | CK 最快，Opus 明显快于 CKTile |
| `16384x7168x7168` | 9002 | 1985.9 | 1466.2 | 3254.0 | 1.35x | 1.64x | 39.0% | CK 最快，Opus 明显快于 CKTile |

4 个 shape 汇总：

| 后端 | 总耗时 us | 相对 Opus |
|---|---:|---:|
| CK | 4663.1 | CK 比 Opus 快，Opus/CK = 1.42x |
| Opus | 6637.7 | baseline |
| CKTile | 9932.2 | Opus 对 CKTile = 1.50x 加速，耗时降低约 33.2% |

结论：这组（大 M、大 N/K）是本轮 tune 的主要收益点。Opus tune 后全部快于 CKTile（整体 1.50x），
但仍没有追上 CK。

### 5.3 ASM 来源组（5 个支持，全部小 M）

| Shape MxNxK | Opus kid | Opus us | CK us | CKTile us | Opus/CK | CKTile/Opus | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| `256x768x7168` | 9002 | 278.4 | 95.1 | 198.2 | 2.93x | 0.71x | Opus 慢于 CK 和 CKTile |
| `2048x768x7168` | 9002 | 247.8 | 108.4 | 189.4 | 2.29x | 0.76x | Opus 慢于 CK 和 CKTile |
| `256x7168x7168` | 9002 | 459.8 | 107.8 | 185.8 | 4.27x | 0.40x | Opus 远慢于 CK 和 CKTile |
| `256x7168x16384` | 9002 | 743.9 | 201.9 | 385.1 | 3.68x | 0.52x | Opus 远慢于 CK 和 CKTile |
| `256x8192x1536` | 9002 | 201.6 | 54.0 | 123.7 | 3.73x | 0.61x | Opus 远慢于 CK 和 CKTile |

5 个 shape 汇总：

| 后端 | 总耗时 us | 相对 Opus |
|---|---:|---:|
| CK | 567.2 | CK 比 Opus 快，Opus/CK = 3.41x |
| Opus | 1931.5 | baseline |
| CKTile | 1082.2 | CKTile 也快于 Opus（Opus 约为 CKTile 的 1.78x） |

结论：ASM 组的 supported shape 全是 M=256/2048 的小 M，Opus bpreshuffle 完全不适合替代 small-M/ASM 路径。
该组绝大多数 shape（M 不是 128/256 倍数）本就被跳过。后续需要单独的小 M kernel 或 tail-M 支持。

### 5.4 总体汇总（23 个 supported shape）

| 后端 | 总耗时 us | 相对 Opus |
|---|---:|---:|
| CK | 9914.7 | Opus/CK = 1.68x |
| Opus | 16641.3 | baseline |
| CKTile | 20004.9 | Opus 对 CKTile = 1.20x 加速 |

结论：跨全部 supported shape，Opus 平均仍落后 CK 约 1.68x，只对 CKTile 整体略优（1.20x），
收益集中在 CKTile 来源组的大 M shape。

## 6. 验证记录

已执行的基础验证：

```bash
cd /root/workspace/aiter
python3 -m py_compile \
  aiter/ops/opus/gemm_op_a8w8_bpreshuffle.py \
  aiter/ops/opus/common.py \
  op_tests/test_opus_a8w8_bpreshuffle_gemm.py \
  op_tests/test_opus_vs_ck_a8w8_bpreshuffle.py
```

正确性 smoke：

```bash
PYTHONPATH=/root/workspace/aiter AITER_USE_SYSTEM_TRITON=1 AITER_REBUILD=2 GPU_ARCHS=gfx950 \
python3 op_tests/test_opus_a8w8_bpreshuffle_gemm.py \
  -b 1 -m 256 -n 256 -k 256 --out-dtype bf16
```

显式 `kid=9002` smoke：

```bash
PYTHONPATH=/root/workspace/aiter AITER_USE_SYSTEM_TRITON=1 AITER_REBUILD=2 GPU_ARCHS=gfx950 \
python3 op_tests/test_opus_a8w8_bpreshuffle_gemm.py \
  -b 1 -m 128 -n 256 -k 256 --out-dtype bf16 --kernel-id 9002
```

目标 shape tuning：

```bash
PYTHONPATH=/root/workspace/aiter AITER_USE_SYSTEM_TRITON=1 GPU_ARCHS=gfx950 \
python3 op_tests/test_opus_vs_ck_a8w8_bpreshuffle.py \
  --tune-opus --opus-kids 3,9002 --opus-out-dtype bf16 \
  --csv-out /tmp/opus_a8w8_bpreshuffle_target_tuned_v2.csv
```

## 7. PR 建议说明

建议 PR 里把这次工作描述为：

- 新增 gfx950 Opus `a8w8_scale_bpreshuffle` pipeline 和 runtime entry。
- 新增 Python wrapper `gemm_a8w8_bpreshuffle_opus`，支持内部 B bpreshuffle、fp32/bf16 输出和显式 `kernelId` tuning。
- 新增 tuned CSV lookup，使 `kernelId=None` 时可自动选择 `libtype=opus, scaleAB=True, bpreshuffle=True` 的 tuned winner。
- 新增 `kid=9002` 作为 `128x256x128` tune variant，并为更正后目标 shape 集合中 23 个可支持 shape 落库。
- 新增 correctness test 和 Opus/CK/CKTile 对比 test。

当前限制也需要在 PR 中明确：

- 只支持 gfx950。
- 当前 Opus bpreshuffle 要求 `N % 256 == 0`，`K % 128 == 0`，`K/128` 为偶数。
- 当前只覆盖 `M % 128 == 0` 或 `M % 256 == 0` 的 shape，没有通用 M-tail。
- small-M/ASM 组不建议切到 Opus。
- 更窄 N tile 需要改 pipeline scale-load/tail 逻辑，本轮未纳入。

