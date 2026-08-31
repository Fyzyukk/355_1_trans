# Opus GEMM 重构 — 施工版

> 完整论证见 `opus_gemm_refactor_plan_0810.md`。本文只留「改什么、在哪、别踩什么」。
> 树：`/root/workspace/0810/aiter` @ `35696023`，三 arch：gfx950 / gfx942 / gfx1250。

**总纲**：C++ 现在同时是 kernel 库、调优数据库、启发式策略引擎、显存分配器。砍掉后三个，只留「kernel + 按 kid dispatch」。

---

# 任务一 — splitK workspace 切到 torch

**边界**：`grep -rln "ws_handle\|opus_splitk_ws" csrc/ aiter/` = 22 个存量文件 + 4 个新增。不在清单里的别碰。

## 施工顺序

### Step 0 — heuristic 移植到 python（只加不删）

**为什么必须先做**：python 要分配 workspace 就得事前知道 kid。CSV 命中路径和 `tuned_gemm.py` 路径都知道，只有「CSV miss → C++ `opus_gemm()` → C++ heuristic」这条不知道。

新建 `aiter/ops/opus/heuristic.py`：

| arch | 源文件 | 现状 | 工作量 |
|---|---|---|---|
| gfx1250 | `opus_gemm_heuristic_dispatch_gfx1250.cuh:27` | 已是 `(M,N,K,has_bias)→int kid`，48 行 | 机械翻译，**先做这个** |
| gfx950 | `..._gfx950.cuh:83` | 同上，127 行 | 机械翻译 |
| gfx942 | `..._gfx942.cuh:142` | **返回 launcher 符号名**（`OpusA16W16NoscaleKernel`），157 行 | 先反查成 kid：从 `opus_gemm_common.py` 的 `gfx942_*_kernels_list` **生成**映射，不手写 |

一并搬 gfx942 的 auto-splitK（`gen_instances_gfx942.py:438-462`）。

**C++ 三个 heuristic 头一行不删，`opus_gemm()` 入口保留。** 只加 python 版本 + shadow 对拍脚本 `op_tests/test_opus_heuristic_parity.py`：同一批 shape 跑两边，`(kid, splitK)` 逐条相同。删除留给任务二。

这一步是独立 commit，纯新增，零行为变化。

### Step 1 — kid workspace 元信息

`csrc/opus_gemm/opus_gemm_common.py`，`OpusGemmInstance` 加字段：

```python
ws_includes_batch: bool = True     # gfx1250 置 False
max_split_k: int = 16              # gfx942 的事实值
```

加两个函数：

```python
def kid_workspace_dtype(kid) -> str | None          # 'fp32_t' / 'bf16_t' / None
def kid_workspace_plan(kid, batch, M, N, K, split_k) -> tuple[shape, dtype_str] | None
```

**返回 shape+dtype，不是 byte 数**——typed tensor 才能让 C++ 断言 dtype。

### Step 2 — ABI 单点改动

`csrc/opus_gemm/gen_instances.py` `SPLITK_REDUCE_ABI_MAP` **L69-94**（三个 arch 条目 L73-74 / L80-81 / L90-91，注释 L87）：

```
"ws_arg":  "const opus_splitk_ws_handle* ws_handle"  →  "void* ptr_ws"
"ws_type": "const opus_splitk_ws_handle*"            →  "void*"
```

改这一处，三个 arch 的生成签名同时变。

### Step 3 — 拆 `D_C`

现在 `D_C` 同时是「累加类型」和「workspace 元素类型」，这是 ws 尺寸算错的病根。

```cpp
using D_ACC = fp32_t;   // 寄存器累加，恒 fp32
using D_WS  = ...;      // workspace 元素类型，由 kid 决定
using D_OUT = ...;      // 最终输出
```

`gen_instances_gfx942.py:505-506` 的 `sizeof(typename Traits::D_C)` → `D_WS`。可以独立于其余部分先做。

### Step 4 — 三侧改造 + 删旧机制

见下方文件清单。

---

## 文件清单

### A. traits — 删结构体 + 改 kargs

| 文件 | 行 | 动作 |
|---|---|---|
| `include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh` | struct 593 / 注释 603 / kargs 611 | 删 struct；kargs → `void* ptr_ws` |
| `include/gfx942/opus_gemm_traits_a16w16.cuh` | struct 106 / helper 114-128 / kargs 137 | 同上。`opus_splitk_ws_ptr<>()` helper **整个删** |
| `include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh` | 头注释 7 / struct 45 / 注释 55,60 / kargs 64 | 同上 |

三份 struct 靠 `#ifndef` 互防，guard 一起删。

### B. pipeline — 每文件 1 行，数值算法零改动

| 文件 | 行 |
|---|---|
| `include/gfx950/..._flatmm_splitk_gfx950.cuh` | 266 |
| `include/gfx1250/..._cluster_tdm_splitk_ws_gfx1250.cuh` | 386 |
| `include/gfx1250/..._clusterlaunch_tdm_splitk_ws_gfx1250.cuh` | 454 |
| `include/gfx942/..._em3en4_lds1_pgr2_sk.cuh` | 620 |
| `include/gfx942/..._kbuf1.cuh` | 74 |
| `include/gfx942/..._kbuf2v.cuh` | 77 |
| `include/gfx942/..._kbuf2v_bk128.cuh` | 114 |
| `include/gfx942/..._quad_mfma32_kbuf1.cuh` | 97 |

统一改成 `reinterpret_cast<D_WS*>(kargs.ptr_ws)`。非 splitK kernel 不做机械重写。

### C. reduce — 改签名 + 取指针

| 文件 | 行 |
|---|---|
| `include/gfx950/splitk_reduce_gfx950.cuh` | 65, 72, 86 |
| `include/gfx942/splitk_reduce_gfx942.cuh` | 8, 15, 50, 197, 205, 212, 220, 228, 260 ← **最重，有 baseline/OOB 两套 + 转发层** |
| `include/gfx1250/splitk_reduce_gfx1250.cuh` | 7, 9, 19, 26, 36, 176 |

### D. codegen — 删 grow 块

| 文件 | 行 | 备注 |
|---|---|---|
| `codegen/gen_instances_gfx950.py` | 1317-1357 | extern 声明 + capture 探测 + grow |
| `codegen/gen_instances_gfx942.py` | 495-528 | 含 500-502 extern、505-506 `sizeof(D_C)`、525 sync、528 device handle |
| `codegen/gen_instances_gfx1250.py` | 289-326 + 头注释 7 | 含 299 无 batch 公式、318 sync、325-326 device mirror |

三个不同的 `ws_bytes` 公式一起消失，统一到 python 的 `kid_workspace_plan()`。

改成：接 workspace tensor → 校验 → 取 `data_ptr()`。

### E. 入口 / 绑定 / 框架

| 文件 | 行 | 动作 |
|---|---|---|
| `csrc/opus_gemm/opus_gemm.cu` | registry 325-429 / include 27-29 / 入口 267 | 删 registry 三函数 + `opus_gemm_workspace_init`；入口加末位 `std::optional<aiter_tensor_t> workspace = std::nullopt`；加三条 `AITER_CHECK` |
| `include/opus_gemm.h` | 44-45 | 加 workspace 参数；删 init 声明 |
| `csrc/include/rocm_ops.hpp` | 285 / 296-302 | 前者加参数；后者整段删 |
| `csrc/pybind/opus_gemm_pybind.cu` | 18 | 删宏调用 |
| `include/opus_gemm_common.cuh` | 24 注释 | 删对 handle 的引用 |
| `csrc/opus_gemm/opus_gemm_common.py` | 27-68 | Step 1 的字段和函数 |
| `aiter/ops/opus/gemm_op_a16w16.py` | binding 73-88 / tune 166 / `_opus_gemm_bf16_dispatch` 264-288 / init 528-529 | 加 workspace 参数、调用前分配；**`_opus_gemm_bf16_dispatch` 改走 python heuristic**，不再进 C++ `opus_gemm()` |
| `aiter/ops/opus/__init__.py` | 44 / 51 / 57 | 删导出（或留空壳） |
| `aiter/tuned_gemm.py` | **41-125 整段删** + 调用点 623 | 整个 prewarm 机制 |
| `csrc/opus_gemm/opus_gemm_tune.py` | 1091 / 1140 | tuner 也要分配 workspace |
| `csrc/opus_gemm/README.md` | 3 处 | 更新为 direct-pointer ABI |

### F. 新增

- `aiter/ops/opus/heuristic.py`（Step 0）
- `op_tests/test_opus_heuristic_parity.py`
- `op_tests/test_opus_workspace.py`
- `op_tests/test_opus_graph.py`

---

## C++ 保留的三条断言

```cpp
AITER_CHECK(!kid_needs_ws || workspace.has_value(), ...);
AITER_CHECK(ws.dtype() == kD_WS_dtype, ...);      // ← 最关键
AITER_CHECK(ws_elems_needed <= ws.numel(), ...);
```

dtype 那条是关键：#4246 挂机的原因是「写 fp32 的 kernel 拿到按 bf16 算大小的 buffer」——numel 是对的，元素宽度是错的，只比 numel 查不出来。

`aiter_tensor_t` 有 `numel()` / `element_size()`，**没有 `nbytes()`**。

python 侧用 typed `torch.empty(shape, dtype=...)`，不用 `uint8` 扁平 buffer——扁平 buffer 传过去只剩 `void*`，上面那条断言就没了。

HIP graph 不需要特殊处理：torch caching allocator 在 capture 期间返回 replay-stable 地址，这正是 #4246 能删掉 prewarm 的原因。

---

## 四个坑

**坑 1 — ws dtype 读 kid 自己的属性。** gfx942 有 bf16ws 变体（`opus_gemm_common.py:569` / `:598`）。硬编码 `sizeof(float)` 对这些 kid 只是浪费；反向（按 bf16 分配却写 fp32）就是越界写 + 挂机。

**坑 2 — batch 项 per-arch 不同。**

| arch | 公式 | 出处 |
|---|---|---|
| gfx950 | `split_k * batch * padded_M * padded_N * sizeof(float)` | `gen_instances_gfx950.py:1335` |
| gfx942 | `... * sizeof(Traits::D_C)` ← 可能 bf16 | `gen_instances_gfx942.py:505` |
| gfx1250 | `split_k * padded_M * padded_N * sizeof(float)` ← **无 batch** | `gen_instances_gfx1250.py:299` |

挂到 `ws_includes_batch` 字段，不要在分配器里 `if arch == "gfx1250"`。

**坑 3 — launcher 会 clamp split_k**（gfx950:1302 / gfx942:481 / gfx1250:277）。按**未 clamp 的上界**分配，不要在 python 复刻 clamp 循环。

**坑 3 特例（最隐蔽的越界点）——gfx942 的 `splitK <= 0` 是「launcher 自己挑」，不是「不切 K」。** `gen_instances_gfx942.py:438-462`：传 0 时按 CU 数自动挑，clamp 到 `[1,16]`。而 `opus_gemm.cu:167/171` 传的正是 0。按 splitK=1 分配、kernel 按最大 16 跑 → 越界。**另两个 arch 没这个语义，只测 gfx950/gfx1250 完全看不见。**

处理：Step 0 之后 python 已经知道 kid，顺手也决定 splitK，不再传 0。分配上界 = python 自己算的值。

**坑 4 — `splitK` 没有上界硬夹。** 现在没人拦 `splitK=100000`，切到 torch 后直接是一次巨额 `torch.empty`。用 `max_split_k` 在 python 侧校验。

---

## 验收（九项，都不能省）

| 项 | 要求 |
|---|---|
| 机械 | 全仓 grep `ws_handle` / `opus_splitk_ws_` / `SplitkWsRegistry` / `workspace_init` 零命中（除 deprecation 空壳） |
| 数值 | `op_tests/test_opus_a16w16_gemm.py` 全绿 |
| **三 arch** | 各跑一遍。gfx1250 batch 项不同、gfx942 ws dtype 不同，只测一个必漏 |
| **HIP graph** | capture/replay + golden 对拍。这是原机制存在的全部理由 |
| **双 stream** | 两 CPU 线程各驱动一 stream 同时跑 splitK（原 TBO 场景） |
| **ws dtype** | 同时覆盖 fp32-ws 和 bf16-ws kid——坑 1 只在后者暴露 |
| **契约负例** | 正好够大 / **少 1 元素** / 错 dtype / 错 device / noncontiguous / caller 自带 ws 复用。三条 `AITER_CHECK` 每条都要有负例 |
| **heuristic 等价** | 三 arch × 一批 CSV-miss shape，python 与 C++ 选出的 `(kid, splitK)` 逐条相同 |
| **bias dtype** | gfx950/942 的 `bias.dtype == Y.dtype`，**以及 gfx1250 的 `bf16 out + fp32 bias`**（见任务二警告） |
| **splitK 上界** | 荒谬大的值在 python 侧被拒，而不是发起巨额 `torch.empty` |

`opus_gemm_workspace_init` 是公开 API 的破坏性删除。仓内只有 `tuned_gemm.py` 用（同批删），下游 vLLM/sglang 可能有 → python 留空壳 + `DeprecationWarning` 一个 release，C++ 直接删。

**规模**：+135 行（Step 0 的 +380 在任务二删 C++ 三个头时还回来）。

---

# 任务二 — 入口重构

约 25 个文件。与任务一只有 6 个文件重叠（`opus_gemm.cu` / `opus_gemm.h` / `rocm_ops.hpp` / `gen_instances.py` / `opus_gemm_common.py` / `gemm_op_a16w16.py`）且段落不同，**可并行**。

## ⚠️ 三个 KEEP（最容易误删）

| 保留什么 | 位置 |
|---|---|
| 三个 `opus_a16w16_tune_dispatch_gfx*<>` | gfx950:122-158 / gfx942:108-140 / gfx1250:40-70 |
| `gen_a16w16_tune_lookup()` | `gen_instances.py:537-597`。**改名不删** |
| subset-compile 集合 | `gen_instances.py:1044` / `1103` / `1107`。**一行都不能动** |

**两张 codegen 表，只能删一张：**

| 头文件 | 键 | 生成函数 | 处置 |
|---|---|---|---|
| `opus_gemm_lookup.h` | **(M,N,K)** → launcher | `gen_instances.py:430 gen_lookup_dict()` | **删** |
| `opus_gemm_a16w16_tune_lookup.h` | **kid** → launcher | `gen_instances.py:537 gen_a16w16_tune_lookup()` | **保留** |

文件名只差 `_a16w16_tune`，两个生成函数在文件里紧挨着（430 / 537）。

**先单独一个 commit 改名**：`opus_gemm_a16w16_tune_lookup.h` → `opus_gemm_a16w16_kid_dispatch.h`，`gen_a16w16_tune_lookup()` → `gen_a16w16_kid_dispatch()`。改完之后 `grep -rn lookup csrc/opus_gemm` 的命中就都是该删的。

**subset-compile 的连坐风险**：`S = (csv_kids | sidecar_kids | HEURISTIC_DEFAULT_KIDS) & valid_kids`。CSV 在这里决定「编译哪些 kid 进 .so」，与「运行时怎么查表」是两回事，但共用同一段读取代码。删烘焙时极易连坐 → CSV 里的 kid 编不进 .so，运行时 `AITER_CHECK(kid not found)`。加 CI：对每个 CSV opus 行断言 `solidx ∈ S`。

## 4.1 kid 元信息去重

`opus_gemm_common.py` 加查询 API：

```python
def kid_info(arch, family, kid)     # 身份是三元组，不是裸整数 —— #4320 有跨 family 重复 kid
def kid_is_splitk(...)
def kid_supports_bias(...)
def kid_arch(kid)
def kid_exceeds_4g(kid, tensor)
```

删副本：
- `gemm_op_a16w16.py:289-290` `_SPLITK_KID_MIN/MAX`（注释自己承认是 documentation anchor）
- `aiter/ops/opus/common.py:197-203` `_MONO_TILE_KID_TILES`（注释写着「Must stay in lock-step with…」）
- `opus_gemm_arch_gfx950.cuh:102-110` `kSplitkKidMin/Max` + `kid_is_splitk`
- `opus_gemm.cu:186-265` 全部 kid 分类常量与谓词（80 行）

后两条依赖 4.2 完成。

**4 GiB 守卫四处合一**：`opus_gemm_arch_gfx950.cuh:196` + `:233` + `opus_gemm_arch_gfx1250.cuh:79` + `opus_gemm_tune.py:391`。

顺手修正算法——现有四处都按 `numel * element_size` 判断，但约束的是实际 storage span。A/B 允许 padded lda 时 span > numel，按 numel 算会**低估、漏判**：

```python
span = 1 + sum((size[d] - 1) * stride[d] for d in range(t.dim()))
exceeds = span * t.element_size() >= 2**32
```

**验证**：`gen_instances.py` 产物 diff 为空（这一步不该改变任何生成代码）。

## 4.2 dispatch 下沉 + 入口收敛

### 新建 python

- `aiter/ops/opus/heuristic.py` — 任务一已交付
- `aiter/ops/opus/dispatch.py` — `resolve_kid()`：explicit → CSV → heuristic
- `aiter/ops/opus/layout.py` — 合并 `_validate_and_reshape` + `_check_a16w16_tune_layout`

三条选择规则：

1. **CSV 命中的 kid 也要过 shape 门。** `gemm_op_a16w16.py:498` 现在无条件信任 solidx，只对 mono-tile 特判。拒绝时 **kid 和 splitK 一起丢**（tuned row 的 splitK 是配着那个 kid 调出来的）。
2. **explicit kid 不兼容抛错，不静默换。** C++ 现在行为已经是对的（`opus_gemm_arch_gfx942.cuh:120`），保持住——别被 #4320 的默认 kid 回退带偏。
3. **kid 必须在本次 build 实际编译的集合里。** 两个 JSON 语义不同：`compiled_kids_opus.json`（sidecar，跨 rebuild 累积的愿望集）vs `{working_path}/compiled_kids.json`（本次结果）。selector 只信后者。首次 JIT 时后者不存在，**不能默认「catalog 里所有 kid 都可用」**。

**不要给 `resolve_kid` 加 `functools.cache`**——真实 key 还含 device/cu_num（`lookup_tuned` 内部就在读 `multi_processor_count`）、batch、layout。漏项的无界 cache 在多卡进程里跨设备串结果。而且不省什么：`lookup_tuned` 已 `lru_cache`。

### C++ 删除

| 位置 | 内容 | 行 |
|---|---|---|
| `gen_instances.py:430-535` | `gen_lookup_dict()` | 108 |
| `gen_instances.py:855` | `get_tune_dict()` | ~45 |
| `gen_instances.py:1128-1175` | CSV 烘焙块 | ~45 |
| `opus_gemm_heuristic_dispatch_gfx1250.cuh` | 整个 | 48 |
| `opus_gemm_heuristic_dispatch_gfx942.cuh` | 整个 | 157 |
| `opus_gemm_heuristic_dispatch_gfx950.cuh` | 只删函数体；`OpusA16W16NoscaleKernel` typedef 和 `opus_gfx950_detail::*` **被另两 arch 复用，先挪到 `opus_gemm_common.cuh`**，然后整个文件删 | 127 |
| `opus_gemm_arch_gfx950.cuh:173-248` | `opus_dispatch_a16w16_gfx950<>` 两个特化 | 86 |
| `opus_gemm_arch_gfx942.cuh:66-106` | 同 | ~50 |
| `opus_gemm_arch_gfx1250.cuh:76-134` | `check_shape_4g` + dispatch | 59 |
| `opus_gemm.cu:55-81` | arch router | 28 |
| `opus_gemm.cu:186-265` | kid 分类 | 80 |
| `opus_gemm.cu:160-177` | `opus_gemm()` bf16 分支 | 18 |

### 最终 C++ 入口（3 → 2）

```cpp
void opus_gemm_a16w16_tune(XQ, WQ, Y, bias, kernelId, splitK, workspace);
void opus_gemm_a8w8_tune  (XQ, WQ, Y, x_scale, w_scale, kernelId);
```

`opus_gemm.cu` 从 431 行降到 ~150 行。

fp8 分支（`opus_gemm.cu:133-158`，硬编码单 launcher、无 tuned 表、gfx950-only）抽成 `opus_gemm_a8w8_tune`，`opus_gemm()` 整个删（待确认下游）。

## 4.3 dtype / layout

### ⚠️ bias dtype 不是全仓统一规则

| arch | 规则 | 出处 |
|---|---|---|
| gfx950 / gfx942 | `bias.dtype() == Y.dtype()` | `gen_instances.py:401` 共享 `BIAS_HOST_VALIDATE` |
| **gfx1250** | `bias.dtype() ∈ {fp32, Y.dtype()}` | `gen_instances_gfx1250.py:155`，**明确绕开共享片段** |

gfx1250 能多支持一种是因为 main kernel 恒写 fp32 workspace，reduce 在 fp32 域折 bias 再 cast。它为此专门多实例化了一个 reduce 模板（`gen_instances_gfx1250.py:53`：`splitk_reduce_kernel_gfx1250<16,64,__bf16,true,float,true>`），launcher 按 `bias_is_fp32_` 分派（`:364`）。

**记精确的 `(out, bias)` 组合对**，不是两个 dtype 列表做笛卡尔积——后者会同时错误放宽 gfx950/942 和错误收紧 gfx1250。这是重构中最容易丢的既有能力。

反方向也要拦：不能顺手把 fp32 bias 推广到另两个 arch（那需要新增 reduce 实例，属于新增能力）。

### ⚠️ A/B 的 padded leading dim 不能收紧

现有 `_check_a16w16_tune_layout`（`gemm_op_a16w16.py:131-152`）刻意区分：

```
A/B:  stride[2] == 1  且  stride[1] >= K  且  (batch == 1 或 stride[0] == rows * stride[1])
Y:    stride == (M*N, N, 1)   严格
```

A/B 放宽是因为 launcher 把真实 leading stride 当 lda 传下去，kernel 用它同时做寻址和 gmem buffer bound——「2880 宽的 tensor 存在 lda=3072 上」是被正确服务的。合并两个校验函数时按 Y 的严格规则统一会拒掉合法输入。

### 其他

- `opus_gemm.cu:289-322` 的 dtype if-else 树 → 表驱动
- **补一个现在缺的门**：gfx1250 cluster_tdm kid 只支持 `batch == 1`（`gen_instances_gfx1250.py:337` 注释确认），python 侧目前不检查
- `BIAS_HOST_VALIDATE` 保留（tuner 和 C++ 直调不走 python 校验），但收敛成共享片段

## 4.4 三 arch 遗留差异

| 差异 | gfx950 | gfx942 | gfx1250 | 处理 |
|---|---|---|---|---|
| heuristic 返回值 | kid | **符号名** | kid | 改 kid |
| ws dtype | fp32 | **有 bf16ws** | fp32 | `kid_workspace_dtype()` |
| ws 含 batch | 是 | 是 | **否** | `ws_includes_batch` |
| batch 支持 | 多 | 多 | **仅 1** | 进 `kid_rejects_shape` |
| splitK 语义 | 正常 | **`<=0` = auto** | 正常 | auto 搬 python |
| bias dtype | == out | == out | **多支持 fp32** | 组合对 |
| 4g_safe kid | 有(+5000) | 无 | 无 | 保持 |

## 4.5 顺手修的 bug

`aiter/ops/opus/common.py:_load_tuned_dict()`（L102-162）glob 全部 `model_configs/*_bf16_tuned_gemm.csv`，concat 后按 `us` 升序 first-wins。key 是 9 元组，**不含模型标识**——跑 A 模型时若 B 模型 CSV 有同 shape 且更快的行，会命中 B 的配置。

| | 用途 | 读哪些 CSV |
|---|---|---|
| build-time（`--tune_files`） | 决定编译哪些 kid | **取并集** |
| runtime（`lookup_tuned`） | 选这次用哪个 kid | **只读框架选定的那个**，不 glob |

另外 key 里应加 `gfx`（CSV schema 已有该列，`_KEY_COLUMNS` 没用）。

## 4.6 验收

| 项 | 要求 |
|---|---|
| **kid 选择等价性** | 三 arch × 全 CSV 行 + 一批 miss shape，重构前后 `(kid, splitK)` **完全相同**。写成脚本不是抽查。这是唯一能证明「没改行为」的手段 |
| codegen 回归 | 加进 `op_tests/test_gemm_codegen.py`：① 产物无 `hipMalloc/hipFree/hipHostMalloc` ② 不生成 runtime shape lookup ③ kid→launcher 映射唯一 ④ 二次生成 diff 空 |
| 编译集合 | 每个 CSV opus 行断言 `solidx ∈ S` |
| 结构 | `opus_gemm.cu` ≤ 160 行；文件内无 CSV / heuristic / hipMalloc / shape 策略 |
| 副本数 | kid 语义全仓一份；`opus_gemm_lookup.h` 不存在；`opus_gemm_a16w16_kid_dispatch.h` **仍存在** |
| bias 矩阵 | 含 gfx1250 的 `bf16 out + fp32 bias` |
| 性能 | 小 shape（M≤64）端到端 host 开销，确认 dispatch 下沉无可观测 CPU 回退 |

**规模**：−920 行。

---

# PR 审查命令

```bash
rg -n "hip(Malloc|Free|HostMalloc)"                csrc/opus_gemm   # 任务一后零命中
rg -n "ws_handle|opus_splitk_ws_|SplitkWsRegistry" csrc/opus_gemm   # 任务一后零命中
rg -n "workspace_init|prewarm"  csrc/opus_gemm aiter/ops/opus aiter/tuned_gemm.py
rg -n "opus_gemm_lookup|heuristic_dispatch"        csrc/opus_gemm   # 任务二后零命中
rg -n "kid_dispatch"                               csrc/opus_gemm   # 任务二后【必须有命中】
rg -n "kid\s*[<>]=?\s*[0-9]"    aiter/ops/opus csrc/opus_gemm       # kid 区间副本
rg -n "sizeof\(.*D_C\)"                            csrc/opus_gemm   # D_C 歧义残留
rg -n "torch\.empty\(.*uint8"                      aiter/ops/opus   # 无类型 workspace
```

第 5 条是**反向**检查：任务二后如果它也零命中，说明「两张表删错一张」的事故已经发生了。

不要求绝对零命中（deprecation 空壳、历史注释、#4320 的 BMM 可能保留文本），要求每个命中都能按 PR 范围解释。

---

# 上游 PR

| PR | 内容 | 与本次 |
|---|---|---|
| **#4246** `gfx1250 opus gemm splitk fuse` | (a) `4ac1fa8` workspace→torch（**仅 gfx1250**）<br>(b) 新 `_fuse` pipeline | 只抄 (a)，**推广到三个 arch**。(b) 无关 |
| **#4320** `opus fp8 mxscale BMM for gfx950`（zoe） | 新增 BMM family，分层已经做对 | **任务二的样板**，不是 workspace 参考（它整个没 workspace 参数） |

**#4246 注意**：不能只 cherry-pick `4ac1fa8`——它最初按默认 bf16 分配 workspace，是后续 `86d1eba` 才改成按 kid 真实 ws dtype/shape 定尺寸并删掉 prewarm。要移植的是两者合起来的最终不变量。

**fuse 并没有消掉 workspace**：非最后一个 split 仍要把 fp32 partial cast 成 ws dtype 存进 GL2，省的是 reduce kernel 的 launch。

**#4320 注意**：
- 顶层 `opus_bmm.cu` 没有 workspace 参数，但它 codegen 出的 BMM split-K launcher **仍有 `opus_splitk_ws_get` / `hipMalloc` / `hipFree`**。若它先合，任务一的 DoD「模块内 hipMalloc 归零」不再自动成立 → 要么同批迁 BMM（partial 布局与 a16w16 不同，需单独 ws plan），要么范围显式限定 a16w16 并在审查命令加 `--glob '!*bmm*'`
- 它现有 tuned CSV 78 行 `splitK` 全是 1，跑 tuned 表触达不到 allocator 路径，要测必须构造显式 `splitK>1`
- 它已有**跨 family 重复的 kid** → kid 身份必须是 `(arch, family, kid)`，裸整数表在它落地当天失效
- 它的 `opus_bmm_a8w8_mxscale_tune_dispatch()` 对 unknown kid **静默回退默认 kid** —— 这条不要照搬

**排期**：#4320 先合（新增模块、改动集中）→ 任务一 → 任务二。反过来会让 #4320 整个重写一遍适配。

---

# 待确认

1. #4246 怎么处理——倾向把 workspace-only 部分拆出来单独先合
2. 任务一是否接受 Step 0 的 +380 行（不接受的话只剩两个都不推荐的方案）
3. `opus_gemm()` 删还是留 deprecated 薄壳（取决于下游有无直调 pybind）
4. a8w8 是否一并改造成 `opus_gemm_a8w8_tune`（它现在无 tuned 表、单 launcher、gfx950-only）
5. #4320 先合时任务一的 DoD 是否覆盖它的 BMM

---

# Bonus — 新 kernel 接入 skill

**在任务二完成后再写**，否则只会固化现在这套要被删掉的 pattern。

`.claude/skills/opus-kernel-integration/SKILL.md`，核心内容：

**要改的文件（6 个，按顺序）**：pipeline → traits → `opus_gemm_common.py` 登记 kid → codegen emit → （仅当要进兜底）`heuristic.py` → test

**不需要改的**（减少冗余输出的关键）：`opus_gemm.cu`、`opus_gemm.h`、`rocm_ops.hpp`、`opus_gemm_pybind.cu`、任何 python 入口。

**kid band**（脚本从 catalog 读取校验，不写死在 prose 里）：
```
gfx950  [0,10000)      splitk [200,300) / persistent [300,316) / mono [1400,1500)
                       nooob +1000, cpol +2000/3000/4000, 4g_safe +5000
gfx942  [10000,20000)
gfx1250 [20000,21000)
```

**注释预算**：只在偏离既有 pattern 处注释；禁止注释掉的旧代码；禁止 changelog 式注释（现状反例 `opus_gemm.cu:325-340` 那 16 行）；新文件头注释 ≤10 行。

**反模式**（每条指向仓内真实位置）：C++ 里加 shape→kernel 策略、C++ 里 `hipMalloc`、复制 kid 区间常量、为新 kernel 加新顶层 pybind 入口、新 arch 复制 `check_shape_4g`（现已有 4 份）。
