# 任务一执行清单 — splitK workspace 切到 torch

> 基线 `ca68b4f3`（本文所有行号在此 commit 核验）。分支 `splitk_to_torch`。
> #4246 (`25dd6281`) / #4320 (`ec93aadf`) 均未合入，树中无 `opus_bmm`。
> 背景论证见 `opus_gemm_refactor_0810_working.md`。

**注意树已变更**：`opus_gemm_refactor_plan_0810.md` 的行号基于 `35696023`，落后 236 个 commit，已失效。
最大的一处：gfx942 头文件搬进了 `include/gfx942/a16w16/` 子目录。

---

## 范围：a16w16 = 当前全部有 workspace 的 family

不是范围收窄。全仓 family 盘点（`opus_gemm_common.py`）：

| family | arch | split-K / workspace |
|---|---|---|
| `a16w16` 及约 20 个变体（`_flatmm_splitk` / `_kbuf*_sk` / `_persistent` / `_mono_tile` / `_cluster_tdm_splitk_ws` / `_clusterlaunch_tdm_splitk_ws` / `_wave_k_coop` …） | gfx942 / gfx950 / gfx1250 | **有** |
| `a8w8` / `a8w8_scale` | gfx950 | 无 |
| `a8w8_blockscale_bpreshuffle_singlebuf`（kid 11000） | gfx942 | 无 |

已核验：`rg "splitk\|split_k" include/gfx942/a8w8/` 零命中；gfx950 两个 a8w8 pipeline
也不含 ws。**opus 里不存在 a4w4 / mxfp4**（MoE 的 `a16w4` 是另一个模块）。

**三个 arch 全覆盖** —— `csrc/opus_gemm/include/` 下只有 gfx942 / gfx950 / gfx1250。

所以任务一覆盖了今天所有需要 workspace 的东西。a8w8 不在范围内不是遗漏，
是它根本不分配 workspace，没有东西可迁。

**但接口必须为将来的 family 留好**：第 1 步的查询一律走 `(arch, family, kid)`，
`kid_workspace_dtype()` 对无 ws 的 family 返回 `None`。将来 a8w8 或新 family 加了
split-K，走同一条路，不用回头改架构。

反例就在眼前：kid 11000 是 a8w8，却落在 gfx942 的 `[10000,20000)` band 内，
且 `gfx942_kernels_list`（`:741`）把它和 a16w16 合并在一张表里。
**任何"按 kid 区间判断要不要 workspace"的写法今天就是错的。**

---

## 分两个提交

```
A0  让 python 在 launch 前知道 kid          （纯新增，C++ 不删）
A1  python 按 kid 分配 workspace + 删 C++ allocator   （原子，第 5-11 步不可拆）
```

不能从改 pipeline 或删 registry 起手。CSV miss 路径现在在 C++ 内部才选出 kid，
没有 A0，python 不知道 workspace 的 dtype 和尺寸。

---

# A0

## 第 1 步 — `opus_gemm_common.py` 加查询接口

文件：`csrc/opus_gemm/opus_gemm_common.py`

保留现有 `OpusGemmInstance` 和所有实例表，不搬家不复制。加：

```python
ws_includes_batch: bool = True    # gfx1250 置 False
max_split_k: int = 16
```

```python
def kid_info(arch, family, kid)                              # 身份是三元组
def kid_workspace_dtype(arch, family, kid) -> str | None     # 'fp32_t'/'bf16_t'/None
def kid_workspace_plan(arch, family, kid, batch, M, N, K, split_k)
    -> tuple[shape, dtype_str] | None                        # 返回 shape+dtype，不是字节数
```

**必须带 family**。gfx942 的 `[10000,20000)` 已经不是纯 a16w16 了 ——
`opus_gemm_common.py:734` 有 `gfx942_a8w8_kernels_list = {11000: ...}`，
`gfx942_kernels_list`（`:743`）把它和 a16w16 合并在一起。selector 不过滤
family 会把 A8W8 实例混进 a16w16 候选集。

**别现在就投影十几类字段**。任务一只用得上 ws dtype / ws 含不含 batch /
block_m,n / max_split_k。dtype capability、addressing mode 那些留给任务二。

**不新建 `aiter/ops/opus/_catalog.py` 转发层**。`jit/core.py` 已做集中
`sys.path.insert`，`from csrc.X import Y` 在 `ops/sampling.py` / `ops/attention.py`
是现成 pattern，`setup.py` 也把 csrc copytree 进 wheel。直接 import。

验收：`gen_instances.py` 产物 diff 为空。

## 第 2 步 — gfx1250 heuristic 移植

新建 `aiter/ops/opus/heuristic.py`（三个 arch 一个文件，别拆包）。

源：`include/gfx1250/opus_gemm_heuristic_dispatch_gfx1250.cuh:27`
```cpp
inline int opus_a16w16_heuristic_kid_gfx1250(int M, int N, int K, bool has_bias)
```
全文 48 行，已经是 `(M,N,K,has_bias) → kid`，逐字翻译。

**从这个起手**，另两个都比它复杂。移植时不调策略，只搬。

## 第 3 步 — gfx950 heuristic 移植

源：`include/gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh:83`
```cpp
inline int opus_a16w16_heuristic_kid_gfx950(int M, int N, int K, bool has_bias = false)
```
全文 127 行，同样已是返回 kid。

## 第 4 步 — gfx942 heuristic + auto splitK 移植

源：`include/gfx942/opus_gemm_heuristic_dispatch_gfx942.cuh:142`
```cpp
inline OpusA16W16NoscaleKernel opus_a16w16_heuristic_dispatch_gfx942(
    int M, int N, int K, int /*batch*/, bool has_bias = false)
```
全文 157 行。**返回 launcher 函数指针，不是 kid**（`L16` 的 typedef，
`L26-39` 声明了 14 个符号）。

要先建 symbol → kid 反查。**从 `opus_gemm_common.py` 的 kernel list 生成，
不手写第二张表**。codegen 时多吐一个 JSON 副产品记录映射，测试读 JSON。

一并移植 auto splitK，`codegen/gen_instances_gfx942.py:470-477`：
```
tiles_mn  = ceil(M/B_M) * ceil(N/B_N) * batch          # <=0 时置 1
target_wg = cu_num * (2 if kernel_tag.endswith("_p1") else 1)
split_k   = clamp(ceil(target_wg / tiles_mn), 1, 16)
```
（`cu_num` 取不到时 fallback 64，见 `:468`）

**只移植这个初值，不移植后面的 clamp 循环**（`:480-494`）。clamp 留在 C++，
python 按未 clamp 的上界分配。

## 第 4b 步 — 把 gfx942 bf16ws 转跳规则搬进 selector

**这一步不能推到 A1。** 详见下方坑 1b。

python selector 里加：

```python
_GFX942_EXACT_N = {64, 128, 256, 384, 512, 1024, 2048}   # 来自 EXACT_N_ROWBLOCK_REDUCE_CONFIGS
_GFX942_BF16WS_REDIRECT = {10210: 10200, 10213: 10203}   # N 不合法时改选
_GFX942_BF16WS_STRICT    = {10216}                       # N 不合法时拒绝
```

**这个集合现在有两份，而且第二份是错的 —— 顺手一并修掉（见坑 1c）。**
`_GFX942_EXACT_N` 应当是**唯一**定义处，codegen 和 tuner 都从这里 import，
不允许再出现第三份字面量。

选中 bf16ws kid 且 `N not in _GFX942_EXACT_N` 时：
- 10210/10213 → 换成 redirect 目标，**同时丢弃 tuned 的 splitK**（它是配着原 kid 调的）
- 10216 → explicit 报错；tuned/heuristic 则退回下一候选

A0 阶段这只是 selector 内部逻辑，C++ 转跳代码还在，两边结果一致，
所以第 5 步的对拍能验证它搬对了。

## 第 5 步 — 对拍测试

新建 `op_tests/test_opus_heuristic_parity.py`。

三 arch × 一批 CSV-miss shape，python 与旧 C++ 选出的 kid 逐条相同。

gfx942 要**拆成两个对拍**：旧 C++ selector 只返回 launcher 不返回 splitK，
所以 kid 和 splitK 分开比。

额外断言：python heuristic 的所有可能返回值都在本次 build 保证编译的
`HEURISTIC_DEFAULT_KIDS` 里。

**C++ 三个 heuristic 头一行不删，`opus_gemm()` 入口保留。** 删除是任务二的事。

## A0 提交前

- 三 arch CSV hit/miss corpus 的 python kid 与 C++ 一致
- gfx942 auto splitK 初值一致
- codegen 产物无变化
- **workspace 一行没动**

---

# A1（第 6-12 步原子，中间可以编不过，但不拆成可合入提交）

## 第 6 步 — 拆 `D_C`

现在 `D_C` 同时当累加类型和 workspace 元素类型，这是 ws 尺寸算错的病根。
gfx942 已经有 `D_WS` 的雏形（`opus_splitk_ws_ptr<D_WS>`，
`include/gfx942/a16w16/opus_gemm_traits_a16w16.cuh:114`），但
`gen_instances_gfx942.py:521` 还在用 `sizeof(typename Traits::D_C)` 算尺寸。

三份 traits 里明确：
```cpp
using D_ACC = fp32_t;   // 寄存器累加，恒 fp32
using D_WS  = ...;      // workspace 元素类型，由 kid 决定
using D_OUT = ...;      // 最终输出
```
workspace 容量、pointer cast、reduce 输入只准引用 `D_WS`。

这步和第 7 步无依赖，可以先做。

## 第 6b 步 — 两张 launcher table（不要给所有 launcher 机械加参）

原来我写的是「给 `opus_gemm_a16w16_tune` 加末位 optional workspace」。
改成**两套签名两张表**，非 split-K 的 ABI 一行不动：

```cpp
// 非 split-K：保持现状，不加参数
using OpusA16W16Kernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    std::optional<aiter_tensor_t> bias, int split_k);

// split-K：三 arch 统一
using OpusA16W16WorkspaceKernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    aiter_tensor_t& workspace,
    std::optional<aiter_tensor_t> bias, int split_k);
```

`gen_a16w16_tune_lookup()`（`gen_instances.py:542`）生成两张 kid 表，
**不能把两种函数指针放进同一张表**。

C++ 判断「这个 kid 要不要 workspace」= **查 generated workspace table 有没有命中**，
不用 `[200,300)` 一类数值区间。这直接把 `opus_gemm.cu:238-279` 那堆
kid 分类谓词的存在理由消掉一半。

split-K 表按 arch 分区，覆盖三个 arch（不是只有 gfx1250 —— #4246 就只改了 gfx1250，
这是它没做完的地方）。错误 arch 的 kid 不能命中当前设备的表。

raw ABI：

```cpp
void opus_gemm_a16w16_tune(
    aiter_tensor_t& XQ, aiter_tensor_t& WQ, aiter_tensor_t& Y,
    std::optional<aiter_tensor_t> bias,
    std::optional<aiter_tensor_t> workspace,   // 无默认值
    int kernel_id, int split_k);
```

non-workspace kid 必须传 `None`，workspace kid 必须传 Tensor —— 两边都校验，
防止「忘了传」和「传了但没人用」都静默通过。

**python 公共 wrapper 保持旧参数顺序**，`workspace=None` 作为末位 keyword-only
参数，别破坏现有 positional `(kid, splitK)` 调用。

## 第 7 步 — ABI 单点改动

`csrc/opus_gemm/gen_instances.py` `SPLITK_REDUCE_ABI_MAP`（**L72-98**，
三 arch 条目 L76-77 / L83-84 / L93-94，注释 L90）：

```
"ws_arg":  "const opus_splitk_ws_handle* ws_handle"  →  "void* ptr_ws"
"ws_type": "const opus_splitk_ws_handle*"            →  "void*"
```

改这一处，三个 arch 的生成签名同时变。消费点在 `:725` 和 `:846`。

## 第 8 步 — traits

顺序：gfx1250 → gfx950 → gfx942（gfx942 最后，它同时有 bf16/fp32 ws）

| 文件 | 行 |
|---|---|
| `include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh` | 头注释 7 / struct 45 / 注释 55,60 / kargs 64 |
| `include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh` | struct 593 / 注释 603 / kargs 611 |
| `include/gfx942/a16w16/opus_gemm_traits_a16w16.cuh` | guard 104-105 / struct 106 / helper 113-131 / kargs 137 |

删 `opus_splitk_ws_handle`，kargs 改 `void* ptr_ws`。

gfx942 的 `opus_splitk_ws_ptr<>()` helper 整个删 —— 它做的是
`readfirstlane` 把指针广播到 scalar 寄存器（`:117-126`）。**这个优化要保住**，
在 pipeline 里对 `kargs.ptr_ws` 原地做同样的 readfirstlane，别顺手删掉。

三份 struct 靠 `#ifndef` 互防，guard 一起删。
`include/opus_gemm_common.cuh:24` 的注释一并改。

## 第 9 步 — pipeline

每文件一行，数值算法零改动。改成 `reinterpret_cast<D_WS*>(kargs.ptr_ws)`。

| 文件 | 行 |
|---|---|
| `include/gfx1250/..._cluster_tdm_splitk_ws_gfx1250.cuh` | 386 |
| `include/gfx1250/..._clusterlaunch_tdm_splitk_ws_gfx1250.cuh` | 454 |
| `include/gfx950/..._flatmm_splitk_gfx950.cuh` | 266 |
| `include/gfx942/a16w16/..._em3en4_lds1_pgr2_sk.cuh` | 584 |
| `include/gfx942/a16w16/..._kbuf1.cuh` | 73 |
| `include/gfx942/a16w16/..._kbuf2v.cuh` | 77 |
| `include/gfx942/a16w16/..._kbuf2v_bk128.cuh` | 114 |
| `include/gfx942/a16w16/..._quad_mfma32_kbuf1.cuh` | 277 |

## 第 10 步 — reduce

| 文件 | 行 |
|---|---|
| `include/gfx1250/splitk_reduce_gfx1250.cuh` | 7, 9, 19, 26, 36, 176 |
| `include/gfx950/splitk_reduce_gfx950.cuh` | 65, 72, 86 |
| `include/gfx942/a16w16/splitk_reduce_gfx942.cuh` | 8, 15, 50, 197, 205, 212, 220, 228, 260 |

gfx942 最重：baseline / OOB 两套 + 转发层。

## 第 11 步 — codegen 删 grow 块

| 文件 | 行 | 内容 |
|---|---|---|
| `codegen/gen_instances_gfx1250.py` | 头注释 7 / decl 61,64 / fwd 228-235 / **289-326** | extern 三个、grow、sync 318、device mirror 325-326、无 batch 公式 299 |
| `codegen/gen_instances_gfx950.py` | 1240 注释 / **1324-1362** / 调用点 1391,1399,1408,1416 | extern 1327、grow 1348-1356、公式 1335-1336 |
| `codegen/gen_instances_gfx942.py` | decl 145,152,164,167,175 / 调用 284,310 / **515-548** | extern 515-517、公式 520-521（`sizeof(D_C)`）、sync 540、device handle 542-543 |

三个 `ws_bytes` 公式一起消失，统一到 python 的 `kid_workspace_plan()`。

**保留每个 arch 原有的 split_k clamp**（gfx950:1302-1308 / gfx942:480-494 /
gfx1250:277-282），只删 workspace 相关。

改成：接 workspace tensor → 校验 → 取 `data_ptr()`。

## 第 12 步 — 入口 / 绑定 / 框架 / 删 registry

| 文件 | 行 | 动作 |
|---|---|---|
| `csrc/opus_gemm/opus_gemm.cu` | **385-487** 全删 / 入口 294 | registry + 三个 `opus_splitk_ws_*` + `opus_gemm_workspace_init`；`opus_gemm_a16w16_tune` 加末位 `std::optional<aiter_tensor_t> workspace` |
| `include/opus_gemm.h` | 39 删 / tune 声明加参数 | |
| `csrc/include/rocm_ops.hpp` | **316-317 整段删** / tune def 加参数 | |
| `csrc/pybind/opus_gemm_pybind.cu` | 删 workspace_init 宏调用 | |
| `aiter/ops/opus/gemm_op_a16w16.py` | binding 72-87 / tune 165 / **550-551 删** / 558 导出 | 加 workspace 参数，调用前分配 |
| `aiter/ops/opus/__init__.py` | 43 / 61 / 68 | 改 deprecated no-op 空壳 |
| `aiter/tuned_gemm.py` | **44-49, 73-135 整段删** + 调用点 | prewarm 机制 |
| `csrc/opus_gemm/opus_gemm_tune.py` | tuner 调用点 | 走公共 wrapper，自动分配 |
| `csrc/gemm_a16w16/gemm_a16w16_tune.py` | `53-63` | 同上。注意它**自己在 `sys.path.insert`**（`:53-55`）后 `from opus_gemm_common import ...` |
| `aiter/ops/deepgemm.py` | `21` / `46-60` | deprecation shim，`:60` 转发 `_opus_tune(XQ, WQ, Y, kernelId, splitK)`。加 workspace 后这个调用点要跟着改 |
| `csrc/opus_gemm/README.md` | 261, 269, 285, 290, 300, 308, 314 | 改成 direct-pointer ABI |

`opus_gemm()`（`opus_gemm.cu:133`）的 bf16 分支（`:188-200`）**这一步先不删**，
但正常 python 路径不再走它 —— `_opus_gemm_bf16_dispatch`
（`gemm_op_a16w16.py:269`）改成先经 python heuristic 得到 kid，再走 tune 入口。
`opus_gemm()` 的删除属于任务二。

新建 `aiter/ops/opus/_workspace.py`：

```python
def make_workspace_plan(arch, family, kid, batch, M, N, K, split_k)
def allocate_workspace(plan, device)
```

```
shape = [storage_batch, alloc_split_k, padded_M, padded_N]
dtype = kid_workspace_dtype(...)
padded_M = ceil(M / block_m) * block_m
padded_N = ceil(N / block_n) * block_n
```

---

# C++ 保留的三条断言

```cpp
AITER_CHECK(!kid_needs_ws || workspace.has_value(), ...);
AITER_CHECK(ws.dtype() == kD_WS_dtype, ...);      // ← 最关键
AITER_CHECK(required_elems <= ws.numel(), ...);
```

dtype 那条是关键：#4246 挂机就是「写 fp32 的 kernel 拿到按 bf16 算大小的
buffer」—— numel 对、元素宽度错，只比 numel 查不出来。

外加 device 一致、contiguous、data_ptr alignment。所有 extent 乘法用
checked `size_t`，溢出在 launch 前失败。

`aiter_tensor_t` 有 `numel()` / `element_size()`，**没有 `nbytes()`**。

python 侧用 typed `torch.empty(shape, dtype=...)`。
**禁止 uint8 扁平 buffer** —— 传过去只剩 `void*`，dtype 断言就没了。
禁止全局缓存 workspace tensor。

HIP graph 不需要特殊处理：torch caching allocator 在 capture 期间返回
replay-stable 地址，这正是能删掉 prewarm 的原因。

---

# 四个坑

**坑 1 — ws dtype 读 kid 自己的属性，不按 kid 区间推。** gfx942 有三个 bf16ws kid
（`opus_gemm_common.py:729-731` → 10210 / 10213 / 10216）。字段**已存在**：
`splitk_workspace_dtype`（`:66` 默认 `"fp32_t"`，`:567`/`:596` 覆写成 `"bf16_t"`），
第 1 步直接读，不用新加。

硬编码 `sizeof(float)` 对这些 kid 只是浪费；反向（按 bf16 分配却写 fp32）是越界写 + 挂机。

**这条不是推测** —— #4246 的 `dc2f4890` 按 kid 区间硬编码 bf16，撞上写 fp32 的 kid，
`ea093a77` 的 commit message 原文：corrupting adjacent device memory and hanging
the machine。详见下方 #4246 一节。

---

**坑 1b（最危险的一条，切 torch 后必挂）—— gfx942 bf16ws launcher 会在 host 侧
偷偷改调另一个 kid 的 launcher。**

`codegen/gen_instances_gfx942.py:321-360` 生成的代码：

```cpp
const bool bf16ws_exact_reduce_shape =
    (N==64)||(N==128)||(N==256)||(N==384)||(N==512)||(N==1024)||(N==2048);
if (!bf16ws_exact_reduce_shape) {
    fp32ws_name<D_C>(XQ, WQ, Y, bias, splitK);   // ← 换成写 fp32 的 launcher
    return;
}
```

白名单来自 `EXACT_N_ROWBLOCK_REDUCE_CONFIGS`（`:127-137`）的 `vec*nvec`。
bf16 workspace 的 reduce 只实现了 exact-N rowblock 版本，N 对不上就整个换 kernel。

**今天安全**：C++ 自己分配 workspace，转跳后重新按 fp32 算尺寸。
**切到 torch 必挂**：python 按 bf16ws 分配了 buffer → launcher 转去跑写 fp32 的 kernel
→ **元素宽度差 2 倍**，正好是 `ea093a77` 挂机的同一形态。

两个 bf16ws kid 行为**不同**，必须分开处理：

| kid | kernel_tag | N 不在白名单时 | 转跳目标 |
|---|---|---|---|
| 10210 | `splitk_legacy_bf16ws` | **静默转跳** | `splitk_legacy_512x128x128x64` = **kid 10200** |
| 10213 | `splitk_p1_bk128_bf16ws` | **静默转跳** | `splitk_p1_bk128_256x64x64x128` = **kid 10203** |
| 10216 | `splitk_quad_mfma32_bf16ws` | `AITER_CHECK` 硬失败 | 无（`:332-339` 走 `is_quad_mfma32_splitk` 分支） |

**这条路径是可达的，不是理论风险。** C++ heuristic 自己就会返回 bf16ws kid：
`opus_gemm_heuristic_dispatch_gfx942.cuh:109-110` 的 `bf16ws_band()` 条件是
`N == 256 || (N >= 512 && N <= 2048)`（`:52-56`）—— **是个区间**，
而白名单是 7 个离散值。N=600 命中 band 但不在白名单 → 转跳触发。
`:72`/`:74` 返回 10213 的两处同理。

**处理（Step 1 必须做，不能推到 Step 5）**：
把这张 N 白名单和转跳规则前移到 python selector：

- **10210 / 10213**：N 不在白名单 → selector 直接改选转跳目标 kid（10200 / 10203），
  按目标 kid 的 fp32 ws 分配。**丢弃整对 `(kid, splitK)`**，因为 tuned row 的
  splitK 是配着 bf16ws kid 调出来的。
- **10216**：N 不在白名单 → 直接拒绝（explicit kid 报错，heuristic/tuned 则改选）。
- codegen 里的转跳代码**同批删除**。launcher 拿到 workspace 之后
  **永远不允许再换成另一种 ws dtype 的 kid**。

Step 6 要有专门的负例：构造 N=600 + bf16ws kid，断言 python 侧改选或报错，
且**没有**任何路径能让 bf16 buffer 喂给 fp32 kernel。

**坑 1c — exact-N 白名单已经有两份，且 tuner 那份漏了 N=384（现存 bug，已核实）。**

| 出处 | 内容 | 384？ |
|---|---|---|
| `codegen/gen_instances_gfx942.py:127-137` `EXACT_N_ROWBLOCK_REDUCE_CONFIGS` | `(16,24,4)  # N=384` 在列 | ✅ |
| `csrc/opus_gemm/opus_gemm_tune.py:98-106` `BF16WS_EXACT_REDUCE_SHAPES` | `(64,8)(128,4)(256,2)(512,1)(1024,4/2/1)(2048,1)` | ❌ **缺** |

tuner 侧 `kid_rejects_shape()`（`opus_gemm_tune.py:366`，判断在 `:415-419`）用它
剔除候选：

```python
if _kid_uses_bf16_workspace(k_inst):
    ...
    return not any(N == n_exact for n_exact, _ in BF16WS_EXACT_REDUCE_SHAPES)
```

后果：**N=384 时 tuner 把三个 bf16ws kid 全判成 rejected，从来没进过 tuning 空间**，
而 codegen 明明为 N=384 生成了 exact-N reduce kernel。所以 CSV 里 N=384 这一档
永远选不到 bf16ws —— 一个静默的调优覆盖漏洞，不是崩溃，因此至今没人发现。

注意两份的**形状也不同**：codegen 是 `(VEC, N_VEC, ROWS)` 三元组、N 由 `vec*nvec`
算出（1024 出现三次）；tuner 是 `(N, ROWS)` 二元组。所以不是简单复制粘贴的产物，
是两次独立手写 —— 这正是它会漂移的原因。

**做法**：`_GFX942_EXACT_N`（第 4b 步）定为唯一真源，放在 `opus_gemm_common.py`；
`opus_gemm_tune.py` 删掉 `BF16WS_EXACT_REDUCE_SHAPES`，`:419` 改成 `N not in _GFX942_EXACT_N`；
codegen 的 `EXACT_N_ROWBLOCK_REDUCE_CONFIGS` 保留（它还带 VEC/ROWS，是 kernel 生成参数），
但加一条 import 后的自检：`assert {v*nv for v,nv,_ in CONFIGS} == _GFX942_EXACT_N`。

**这一改会让 N=384 的 bf16ws kid 首次进入 tuning 空间。** 属于行为变更，不是纯重构，
commit message 要写明；如果不想在任务一里引入调优结果变化，可以先只加 assert 让它
炸出来、把修复单独拆一个 commit。但**不能就这么留着**——任务一之后 python selector
和 tuner 用同一个集合，不一致会直接体现为 selector 选了个 CSV 里根本没有的 kid。

**坑 2 — batch 项 per-arch 不同。**

| arch | 公式 | 出处 |
|---|---|---|
| gfx950 | `split_k * batch * padded_M * padded_N * sizeof(float)` | `gen_instances_gfx950.py:1335-1336` |
| gfx942 | `split_k * batch * padded_M * padded_N * sizeof(Traits::D_C)` ← 可能 bf16 | `gen_instances_gfx942.py:520-521` |
| gfx1250 | `split_k * padded_M * padded_N * sizeof(float)` ← **无 batch** | `gen_instances_gfx1250.py:299` |

挂到 `ws_includes_batch` 字段，不要在分配器里 `if arch == "gfx1250"`。
gfx1250 同时要先拒绝 `batch != 1`。

**坑 3 — launcher 会 clamp split_k**（gfx950:1302-1308 / gfx942:480-494 /
gfx1250:277-282）。按**未 clamp 的上界**分配，不要在 python 复刻 clamp 循环。
clamp 只会让 split_k 变小，buffer 只会富余。

**坑 3 特例（最隐蔽的越界点）—— gfx942 的 `splitK <= 0` 是「launcher 自己挑」，
不是「不切 K」。** `gen_instances_gfx942.py:470-477`：传 0 时按 CU 数自动挑，
clamp 到 `[1,16]`。而 `opus_gemm.cu:197/201` 传的正是字面 `0`
（`:188` 注释写着 "splitK=0 = launcher decides"）。按 splitK=1 分配、
kernel 按最大 16 跑 → 越界写。

**另两个 arch 是 `splitK <= 1 → 1`**（gfx950:1299 / gfx1250:275），
没有这个语义。**只测 gfx950/gfx1250 完全看不见这个 bug。**

处理：A0 之后 python 已知 kid，顺手也决定 splitK，不再传 0。

**坑 4 — `splitK` 没有上界硬夹。** 现在没人拦 `splitK=100000`，切到 torch 后
直接是一次巨额 `torch.empty`。python 侧用 `max_split_k` 校验，
**超限报错，不静默夹**。

---

# 验收

| 项 | 要求 |
|---|---|
| 机械 | 见下方 grep |
| 数值 | `op_tests/test_opus_a16w16_gemm.py` 全绿 |
| **三 arch** | 各跑一遍。gfx1250 batch 项不同、gfx942 ws dtype 不同，只测一个必漏 |
| **ws dtype** | 同时覆盖 fp32-ws 和 bf16-ws kid —— 坑 1 只在后者暴露 |
| **HIP graph** | capture/replay + golden 对拍。这是原机制存在的全部理由 |
| **双 stream** | 两 CPU 线程各驱动一 stream 同时跑 splitK（原 TBO 场景） |
| **契约负例** | 正好够大 / **少 1 元素** / 错 dtype / 错 device / noncontiguous / 未对齐。三条断言每条要有负例 |
| **gfx1250 batch>1** | 明确失败 |
| **heuristic 等价** | 三 arch × CSV-miss corpus，`(kid, splitK)` 逐条相同 |
| **bias dtype** | gfx950/942 的 `bias.dtype == Y.dtype`，**以及 gfx1250 的 `bf16 out + fp32 bias`**（`gen_instances_gfx1250.py:155` 明确绕开共享 `BIAS_HOST_VALIDATE`，`gen_instances.py:406`）。这是最容易在重构中丢掉的既有能力 |
| **splitK 上界** | 荒谬大的值在 python 侧被拒 |
| **gfx942 bf16ws 转跳** | 构造 N=600（命中 `bf16ws_band` 但不在 exact-N 白名单）+ kid 10210/10213，断言 python 改选到 10200/10203 并按 fp32 分配；kid 10216 同 shape 断言拒绝。**没有任何路径能让 bf16 buffer 喂给 fp32 kernel** |
| **exact-N 单源** | `N=384` 专项：断言 tuner 不再把 bf16ws kid 判 rejected（坑 1c 的既有 bug），且 codegen 自检 assert 通过。白名单内外各取几个点 |
| **两张表** | non-workspace kid 传 workspace 报错；workspace kid 传 `None` 报错 |
| **scope 隔离** | a8w8 / a8w4 MoE 的 API、launcher、codegen 产物不因本任务改变 |
| 显存 | 多 shape 循环后不因 python tensor 持有而持续增长 |

```bash
rg -n "ws_handle|opus_splitk_ws_|SplitkWsRegistry|workspace_init" csrc/ aiter/
rg -n "hipMalloc|hipFree|hipHostMalloc" csrc/opus_gemm/
rg -n "sizeof\(.*D_C|reinterpret_cast<.*D_C" csrc/opus_gemm/
rg -n "torch\.empty\(.*uint8" aiter/ops/opus/
rg -n "bf16ws_exact_reduce_shape|bf16ws_host_redirect" csrc/opus_gemm/    # 坑 1b，须归零
rg -n "sys\.path\.(insert|append)" aiter/ops/opus/                        # 别在 opus 模块注入 path
rg -n "BF16WS_EXACT_REDUCE_SHAPES" csrc/ aiter/                           # 坑 1c，须归零（只剩 _GFX942_EXACT_N）
git diff --check
```

前两条应只剩 deprecation 空壳和历史注释；每个命中都要能解释。

---

# 下游

`opus_gemm_workspace_init` 是公开 API 的破坏性删除。仓内只有 `tuned_gemm.py`
用（同批删）。已扫的两份 sglang checkout 未见直调，但合入前再扫一次未挂载的仓库。
python 侧留 deprecated no-op 空壳一个 release，C++ 直接删。

---

# 参考：#4246 实际做了什么（已 fetch 核实，head `25dd6281`，17 个 commit）

三个 workspace 相关 commit（**之前提到的 `4ac1fa8` 这个 hash 不存在，作废**）：

| commit | 内容 | 范围 |
|---|---|---|
| `dc2f4890` | workspace → `torch.empty`，删 registry 调用 | **仅 gfx1250**，14 文件 |
| `a9ed164e` | 删 `lru_cache` | 5 文件 |
| `ea093a77` | **按 kid 自己的 shape/dtype 定尺寸 + 删 prewarm** | 2 文件 |

其余 14 个 commit 是 TDM API 重构和 `_fuse` pipeline，与本任务无关。

## `ea093a77` 的 commit message 原文 —— 坑 1 的直接证据

> The gfx1250 split-K partial-sum workspace was allocated as a bf16 buffer sized
> from an over-estimated raw element count. Kids whose kernel writes an fp32
> workspace then overran the buffer (fp32 is 2x the bytes), **corrupting adjacent
> device memory and hanging the machine**.

而 `dc2f4890` 的 message 里正是那行肇事代码：

> Python layer: allocate workspace via `torch.empty(dtype=bf16)` **for kid [20000,30000)**

**按 kid 区间硬编码 ws dtype** —— 挂机的直接原因。`ea093a77` 的修法与本文第 1 步一致：
按 kid 自己的 kernel 定义算 `[batch, split_k, padded_M, padded_N]` 元素数 + 其 ws dtype，
永远不用裸字节数。

`ea093a77` 同时确认了 HIP graph 的判断：

> torch's HIP graph-capture-aware caching allocator makes a single `torch.empty`
> path valid for both eager and capture (replay-stable address, no pre-warm needed)

## #4246 没做完的事 —— 任务一要比它彻底

PR **最终 head** 的 `opus_gemm.cu:188-209`，heuristic 路径**仍然在 `hipMalloc`**：

```cpp
size_t ws_elems = (size_t)16 * padded_M * padded_N;   // 16 = splitK 上界魔数
size_t ws_bytes = ws_elems * sizeof(bf16_t);
HIP_CALL(hipMalloc(&ws_ptr, ws_bytes));
...
HIP_CALL(hipFree(ws_ptr));
```

它只切了 tuned 路径；CSV miss 走 C++ heuristic 时 python 不知道 kid，
只能就地 malloc 一个硬编码上界的 buffer（而且又是 bf16 硬编码）。

**这正是 A0 存在的理由**：不先让 python 知道 kid，path B 就消灭不掉 allocator。
任务一做完后这段代码不应存在。

## 若 #4246 先合入

它的 registry 比当前树**多两个函数** —— `opus_gemm_workspace_release()` 和
`opus_gemm_workspace_release_all()`（PR head `opus_gemm.cu:594` / `:615`）。
第 12 步的删除清单要相应加上，`rocm_ops.hpp` / pybind / python 导出同理。

## 若 #4320 先合入

已核实：它的 BMM split-K launcher 里**有 `opus_splitk_ws_get` + `hipMalloc` + grow**
（`gen_instances_gfx950.py` 两处，diff 中 6 个命中）。顶层 `opus_bmm.cu` 没有
workspace 参数，但生成的 launcher 有。

后果：任务一的 DoD「模块内 hipMalloc 归零」**不再自动成立**。二选一：

- 同批迁 BMM —— partial 布局与 a16w16 不同，需要单独的 ws plan
- 范围显式限定 a16w16 —— 审查命令加 `--glob '!*bmm*'`，并在 DoD 里写明

另外它引入了**跨 family 重复的 kid**，进一步坐实第 1 步的 `(arch, family, kid)` 三元组。

---

# 不在任务一范围

- 删 C++ heuristic / lookup 表 / `opus_gemm()` 入口 → 任务二
- a8w8 分支改造 → 任务二
- dtype / layout 整理 → 任务二
- gfx1250 fused split-K（#4246 的 (b) 部分）→ 本期不做
- BMM（#4320 未合入，树中无 `opus_bmm`）→ 若先合入需重新界定
