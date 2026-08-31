# 任务一 修改记录（splitK workspace → torch）

> 只记录**改了哪些文件、改了哪一处**。每完成一步在文末追加一节，不回填、不改写历史节。
>
> - 目标树：`/root/workspace/0810/aiter`
> - 基线 commit：`ca68b4f3501762c15c550cb920a5516e9710cf89`
> - 计划文档：`task1_steps.md`（步骤 / 坑 / 验收清单）

## 进度

| 步 | 内容 | 状态 |
|---|---|---|
| A0-1 | `opus_gemm_common.py` 加查询接口 | ✅ |
| A0-2 | gfx1250 heuristic 移植 | ✅ |
| A0-3 | gfx950 heuristic 移植 | ✅ |
| A0-4 | gfx942 heuristic + auto splitK 移植 | ✅ |
| A0-4b | gfx942 bf16ws 转跳规则搬进 selector | ✅ |
| A0-5 | 对拍测试 | ✅ |
| A0-gate | 提交前四项验收 | ✅ |
| A1-6 | 拆 `D_C` → `D_ACC`/`D_WS`/`D_OUT` | ✅ |
| A1-6b | 两张 launcher 表 | ✅ |
| A1-7 | launcher ABI 单点改动 | ✅ |
| A1-8 | traits | ✅ |
| A1-9 | pipeline | ✅ |
| A1-10 | reduce | ✅ |
| A1-11 | codegen | ✅ |
| A1-12 | 入口 / binding / 删 registry | ✅ |
| 13 | 坑 4 splitK 上界 + gfx1250 batch>1 拒绝 + 验收补测 | ✅ |
| 14 | 坑 1b 删 C++ redirect（收敛到计划）+ 验收清单剩余 7 行补测 | ✅ |
| 15 | review 两条：修 resolver 静默失效 + 删 `_workspace.py` | ✅ |

## 累计改动文件

| 文件 | 步 | 性质 |
|---|---|---|
| `csrc/opus_gemm/opus_gemm_common.py` | 1, 4b | 改（步 1 `+144/-0`，步 4b `+109/-0`，累计 `+253/-0`） |
| `csrc/opus_gemm/gen_instances.py` | 4 | 改（+76/-0） |
| `csrc/opus_gemm/opus_gemm_tune.py` | 4b | 改（+6/-10，**删掉错的那份副本**） |
| `csrc/opus_gemm/codegen/gen_instances_gfx942.py` | 4b | 改（+21/-6） |
| `aiter/ops/opus/common.py` | 4b | 改（+43/-0） |
| `aiter/ops/opus/gemm_op_a16w16.py` | 4b | 改（+29/-6） |
| `aiter/ops/opus/heuristic.py` | 2, 3, 4 | 新建（步 2 = 136 行）→ 步 3 `+66/-0` → 步 4 `+273/-0`，现 475 行 |
| `op_tests/test_opus_heuristic_parity.py` | 5, 5补 | 新建（步 5 = 681 行）→ 步 5补 `+50/-8`，现 723 行 |
| `op_tests/test_opus_splitk_workspace.py` | 13 | 新建（417 行，19 用例） |

---

# 第 1 步 — `opus_gemm_common.py` 加查询接口

**改动文件：1 个。纯 add-only，`+144 / -0`，三个 hunk。**

## `csrc/opus_gemm/opus_gemm_common.py`

| # | hunk | 位置 | 改了什么 |
|---|---|---|---|
| 1 | `@@ -85,0 +86,14 @@` | `:86-99`，`OpusGemmInstance` 内，`cluster_wg_n` 之后 | 加 2 个字段 |
| 2 | `@@ -783,0 +798,3 @@` | `:798-800`，`_a16w16_cluster_tdm_splitk_ws_gfx1250()` 构造末尾 | 置 `ws_includes_batch=False` |
| 3 | `@@ -1194,0 +1212,127 @@` | `:1212-1338`，`heuristic_kids_for_arch()` 之后 | 加查询 API |

### 改动 1 — `:86-99` 加字段

```python
ws_includes_batch: bool = True   # True -> [split_k, batch, pM, pN]；False -> [split_k, pM, pN]
max_split_k: int = 16            # launcher 只向下 clamp，故为有效上界
```

带默认值、加在 dataclass 末尾，现存位置参数构造点不受影响。

`max_split_k=16` 依据：gfx942 硬 clamp（`gen_instances_gfx942.py:477`）；gfx950
（`gen_instances_gfx950.py:1299-1308`）/ gfx1250（`gen_instances_gfx1250.py:275-282`）
只按 iters-per-split 向下 clamp；tuner 统一 `min(16,...)`（`opus_gemm_tune.py:309`）。

### 改动 2 — `:798-800` gfx1250 置 `ws_includes_batch=False`

依据 `gen_instances_gfx1250.py:299`（`ws_bytes` 无 batch 项）和 `:367`（reduce 传字面量 1）。

只改这一个工厂即可：`_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250()`（`:939`）用
`dataclasses.replace()` 从它派生，自动继承。覆盖 28 + 468 = 496 个 gfx1250 kid。

### 改动 3 — `:1212-1338` 加查询 API

| 符号 | 行 |
|---|---|
| `_DEFAULT_ARCH = "gfx950"` | `:1228` |
| `kid_arch(kid_or_inst)` | `:1231` |
| `kid_family(kid_or_inst)` | `:1241` |
| `kid_info(arch, family, kid)` | `:1262` |
| `kid_workspace_dtype(arch, family, kid)` | `:1280` |
| `kid_workspace_plan(arch, family, kid, batch, M, N, K, split_k)` | `:1300` |

读现成的 `kernels_list` / `SPLITK_KIDS`，不复制实例表，不建 `_catalog.py`。

三个要点：
- `kid_family` 从 `kernel_tag` 推，**不从 kid 段推**。kid 段按 arch 划分，gfx942 的
  `[10000,20000)` 里混着 a8w8 kid 11000（`:734` 定义，`:741` 并入 `gfx942_kernels_list`）。
- `kid_info` 断言完整三元组，不符抛 `KeyError`。
- `kid_workspace_plan` 返回 `(shape, dtype)` 而非字节数，元素数精确。
- `kid_workspace_dtype` 的 docstring 写死：**workspace dtype ≠ launcher 的 `D_C`**。
  splitK launcher 一律 `D_C=fp32_t`（`gen_instances_gfx942.py:416` 有 static_assert），
  bf16 只在 traits 第三个 DTYPE 槽（`:222-231`）。按 `D_C` 算尺寸 = 2 倍欠分配 = `ea093a77` 挂机。

## 验证

| 项 | 结果 |
|---|---|
| codegen 产物 diff | gfx942(39 文件)/gfx950(30)/gfx1250(20) 三个 arch 全部**为空** |
| 全 kid 解析 | 662 个 kid 全部解出 `(arch,family)`；ws dtype: None=109 / fp32_t=549 / bf16_t=3(`{10210,10213,10216}`) |
| 三元组守卫 | kid 11000 查 gfx950 / 查 a16w16、kid 10200 查 a8w8，均正确抛 KeyError |
| plan 对 C++ 公式 | 552 splitK kid × 5 形状 = **2760 对，mismatch 0** |
| 语法 | `py_compile` 通过 |

codegen diff 必须三个 arch 各跑：`GPU_ARCHS` 默认 `native` 本机只探到 gfx950
（`gen_instances.py:1108-1125`），而改动 2 只影响 gfx1250。

```bash
for A in gfx942 gfx950 gfx1250; do GPU_ARCHS=$A python gen_instances.py -w /tmp/gia_$A --tune; diff -r /tmp/gib_$A /tmp/gia_$A; done
```

## 未做

无调用方接入；坑 1c（`opus_gemm_tune.py:98-106` 漏 N=384）本步未动，留第 4b 步。

---

# 第 2 步 — gfx1250 heuristic 移植

**改动文件：1 个（新建）。C++ 头文件一行未动，第 1 步的文件也未再动。**

## `aiter/ops/opus/heuristic.py`（新建，136 行）

源：`csrc/opus_gemm/include/gfx1250/opus_gemm_heuristic_dispatch_gfx1250.cuh:27-48`，逐字翻译。

| 符号 | 行 | 内容 |
|---|---|---|
| 模块 docstring | `:1-55` | 为什么建、ADD-ONLY 状态、"只搬不改"规则、不在范围内的四件事 |
| `__all__` | `:57` | 3 个导出 |
| `heuristic_kid_gfx1250(M,N,K,has_bias)` | `:72` | 主体，对应 C++ `:27` |
| `_HEURISTIC_BY_ARCH` | `:117` | 目前只有 gfx1250 一项 |
| `SUPPORTED_ARCHES` | `:121` | |
| `heuristic_kid(arch,M,N,K,has_bias)` | `:124` | arch 分发，未移植的抛 `NotImplementedError` |

三个 arch 放一个文件，不拆包。模块不 import torch、不 import aiter 内部任何东西。

### 翻译要点

C++ 的 `if (M % 32 == 0) { ... }` 块**不带 else**，三个 N 检查都不中时 fall through
到 tileN 家族。这个结构必须保留——`32x256` tile 根本不存在（per-TDM B 请求数撞上
SIMD-pair 的 256 direct-copy 上限，`_ctdm_pick_configs()` 已丢弃），所以
`N % 256 == 0` 必须掉到 B_M=16 家族。改写成 `if/else` 即错。

`K` / `has_bias` 收下即丢（`del K, has_bias`），对应 C++ `(void)K; (void)has_bias;`。

## 验证

把 C++ 头直接 `g++ -O2 -I csrc/opus_gemm/include/gfx1250` 编成 oracle（纯整数函数、
无 HIP 依赖，可独立编译），做差分测试：

| 项 | 结果 |
|---|---|
| 穷举 `(M,N) ∈ [1,600]² × has_bias` | 720,000 例，**mismatch 0** |
| 边界/大值网格 24M × 25N × 8K × 2bias | 9,600 例，**mismatch 0** |
| 返回 kid 集合 | `{20000,20003,20004,20005,20006,20007}` == `HEURISTIC_DEFAULT_KIDS_GFX1250` |
| 分发层 | 大小写不敏感；gfx950/gfx942/gfx900 均抛 `NotImplementedError` |
| 与第 1 步串联 | `heuristic → kid → kid_workspace_plan` 通；`M=17,N=600` → kid 20000 → `((16,32,608),'fp32_t')`（N pad 到 608、三维无 batch） |
| 语法 | `py_compile` 通过 |

边界值含质数、刚过 tile 边界、刚差 1（`17,31,33,127,255,1000003` 等）；K 扫 8 个值
验证其确实不参与决策。

kid 集合相等这条不是形式检查：`HEURISTIC_DEFAULT_KIDS_GFX1250` 的存在理由就是
"heuristic 能返回的 kid 必须强制编进 .so"，不等则运行时找不到符号。

**环境注意**：`import aiter` 本机因 triton 3.4.0 < 3.6.0 失败（既有问题，第 1 步已
用 `git stash` 确认与本任务无关）。本模块零依赖，测试用
`importlib.util.spec_from_file_location` 独立加载。

## 未做

gfx950（`opus_gemm_heuristic_dispatch_gfx950.cuh:83`，127 行）留第 3 步；
gfx942（`opus_gemm_heuristic_dispatch_gfx942.cuh:142`，157 行）留第 4 步，注意它
**返回函数指针不是 kid**，需先反查回 kid。无调用方接入，`__init__.py` 未改。

## 复查

```bash
git status --porcelain                          # M csrc/.../opus_gemm_common.py + ?? aiter/ops/opus/heuristic.py
git grep -n "heuristic_kid_gfx1250" -- aiter/   # 只有定义处，无调用方
git diff --stat -- csrc/                        # 第 2 步不应给 csrc/ 增加改动
```

---

# 第 3 步 — gfx950 heuristic 移植

**改动文件：1 个（改第 2 步建的那个）。纯 add-only，`+66 / -0`，四个 hunk。C++ 头一行未动，`opus_gemm_common.py` 也未再动（`git diff --stat -- csrc/` 仍是 `144 insertions`，与第 1 步一致）。**

## `aiter/ops/opus/heuristic.py`（136 → 202 行）

源：`csrc/opus_gemm/include/gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh:83-127`，逐字翻译。

| # | 位置 | 改了什么 |
|---|---|---|
| 1 | `:38` docstring SOURCE HEADERS | gfx950 那行去掉 "(not yet ported)"；gfx942 保留 |
| 2 | `:57-62` `__all__` | 加 `"heuristic_kid_gfx950"`（放首位） |
| 3 | `:65-127` 新增整节 | gfx950 段注释 + `heuristic_kid_gfx950()` 主体，插在 gfx1250 段**之前** |
| 4 | `:182-185` `_HEURISTIC_BY_ARCH` | 加 `"gfx950": heuristic_kid_gfx950` |

改动 3 插在前面，gfx1250 段整体下移：`heuristic_kid_gfx1250` 由 `:72` → `:137`，
`_HEURISTIC_BY_ARCH` 由 `:117` → `:182`，`SUPPORTED_ARCHES` `:121` → `:187`，
`heuristic_kid` `:124` → `:190`。gfx1250 函数体一个字符未改。

| 符号 | 行 |
|---|---|
| `heuristic_kid_gfx950(M,N,K,has_bias=False)` | `:73` |
| ├ `split_barrier_ok` | `:94` |
| ├ `M <= 4` → 208 / 1208 | `:96-101`（C++ `:88-95`） |
| ├ `M <= 64` → 206 / 1206 | `:102-107`（C++ `:96-103`） |
| ├ `M <= 128` → 200 / 1200 | `:108-113`（C++ `:104-111`） |
| ├ `M > 128` persistent → 300 / 1300 | `:114-121`（C++ `:113-121`） |
| └ fall-through → 200 / 1200 | `:122-126`（C++ `:124-126`） |

### 翻译要点

- **默认参数照搬**：C++ 是 `bool has_bias = false`，python 写 `has_bias: bool = False`。
  省略实参时两边落同一分支。（gfx1250 无默认值，故那边不加。）
- **`has_bias` 真参与决策**，不像 gfx1250 是 `(void)`。persistent 家族没有
  `HAS_BIAS=true` 实例，带 bias 必须掉回 splitk（200/1200）。所以这里**不能**有 `del`。
- **两处 `return 200 / 1200` 是两个独立分支**（C++ `:104-111` 和 `:124-126`），
  条件不同（前者进过 `M<=128`，后者是 `M>128` 的兜底）。合并即错。
- `split_barrier_ok` 的 `(K // 64) % 2 == 0`：C++ 是 `(K / 64) % 2`，整数除。
  K 已被同表达式的 `K % 64 == 0` 约束，python `//` 与 C++ `/` 在正数上一致。

## 验证

C++ oracle：这个头不像 gfx1250 那样零依赖——`:39` `aiter_tensor.h` 拉进 HIP，
`:41` `opus_gemm_manifest.h` 是 **codegen 产物、不在源码树里**。故用 `hipcc`
（`/opt/rocm/bin/hipcc`）编，并把第 1 步跑 codegen 时生成的 `/tmp/gib_gfx950` 加进
include path：

```bash
hipcc -O2 -I csrc/opus_gemm/include -I csrc/include -I /tmp/gib_gfx950 h950.cpp -o h950
```

| 项 | 结果 |
|---|---|
| 穷举 `M,N ∈ [1,300]² × 8 个 K × 2 bias` | 1,440,000 例，**mismatch 0** |
| 边界/大值网格 21M × 21N × 17K × 2bias | 14,994 例，**mismatch 0** |
| 合计 | **1,454,994 例，mismatch 0** |
| 默认实参 | `f(M,N,K)` == `f(M,N,K,False)` 全网格成立 |
| 返回 kid 集合 | `{200,206,208,300,1200,1206,1300}` ⊂ `HEURISTIC_DEFAULT_KIDS_GFX950` |
| 分发层 | `gfx950`/`GFX950` 都通；gfx942、gfx900 抛 `NotImplementedError`；`SUPPORTED_ARCHES == {gfx950, gfx1250}` |
| 与第 1 步串联 | `M=3,N=64,K=128` → 208 → `((4,1,64,64),'fp32_t')`；`M=512,N=512` bias=False → 1300（persistent，非 splitK，plan 返回 `None`），bias=True → 1200 → `((4,1,512,512),'fp32_t')` |

边界值含 `1,2,3,4,5`（M<=4 桶）、`63/64/65`、`127/128/129`、`255/256/257`、
`383/384`、`600`、质数 `1000003`；K 含 `1,63,65,127,191,320,448`（专打
`split_barrier_ok` 三个子条件的各种失败组合）。

### 观察：kid 1208 不可达（**不修**）

`M <= 4` 分支里 nooob 的门是 `M % 64 == 0`，而 M ∈ [1,4] 永远不成立 →
`return 1208`（C++ `:93`）是死代码。1,454,994 例中 1208 出现 0 次，故上表
"返回 kid 集合" 是**真子集**而非相等（gfx1250 那步是相等）。

C++ 里就是这样（`opus_gemm_heuristic_dispatch_gfx950.cuh:88-95`），python 照抄。
按"只搬不改"规则本步不动它——改门（比如放宽成 `N%64==0 && K%128==0`）会让 1208
开始被选中，这是**行为变更不是重构**，且 1208 已在 `HEURISTIC_DEFAULT_KIDS_GFX950`
（`opus_gemm_common.py:1135`）里、白占一个编译实例。留作独立议题，第 5 步对拍时
连同 gfx942 的类似情况一起提。

## 未做

gfx942（`opus_gemm_heuristic_dispatch_gfx942.cuh:142`，157 行）留第 4 步，
注意它**返回函数指针不是 kid**，需先反查回 kid。无调用方接入
（`git grep heuristic_kid_gfx950 -- aiter/` 只有定义处），`__init__.py` 未改。

---

# 第 4 步 — gfx942 heuristic + auto splitK 移植

**改动文件：2 个。均纯 add-only：`gen_instances.py` `+76/-0`，`heuristic.py` `+273/-0`（202 → 475 行）。C++ 头一行未动，`opus_gemm_common.py` 未再动。**

新增 codegen 产物 1 个：`opus_gemm_kid_symbols.json`（测试/工具用，无 TU include）。

## 一、`csrc/opus_gemm/gen_instances.py`（+76/-0，2 个 hunk）

| # | 位置 | 改了什么 |
|---|---|---|
| 1 | `:30` `from opus_gemm_common import (...)` | 加 `kid_arch`（第 1 步加的函数） |
| 2 | `:692-763` | 新增 `gen_kid_symbol_map()`，插在 `gen_a8w8_tune_lookup()` 之后、`_emit_fused_host_tu()` 之前 |
| 3 | `:975` `gen()` 收尾 | 加一行 `self.gen_kid_symbol_map(kernels_dict)`，跟在 `gen_a8w8_tune_lookup` 后 |

产物 `opus_gemm_kid_symbols.json` 结构：

```
{"arch_kids": {gfx950|gfx942|gfx1250: [kid...]},
 "symbol_to_kid": {launcher符号: kid},
 "kids": {kid: {name, kernel_tag, B_M, B_N, B_K, output_dtypes,
                splitk_workspace_dtype, ws_includes_batch, max_split_k, is_splitk}}}
```

为什么加这个：gfx942 的 heuristic **返回 launcher 函数指针不是 kid**，移植要做
symbol → kid 反查。这张表**必须从 `kernels_dict` 派生**——手写第二张表在 kid 改名/
改号时会静默失配，失败模式是选错 kernel 而不是编译报错。放在 codegen 里 = 和声明这些
符号的 manifest 由同一趟生成，两者不可能对不上。

函数里带 `assert k.name not in symbol_to_kid`：符号全局唯一才敢用一张扁平表
（实测 gfx942 23 个 kid / 23 个唯一 name，无重名）。

## 二、`aiter/ops/opus/heuristic.py`（+273/-0，4 个 hunk）

源：`include/gfx942/opus_gemm_heuristic_dispatch_gfx942.cuh:142`（主函数）+ `:45-137`
（detail 命名空间）；auto splitK 源 `codegen/gen_instances_gfx942.py:470-477`。

| # | 位置 | 改了什么 |
|---|---|---|
| 1 | `:38` | docstring SOURCE HEADERS 去掉 gfx942 的 "(not yet ported)"，三个 arch 全绿 |
| 2 | `:57-64` `__all__` | 加 `heuristic_kid_gfx942`、`auto_split_k_gfx942`、`GFX942_DEFAULT_CU_NUM` |
| 3 | `:67-338` | 新增 gfx942 整节，插在 gfx950 段**之前** |
| 4 | `:415-419`、`:426-451` | `_HEURISTIC_BY_ARCH` 加 gfx942 项；`heuristic()` 分发加 `batch`/`out_dtype` 形参 |

改动 3 插在最前，后面两段整体下移：`heuristic_kid_gfx950` `:73` → `:344`，
`heuristic_kid_gfx1250` `:137` → `:408`。两个函数体一字未改（步 3 的 1.44M 例回归重跑，mismatch 0）。

### gfx942 节符号表

| 符号 | 行 | 对应 C++ |
|---|---|---|
| `_G942_*` 14 个 kid 常量 | `:88-101` | `.cuh:26-39` 声明的 14 个 launcher 符号 |
| `_g942_split_barrier_ok(N,K)` | `:104` | `detail::split_barrier_ok` `:45-50` |
| `_g942_bf16ws_band(M,N,K)` | `:116` | `detail::bf16ws_band` `:52-57` |
| `_g942_dispatch_bf16(M,N,K)` | `:132` | `detail::dispatch_bf16<CDataType>` `:59-137` |
| `heuristic_kid_gfx942(M,N,K,batch=1,has_bias=False,out_dtype="bf16_t")` | `:222` | 主函数 `:142-157` |
| `GFX942_DEFAULT_CU_NUM = 64` | `:264` | `gen_instances_gfx942.py:468` fallback |
| `auto_split_k_gfx942(B_M,B_N,M,N,batch,kernel_tag,cu_num=None)` | `:267` | `gen_instances_gfx942.py:470-477` |

### 翻译要点

- **返回 kid 不返回指针**。14 个 `_G942_*` 常量每个都带注释写明它代表哪个符号，
  测试拿生成的 JSON 逐个核对（见验证 [2]），不靠"我抄对了"。
- **`CDataType` → `out_dtype: str` 形参**。C++ 用
  `if constexpr (std::is_same_v<CDataType, bf16_t>)` 把整张 bf16 回退表挡在编译期后面，
  fp32 输出只走最后两行。做成运行时形参才能两条路都可测；默认 `"bf16_t"`。
  `dispatch_bf16` 的模板参数在 kid 粒度不可见（非 splitk 返回 `<CDataType>`、
  splitk 返回 `<fp32_t>`，同一个 kid），故 port 里丢掉。
- **`batch` 收下即丢**（C++ 里就写作 `int /*batch*/`）。保留在签名里是因为 workspace
  尺寸**确实**依赖 batch（见 `auto_split_k_gfx942`），删掉会诱导调用方误判。
- **`K == 4096` 块是等值不是区间**，且块内不中时**继续往下走**（C++ 无终结 return），
  这个 fall-through 必须保留。
- **gfx942 的 `split_barrier_ok` 与 gfx950 的不是同一个谓词**：gfx942 是
  `loops = ceil(K/64)` 且要求 `loops >= 2`；gfx950 是 `(K//64) % 2 == 0` 无下界。
  两者不能合并。

### auto splitK：只搬初值，不搬 clamp 循环

按要求只移植 `:470-477` 的初值，`:480-494` 的 clamp 循环留在 C++。

理由写进 docstring 了：**clamp 循环只会把 split_k 调小**，所以 python 这个值是上界，
按它分配至多浪费、绝不短缺。把循环也搬过来则两边必须逐位一致，任何漂移都变成
**欠分配 = 踩内存**。单向留余量是更便宜的契约。

docstring 明确写死：这个函数的返回值是"**该按多大分配 workspace**"，
不是"kernel 实际会用的 split_k"。

`cu_num` 默认 `GFX942_DEFAULT_CU_NUM=64` 而不去探设备——本模块坚持不 import torch。
同时警告：cu_num 传小了 → split_k 偏小 → **会**欠分配。

## 验证

C++ oracle 用 `hipcc` 编，include 路径带上本步新生成的 `/tmp/gic_gfx942`。
指针 → kid 的还原表**由 JSON 产物生成**（不是手打），14 个符号 × `<bf16_t>`/`<fp32_t>`
两种实例化共 28 行；符号只有声明没有定义，测试 TU 里补 14 个空定义 + 显式实例化。

| # | 项 | 结果 |
|---|---|---|
| 1 | 对 C++ oracle 差分（30M × 22N × 22K × 2bias × 2dtype） | **58,080 例，mismatch 0** |
| 1b | 指针全部还原成功 | unmapped(-1) = **0** |
| 2 | 14 个 `_G942_*` 常量 vs 生成的 JSON | **mismatch 0**，别名集合完全覆盖 |
| 3 | JSON 的 kid→name vs `kernels_list` | 全等；gfx942 15 kid / gfx950 10 / gfx1250 6 |
| 4 | `out_dtype="fp32_t"` 返回集 | `{10200, 10201}`，确实只走两行尾巴 |
| 5 | `bf16 + has_bias=True` 返回集 | `{10200, 10201}`，同上，bf16 表整个跳过 |
| 6 | `auto_split_k_gfx942` vs 独立重写的公式 | 20 万随机例（含 M=0/N=0/batch=0/cu∈{64,104,304}），**mismatch 0**；值域恒在 [1,16] |
| 7 | 返回 kid 集合 | 14 个，⊂ `HEURISTIC_DEFAULT_KIDS_GFX942`（15 个） |
| 8 | codegen 三 arch 产物 diff | 除新增的 `opus_gemm_kid_symbols.json` 外**全部为空** |
| 9 | 回归：gfx950 | 1,440,000 例重跑，**mismatch 0** |
| 10 | 回归：gfx1250 | 抽样一致 |
| 11 | 分发层 | `gfx942`/`GFX942` 通；`SUPPORTED_ARCHES == {gfx942, gfx950, gfx1250}`（三个 arch 齐了，已无 `NotImplementedError` 分支可测） |
| 12 | 与第 1 步串联 | `M=256,N=256,K=4096` → kid 10213（**ws=bf16_t**）→ sk=4 → `((4,1,256,256),'bf16_t')`；`M=300,N=512,K=4096` → 10210（bf16_t）→ sk=6 → `((6,1,384,512),'bf16_t')`（M pad 到 384） |
| 13 | 语法 | 两个文件 `py_compile` 通过 |

测试网格专挑 C++ 里出现过的边界常量：M 含 `4/8/16/32/48/64/103/104/105/128/224/225/391/392/512/608/609`，
N 含 `64/256/383/384/385/512/800/801/1024/1536/2048/4096`，
K 含 `512/640/641/1024/3200/3264/4096/4097/7168`——即每个 `<=`/`>=`/`==` 的两侧各取一点。

第 12 条是本步最要紧的一条：gfx942 是唯一有 **bf16 workspace** 的 arch，
`kid_workspace_plan` 对 10210/10213 返回 `'bf16_t'` 而非 `'fp32_t'`，
正是第 1 步 docstring 里写死的那个 `ea093a77` 2 倍欠分配陷阱的正例。

### 观察：`target_wg = 2 * cu_num` 分支在 C++ 里是死的（**不修**）

`gen_instances_gfx942.py:474` 的条件是 `k.kernel_tag.endswith("_p1")`，但 gfx942
**没有任何 kernel_tag 以 `_p1` 结尾**——实际的 tag 是 `a16w16_kbuf2v_sk`、
`a16w16_kbuf2v_bk128_sk` 等，`_p1` 只出现在 launcher **符号名**里
（`opus_gemm_gfx942_splitk_p1_...`）。核实生成物：7 个 gfx942 splitk launcher
全部生成 `target_wg_dbuf2 = cu_cached`，`2 * cu_cached` **出现 0 次**。

按"只搬不改"照抄了这个条件，并在 `auto_split_k_gfx942` 里把上述核实结果写成注释。
改判据（比如换成看符号名或 `GFX942_EVEN_LOOP_SPLITK_TAGS`）会让 p1 kid 的 split_k
初值翻倍 → **行为变更不是重构**，且会改变 workspace 尺寸，必须单独立项。
与第 3 步的 kid 1208 死分支一并留到第 5 步对拍时提。

### 观察：kid 10003 不可达

`HEURISTIC_DEFAULT_KIDS_GFX942` 有 15 个 kid，heuristic 只能返回 14 个，
`10003`（`opus_gemm_gfx942_p1_bk128_256x64x64x128_2x2_...`）从不被返回。
它在 default 集里是为了别的原因（CSV 命中路径），不是 heuristic 的漏。与 gfx950 的
1208 不同——1208 是 heuristic 里写了但走不到，10003 是 heuristic 里压根没写。仅记录。

## 未做

三个 arch 的 heuristic 全部移植完毕。仍无调用方接入
（`git grep heuristic_kid_gfx942 -- aiter/` 只有定义处），`__init__.py` 未改，
C++ 头文件三个都没动。gfx942 bf16ws 转跳规则 + 坑 1c（N=384）留第 4b 步；
对拍测试落盘成正式 test 文件留第 5 步。

## 复查

```bash
git status --porcelain      # M gen_instances.py + M opus_gemm_common.py + ?? heuristic.py
git diff --stat             # 220 insertions, 0 deletions
git grep -n "heuristic_kid_gfx942" -- aiter/   # 只有定义处
for A in gfx942 gfx950 gfx1250; do GPU_ARCHS=$A python gen_instances.py -w /tmp/gic_$A --tune; diff -rq /tmp/gib_$A /tmp/gic_$A; done
# 唯一差异应为 "Only in /tmp/gic_$A: opus_gemm_kid_symbols.json"
```

---

# 第 4b 步 — gfx942 bf16ws 转跳规则搬进 selector + 坑 1c 修复

**改动文件：5 个，`+213/-22`。这一步含一处行为变更（N=384），见文末。**

## 一、`csrc/opus_gemm/opus_gemm_common.py`（+109/-0，1 个 hunk）

`@@ -1194,0 +1212,236 @@` —— 第 1 步那个 hunk 从 127 行长到 236 行，
新增部分接在 `kid_workspace_plan()` 之后、`_opus_sidecar_path()` 之前。

| 符号 | 行 | 内容 |
|---|---|---|
| `_GFX942_EXACT_N` | `:1361` | `frozenset({64,128,256,384,512,1024,2048})`，**唯一定义处** |
| `_GFX942_BF16WS_REDIRECT` | `:1376` | `{10210: 10200, 10213: 10203}` |
| `_GFX942_BF16WS_STRICT` | `:1380` | `frozenset({10216})` |
| 三条防漂移 assert | `:1384-1400` | 见下 |
| `gfx942_bf16ws_exact_n_ok(N)` | `:1402` | 谓词 |
| `resolve_gfx942_bf16ws(kid,N,splitK=None,*,explicit=False)` | `:1407` | 返回 `(kid, splitK, rejected)` |

模块级 assert 三条：
1. `REDIRECT ∪ STRICT` 必须**恰好等于** `gfx942_kernels_list` 里所有 `ws==bf16_t` 的 kid；
2. 每个 redirect 的 `(B_M,B_N,B_K)` 源与目标必须相同（几何变了 = workspace 尺寸错）；
3. redirect 目标必须是 `fp32_t` workspace。

`resolve_*` 对非 bf16ws kid、对其它 arch 的 kid 一律原样返回，可无条件调用。

## 二、`csrc/opus_gemm/opus_gemm_tune.py`（+6/-10，3 个 hunk）—— 删掉错的副本

| # | hunk | 改了什么 |
|---|---|---|
| 1 | `@@ -66,0 +67 @@` | import 加 `gfx942_bf16ws_exact_n_ok` |
| 2 | `@@ -98,10 +99,6 @@` | **删除 `BF16WS_EXACT_REDUCE_SHAPES` 元组（10 行）**，换成 6 行注释说明它为什么错、去哪儿找 |
| 3 | `@@ -419 +416 @@` | `kid_rejects_shape()` 里 `return not any(N == n_exact for n_exact, _ in BF16WS_EXACT_REDUCE_SHAPES)` → `return not gfx942_bf16ws_exact_n_ok(N)` |

## 三、`csrc/opus_gemm/codegen/gen_instances_gfx942.py`（+21/-6，4 个 hunk）

| # | hunk | 改了什么 |
|---|---|---|
| 1 | `@@ -8 +8 @@` | import 加 `_GFX942_EXACT_N` |
| 2 | `@@ -138,0 +139,16 @@` | `EXACT_N_ROWBLOCK_REDUCE_CONFIGS` 之后加模块级 assert（`:147`）+ 说明 |
| 3 | `@@ -324,0 +341,4 @@` / `@@ -326,4 +346 @@` | `exact_reduce_shape_conditions` 改成**直接从 `_GFX942_EXACT_N` 生成**，不再从 CONFIGS 反推 |

`EXACT_N_ROWBLOCK_REDUCE_CONFIGS` **留在 codegen**，因为它携带 per-N 的 kernel 生成参数
（VEC / N_VEC / ROWS），`opus_gemm_common` 只需要知道"哪些 N 合法"。但它覆盖的 N 集合
被 assert 钉死在单一源上。改动 3 是为了让 C++ guard 和 python selector 是**同一个谓词**，
不留任何瞬时漂移窗口。

至此 `_GFX942_EXACT_N` 是**唯一**字面量，第三份（codegen 内联生成 C++ 条件的那份）也消除了。

## 四、`aiter/ops/opus/common.py`（+43/-0，2 个 hunk）

| # | 位置 | 改了什么 |
|---|---|---|
| 1 | `:218-260` | 新增 `_bf16ws_resolver()`（`:219`，`@lru_cache(1)`）+ `resolve_bf16ws_kid()`（`:245`），插在 `mono_kid_shape_ok` 之后、`lookup_tuned` 之前 |
| 2 | `:282` | `__all__` 加 `"resolve_bf16ws_kid"` |

**没有在这里复制第四份表**。`opus_gemm_common` 走**惰性 import**：模块加载期是循环依赖
（aiter 导入本模块，opus_gemm_common 导入 aiter.jit.core），且 `csrc/opus_gemm` 要等
`aiter.jit.core` 算出 `AITER_META_DIR` 才进 `sys.path`（`aiter/jit/core.py:414`）。
与 `opus_gemm_common._opus_sidecar_path()` 里 `from aiter.jit.core import bd_dir` 同一手法。

import 失败（无 csrc sidecar 的打包安装）时 `resolve_bf16ws_kid` **原样放行**，
退回 C++ launcher 自己的转跳——A0 阶段那条路还在，所以是退回现状而不是误路由。

## 五、`aiter/ops/opus/gemm_op_a16w16.py`（+29/-6，2 个 hunk）

| # | 位置 | 路径 | 行为 |
|---|---|---|---|
| 1 | `:503-518`（调用在 `:508`） | 显式 kid | `explicit=True`；`rejected` → **抛 `ValueError`**（调用方点名了这个 kid，静默换成别的 = 跑了不是他要的 kernel） |
| 2 | `:540-554`（调用在 `:549`） | tuned CSV 命中 | `explicit=False`；`rejected` → **不 return，落到下面的 heuristic 分支**（tuned 只是建议） |

两条路都吃 redirect（10210/10213），差别只在 10216 被拒时的反应。

tuned 那条为什么会撞上：CSV key 经 `tuned_gemm.get_padded_m` 只按 M 补齐，
所以在合法 N 上调出来的行可能在别的 N 上被取到。

## 验证

| # | 项 | 结果 |
|---|---|---|
| 1 | 三份副本清零 | `grep BF16WS_EXACT_REDUCE_SHAPES` 只剩注释；`_GFX942_EXACT_N` 单一字面量 |
| 2 | codegen 三 arch 产物 diff | 对第 4 步产物**逐字节相同**（gfx942/gfx950/gfx1250 全空） |
| 3 | 生成的 C++ guard | 两个 bf16ws launcher 都是 `N==64\|\|128\|\|256\|\|384\|\|512\|\|1024\|\|2048`，与 python 谓词同集合 |
| 4 | `resolve_*` 语义 | 24 组（3 kid × 3 N × 2 explicit）逐条核对：合法 N 原样；10210/10213 非法 N → 转跳且 splitK 置 None；10216 非法 N → `rejected=True` 且不改 kid |
| 5 | 非 bf16ws / 他 arch 透传 | kid 10200、kid 200 原样返回，splitK 保留 |
| 6 | redirect 几何 | 10210→10200 `(128,128,64)`、10213→10203 `(64,64,128)`，源目标全等；目标 ws 均 fp32_t |
| 7 | 防漂移 assert 真的会炸 | 三种人为改坏（改 redirect 目标致几何不符 / 从 STRICT 删 10216 / redirect 到另一个 bf16ws kid）**全部 AssertionError**；codegen 那条删掉 384 也炸，报文列出两边集合 |
| 8 | 与第 1 步串联 | N=300 时 10210→10200 的 plan `((4,1,256,384),'bf16_t')` → `((4,1,256,384),'fp32_t')`，**shape 不变只换 dtype**（几何相同的直接后果） |
| 9 | 回归：gfx942 heuristic | 58,080 例重跑 mismatch 0 |
| 10 | 回归：gfx950 heuristic | 1,440,000 例重跑 mismatch 0 |
| 11 | 语法 | 5 个文件 `py_compile` 全过 |

### 行为变更（本步唯一一处，范围已量化）

坑 1c 的修复**不是纯重构**。把新旧两个谓词在 `N ∈ [1,4096]` 上逐点对比：

| 项 | 结果 |
|---|---|
| 新旧谓词不一致的 N | **恰好 `[384]`**，别无其它 |
| N=384、K=4096 下三个 kid 的准入 | 10210：拒→**收**；10213：拒→**收**；10216：拒→拒（B_N=256，`padded_N != N` 先挡掉，与 exact-N 无关） |
| 其余 N 回归 | 3 kid × 4096 N × 2 K = 24,576 例，**diff 0** |

即：**两个 kid 在 N=384 重新进入 tuning 空间**。后果是该形状下 tuner 会多探两个候选、
CSV 可能变，属预期修复。之前 codegen 明明生成了 N=384 的 reduce kernel、tuner 却把它
排除在外，两个 kid 白白编进 .so 却永不被选中——不崩不报错，所以一直没被发现。

**这一处应当单独成 commit**（与本步其余纯重构部分分开），commit message 需写明
"两个 gfx942 bf16ws kid 在 N=384 重新进入 tuning 空间，相关 CSV 需重调"。

## 未做

C++ 侧转跳代码原样保留（A0 阶段两边并存、结果一致，正是第 5 步对拍能验证它搬对了的前提）。
`_GFX942_EXACT_N` 目前只有 gfx942 用；gfx950/gfx1250 无 bf16 workspace kid，无需对应物。

## 复查

```bash
grep -rn "BF16WS_EXACT_REDUCE_SHAPES" --include=*.py .    # 只剩 opus_gemm_tune.py 的注释
grep -rn "_GFX942_EXACT_N" --include=*.py .               # 1 处定义 + 2 处 import 使用
for A in gfx942 gfx950 gfx1250; do GPU_ARCHS=$A python gen_instances.py -w /tmp/gid_$A --tune; diff -rq /tmp/gic_$A /tmp/gid_$A; done   # 应全空
grep -A9 "bf16ws_exact_reduce_shape =" /tmp/gid_gfx942/impl/*bf16ws*.cuh   # 应含 N == 384
```

---

# 第 5 步 — 对拍测试

**改动文件：1 个（新建）。C++ 三个 heuristic 头一行未删，`opus_gemm()` 入口保留；
前四步的 7 个文件本步一处未动。**

## `op_tests/test_opus_heuristic_parity.py`（新建，681 行）

| 区块 | 行 | 内容 |
|---|---|---|
| 模块 docstring | `:1-55` | 为什么建 / oracle 怎么造 / 为什么只取 CSV-miss 形状 / 不在范围内 |
| 路径常量 `_THIS_DIR`.. `_ARCHES` | `:69-74` | |
| `_load(name, path)` | `:88` | 按路径直载模块，绕开 `import aiter` |
| `heur` / `common` | `:95` / `:98` | 被测的 `heuristic.py` 与 `opus_gemm_common.py` |
| `_MS` / `_NS` / `_KS` | `:109` / `:111` / `:113` | 28 × 24 × 23 形状轴 |
| `_tuned_shapes()` | `:117` | 扫 13 个 CSV 的 `libtype=='opus'` 行，取 `(M,N,K)` |
| `_shape_grid()` | `:150` | 笛卡儿积减去已调形状 |
| `_hipcc()` | `:166` | 找不到就 `pytest.skip` |
| `_codegen(arch, workdir)` | `:176` | `GPU_ARCHS=<arch> gen_instances.py -w <tmp>` |
| `_compile` / `_run_lines` / `_write_grid` | `:197` / `:209` / `:216` | |
| `_READ_GRID_CPP` | `:227` | C++ 侧读形状表，两边跑的是同一批 case |
| fixtures `workdir`/`gridfile`/`gen_dirs` | `:257` / `:262` / `:267` | 均 session 级，codegen 每 arch 只跑一次 |
| `_SIMPLE_ORACLE` / `_build_simple_oracle` | `:275` / `:287` | gfx950 + gfx1250 |
| `test_kid_parity_simple[arch]` | `:311` | |
| `_DECL_RE` / `_LAUNCHER_ARGS` / `_gfx942_declared_symbols` | `:346` / `:347` / `:353` | |
| `_build_gfx942_kid_oracle` | `:363` | 生成 dummy 定义 + 指针→kid 表 |
| `test_kid_parity_gfx942` | `:422` | |
| `_SK_BLOCK_RE` / `_splitk_kids` / `_build_gfx942_splitk_oracle` | `:464` / `:469` / `:475` | |
| `test_splitk_parity_gfx942` | `:530` | |
| `test_splitk_gfx942_default_cu_num` | `:562` | |
| `_KNOWN_UNREACHABLE` | `:597` | 第 3/4 步记录的两个不可达 kid |
| `_returned_kids(arch)` | `:608` | |
| `test_returned_kids_are_compiled[arch]` | `:625` | 即"额外断言" |
| `test_returned_kids_belong_to_arch[arch]` | `:641` | kid 不跨 arch 段 |
| `test_cpp_heuristics_still_present` | `:658` | 任务一 scope 守卫 |

### oracle 三样输入全部是**生成**的，无一手写

| oracle | 依赖 | 来源 |
|---|---|---|
| gfx950 | `opus_gemm_manifest.h`（头文件 `:41` include 它） | `_codegen("gfx950")` 现跑现产 |
| gfx1250 | 无（该头零 include） | — |
| gfx942 kid | 指针 → kid 表 | 第 4 步的 `opus_gemm_kid_symbols.json` 的 `symbol_to_kid` |
| gfx942 splitK | 那段初值语句本身 | `_SK_BLOCK_RE` 从 `impl/*.cuh` **按文本原样切出来**编译 |

splitK 这条是本步唯一需要解释的取舍：那段公式不在头文件里，是 codegen 逐 kid 吐进
launcher 的，而 launcher 要发 kernel，host-only 调不动。把公式**再敲一遍**当参照只能
证明"我能敲对两次"，所以改成把 codegen 吐出来的语句原文切出来编译。切之前断言这段
不含花括号（`:508`），否则切法本身不安全。

三个符号名 `_DECL_RE` 只认**调用形式**（行首 + `);` 收尾），否则会把
`#define OPUS_GFX942_A16W16_DECL(NAME)` 这行自己也匹配进去，得到一个叫 `NAME` 的符号。

### 为什么 gfx942 拆两个对拍

C++ `opus_a16w16_heuristic_dispatch_gfx942` 只返回 launcher 指针，splitK 不在返回值里，
同一次调用观测不到。所以 kid 与 splitK 各自成一个 test：

- `test_kid_parity_gfx942`：`CDataType` 是模板参数（`:148` `if constexpr` 用它 gate 整张
  bf16 回退表），python 侧是 `out_dtype` 字符串，两个实例化都扫。
- `test_splitk_parity_gfx942`：另外扫 batch ∈ {1,3,8} 与 cu_num ∈ {64,104,304}——这两个
  量在 kid 对拍里根本到不了（`batch` 乘进 `tiles_mn`，`cu_num` 是分子）。**只比初值**，
  后面的 clamp 循环（`:480-494`）留在 C++、本就没移植。

### 形状取 CSV-miss

heuristic 是"CSV 查不到"那条臂。13 个 CSV 里 259 条 opus 行，落在网格上的 62 组
`(M,N,K)` 被剔除；剩 **15,394 组**形状。

## 验证

| # | 项 | 结果 |
|---|---|---|
| 1 | `test_kid_parity_simple[gfx950]` | 15,394 × 2 bias = **30,788 例，mismatch 0** |
| 2 | `test_kid_parity_simple[gfx1250]` | **30,788 例，mismatch 0** |
| 3 | `test_kid_parity_gfx942` | × 2 dtype = **61,576 例，mismatch 0**；无 `-1`（指针全部映得上） |
| 4 | `test_splitk_parity_gfx942` | 15,394 × 3 batch × 3 cu × 6 splitk kid = **831,276 例，mismatch 0** |
| 5 | `test_splitk_gfx942_default_cu_num` | 6 个 launcher 吐的 fallback 全是 64 = `GFX942_DEFAULT_CU_NUM` |
| 6 | `test_returned_kids_are_compiled` | 三 arch 均无越界 kid；缺口恰为 `{1208}` / `{10003}` / `{}` |
| 7 | `test_returned_kids_belong_to_arch` | 三 arch 全部落在本段 |
| 8 | `test_cpp_heuristics_still_present` | 三个头 + `opus_gemm()` 都在 |
| 9 | 全量 | `pytest -q` → **12 passed in 6.25s** |

### 变异测试：这 12 条真的会炸

对拍测试自己得先被对拍。逐个人为改坏再跑：

| 改坏处 | 落点 | 结果 |
|---|---|---|
| python gfx950 `return 1206` → `1200` | `heuristic.py` | `kid_parity_simple[gfx950]` + `returned_kids_are_compiled[gfx950]` 挂 |
| python gfx1250 `return 20006` → `20003` | `heuristic.py` | `kid_parity_simple[gfx1250]` + `returned_kids_are_compiled[gfx1250]` 挂 |
| python gfx942 `N <= 256` → `N <= 128` | `heuristic.py` | `kid_parity_gfx942` 挂 |
| python splitK 上限 `16` → `15` | `heuristic.py:318` | `splitk_parity_gfx942` 挂 |
| python splitK 阈值 `> 16` → `> 12` | `heuristic.py:317` | `splitk_parity_gfx942` 挂 |
| python `tiles_mn` 里 `B_M` → `B_N` | `heuristic.py:301` | `splitk_parity_gfx942` 挂 |
| python `GFX942_DEFAULT_CU_NUM` 64 → 104 | `heuristic.py` | `splitk_gfx942_default_cu_num` 挂 |
| **C++** gfx1250 `return 20003` → `20000` | `.cuh` | `kid_parity_simple[gfx1250]` 挂 |

最后一条是关键：改 **C++** 一侧也能炸，说明 oracle 是真编译真执行的，不是把 python
自己的结果抄一遍。八次改坏全部当场恢复，`diff` 确认还原干净。

### `_KNOWN_UNREACHABLE` 断的是相等，不是包含

`test_returned_kids_are_compiled` 两条断言方向相反：

- `returned - defaults` 必须**空**——heuristic 返回一个没编进 .so 的 kid，运行期是
  找不到符号，不是回退。
- `defaults - returned` 必须**恰等于** `_KNOWN_UNREACHABLE[arch]`——第 3 步的 gfx950
  kid 1208（`M<=4` 臂把 nooob 镜像 gate 在 `M%64==0` 上，`M∈[1,4]` 无解）和第 4 步的
  gfx942 kid 10003 都已记录在案、按"只搬不改"未修。写成相等而不是包含，是为了让
  **新增**一个不可达 kid、或这两个**变成可达**，都当场失败。

## 观察

`gen_instances.py` 在 codegen 期已断言 `HEURISTIC_DEFAULT_KIDS.issubset(S)`；本步补的是
反向那一半——python heuristic 的值域不越出 `HEURISTIC_DEFAULT_KIDS`。两条合起来才闭环。

session 级 fixture 让三次 codegen（每次约 1s）+ 四次 hipcc 各只跑一遍，全量 6.25s。
无需 GPU：oracle 是 host-only `main()`，hipcc 只是因为头文件链一路 include 到
`hip/hip_runtime.h`（用 `g++` 直接编会 fatal error）。

## 未做

C++ 侧一行未删——删除是任务二的事，`test_cpp_heuristics_still_present` 就是把这条
写死成断言，免得任务二动手时这个对拍悄悄变成空跑。

## 复查

```bash
cd /root/workspace/0810/aiter
python3 -m pytest op_tests/test_opus_heuristic_parity.py -q        # 12 passed
python3 -m pytest op_tests/test_opus_heuristic_parity.py -v        # 逐条
git status --short                                                  # 只多一个 ?? op_tests/test_opus_heuristic_parity.py
```

---

# 第 5 步（补） — 对拍加 CSV-hit corpus

**改动文件：1 个。只动第 5 步自己新建的那个测试，其余文件一处未动。**

验收清单第一条写的是"三 arch **CSV hit/miss corpus**"，原第 5 步只扫了 miss
（tuned 形状被 `_tuned_shapes()` 剔掉了）。补上 hit，两个 corpus 都扫。

## `op_tests/test_opus_heuristic_parity.py`（681 → 723 行，`+50/-8`）

| 改动 | 行 | 内容 |
|---|---|---|
| 模块 docstring `SHAPES` 节 | `:45-58` | 由"只取 CSV-miss"改写为两个 corpus 各自的定义与理由 |
| `_tuned_shapes()` → `_tuned_rows()` | `:126` | 改为返回整行，`(M,N,K)` 的提取下放给 `_hit_shapes` |
| `_hit_shapes()` | `:149` | 新增，tuned 行去重后的 `(M,N,K)` |
| `_miss_shapes()` | `:171` | 原 `_shape_grid` 的主体（合成网格减 tuned） |
| `_shape_grid()` | `:183` | 改为 `miss + hit`，四个对拍全部经它取形状 |
| `test_corpora_are_both_populated` | `:290` | 新增，见下 |

四个对拍一行未改——它们本来就调 `_shape_grid()`，corpus 变大自动生效。

### 为什么 hit corpus 也要扫

runtime 上 tuned 形状走 lookup 表、不走 heuristic，但：两份实现在**任何**输入上都该
相等；且 tuned 形状随时会退化成 heuristic 形状——CSV 行被删、换 arch 重调、或者用
该行没覆盖的 bias / out_dtype 组合来查。

### `test_corpora_are_both_populated`

CSV 路径变了、或 `libtype=='opus'` 过滤器失配，hit corpus 会静默变空，四个对拍照样
全绿——只是不再测 tuned 形状了。所以断言两个 corpus 都非空、不相交、且
`_shape_grid() == miss + hit`。

### 形状数变化

| corpus | 组数 |
|---|---|
| CSV-miss | 15,394 |
| CSV-hit | 259（13 个 CSV 共 259 条 opus 行，`(M,N,K)` 恰好两两不同） |
| 合计 | **15,653** |

## 验证

| # | 项 | 结果 |
|---|---|---|
| 1 | gfx950 kid | 15,653 × 2 bias = **31,306 例，mismatch 0** |
| 2 | gfx1250 kid | **31,306 例，mismatch 0** |
| 3 | gfx942 kid | × 2 dtype = **62,612 例，mismatch 0** |
| 4 | gfx942 splitK | × 3 batch × 3 cu × 6 kid = **845,262 例，mismatch 0** |
| 5 | 全量 | `pytest -q` → **13 passed in 6.33s** |

### 顺带查明：第 4b 步的行为变更不会动现有 CSV

第 4b 步记过"N=384 处两个 bf16ws kid 重新进入 tuning 空间，相关 CSV 需重调"。
现在有了 hit corpus，把 CSV 里 bf16ws 那些行的 N 逐个对 `_GFX942_EXACT_N` 查了一遍：

| kid | 行数 | 出现的 N | 落在 exact-N 之外的 |
|---|---|---|---|
| 10210 | 24 | 512 / 1024 / 2048 | **无** |
| 10213 | 0 | — | — |
| 10216 | 7 | 256 / 2048 | **无** |

31 行 bf16ws 全部合法，转跳规则一条都不触发。即第 4b 的修复**不改变任何现有 CSV
行的行为**，只是让 N=384 这个此前被误排除的形状能重新被 tune 到。前面那句"CSV 需
重调"应理解为"N=384 形状值得重调"，而非"现有行会失效"。

---

# A0 提交前 — 四项验收

四项逐条跑过，全绿。基线 `ca68b4f3`。

| # | 项 | 结果 |
|---|---|---|
| 1 | 三 arch CSV hit/miss corpus 的 python kid 与 C++ 一致 | ✅ 15,653 形状 × 三 arch；gfx950 31,306 / gfx1250 31,306 / gfx942 62,612 例，**mismatch 全 0** |
| 2 | gfx942 auto splitK 初值一致 | ✅ **845,262 例，mismatch 0**；cu_num fallback 64 与 6 个 launcher 吐的字面量一致 |
| 3 | codegen 产物无变化 | ✅ **9/9 相同**（3 模式 × 3 arch），唯一新增物是 `opus_gemm_kid_symbols.json` |
| 4 | workspace 一行没动 | ✅ `opus_gemm.cu` 与 `include/` diff 全空；全 diff 中 workspace 机构行 **0** |

## 第 3 项的做法

不拿之前遗留的临时目录当基线——`git archive HEAD` 把 `csrc/opus_gemm` 与
`aiter/configs` 导到 `/tmp/a0base`，用**未改动的那份源码**现跑一遍，再和当前树对比。
三种模式各跑，因为它们走的代码路径不同：

| 模式 | 覆盖 |
|---|---|
| 无参 | 默认 subset（只有 `HEURISTIC_DEFAULT_KIDS`） |
| `--tune` | 全 kid 枚举（tuner 用的那条） |
| `--tune_files <真实 CSV>` | lookup 表 + CSV 撑开的 subset S；instances 数 gfx942 23 / gfx950 65 / gfx1250 8 |

```bash
rm -rf /tmp/a0base && mkdir -p /tmp/a0base
git archive HEAD csrc/opus_gemm aiter/configs | tar -x -C /tmp/a0base
for A in gfx942 gfx950 gfx1250; do
  (cd /tmp/a0base/csrc/opus_gemm && GPU_ARCHS=$A python3 gen_instances.py -w /tmp/a0_base_$A)
  (cd csrc/opus_gemm            && GPU_ARCHS=$A python3 gen_instances.py -w /tmp/a0_now_$A)
  diff -r -x opus_gemm_kid_symbols.json /tmp/a0_base_$A /tmp/a0_now_$A
done
```

### 唯一的产物差异：`opus_gemm_kid_symbols.json`

第 4 步加的测试/工具用副产品。确认它对构建完全惰性：

| 查 | 结果 |
|---|---|
| 有没有 TU include 它 | `grep --include=*.{cu,cuh,h,cpp}` **零命中** |
| 除发射处外谁读它 | 只有 `op_tests/test_opus_heuristic_parity.py` |
| 会不会被构建的 source glob 扫到 | 不会，glob 只收 `.cu`/`.cuh` |

排除它之后，9 次比对**逐字节相同**。

## 第 4 项的做法

三条独立证据：

1. `git diff --stat -- csrc/opus_gemm/opus_gemm.cu csrc/opus_gemm/include` → **空**。
   `hipMalloc` 注册表、`opus_splitk_ws_get/_device_handle/_sync_to_device`、
   launcher 里的 `ws_bytes` 计算全在这两处，都没碰。
2. 全 diff 里 grep `hipMalloc|hipFree|opus_splitk_ws|ws_handle|ws_bytes|workspace_init`
   → 增删行 **0**。唯一命中是 `heuristic.py` 注释里的 `torch.empty` 字样（描述 A1 要
   做什么，不是代码）。
3. `gen_instances_gfx942.py` 是唯一"既被改、又含 ws 代码"的文件，逐 hunk 看过：一个
   import、一条防漂移 assert、一处把 emit 来源从 `EXACT_N_ROWBLOCK_REDUCE_CONFIGS`
   换成 `_GFX942_EXACT_N`（附注释）。ws 相关一行未动——第 3 项产物逐字节相同也
   反过来印证了这点。

## A0 交付物

| 文件 | 性质 |
|---|---|
| `csrc/opus_gemm/opus_gemm_common.py` | 改 `+253/-0`（kid 查询 API、`_GFX942_EXACT_N` 单一来源、转跳规则） |
| `csrc/opus_gemm/gen_instances.py` | 改 `+76/-0`（吐 kid symbol map） |
| `csrc/opus_gemm/opus_gemm_tune.py` | 改 `+6/-10`（删掉漏 N=384 的错副本） |
| `csrc/opus_gemm/codegen/gen_instances_gfx942.py` | 改 `+21/-6`（防漂移 assert + emit 来源） |
| `aiter/ops/opus/common.py` | 改 `+43/-0`（`resolve_bf16ws_kid` 懒加载桥） |
| `aiter/ops/opus/gemm_op_a16w16.py` | 改 `+29/-6`（explicit / tuned 两条路接转跳） |
| `aiter/ops/opus/heuristic.py` | **新建 475 行**（三 arch heuristic + auto splitK） |
| `op_tests/test_opus_heuristic_parity.py` | **新建 723 行**（对拍） |

合计 6 改 + 2 新建，`+433/-20` 加 1,198 行新文件。

## A0 未做（按设计留给 A1 / 单独 commit）

| 项 | 去处 |
|---|---|
| workspace 从 C++ registry 迁到 torch | A1，这正是任务一的正题 |
| 删 C++ 三个 heuristic 头、删 `opus_gemm()` | 任务二 |
| N=384 那处行为变更 | **单独 commit**，message 需写明两个 gfx942 bf16ws kid 在 N=384 重新进入 tuning 空间（现有 CSV 行不受影响，见第 5 步补节） |
| 两个已记录的不可达 kid（gfx950 1208、gfx942 10003） | 只记录不修；对拍已把它们断言成**相等**，变了会炸 |

---

# A1 第 6 步：拆掉 `D_C` 的多角色，落成 `D_ACC` / `D_WS` / `D_OUT`

`D_C` 一个名字同时当"累加类型"和"workspace 元素类型"用，是 workspace 尺寸算错的
根因。这一步把三个 arch 的 traits 里的角色拆开，并把每个使用点重新指向唯一正确的
那一个。**workspace 容量、pointer cast、reduce 输入现在只引用 `D_WS`。**

## 三个角色的定义

| 别名 | 含义 | 谁有 |
|---|---|---|
| `D_ACC` | 寄存器累加类型，恒为 fp32 | 三 arch 全有，全部加了 static_assert |
| `D_WS` | split-K workspace **元素**类型 | 只有 splitK traits 有 |
| `D_OUT` | 最终 C 的元素类型 | 只有非 splitK traits 有 |

gfx942 是唯一例外：DTYPE slot 2 按实例化**双角色**（splitK kid 里它是 workspace
元素类型，非 splitK kid 里它是 C 元素类型），所以两个名字都导出、指同一个 slot，
由 kid 决定哪个有意义。这不是妥协——4 个 dual-mode pipeline 在 `IS_SPLITK` 两个
分支里各取各的名字，读代码时角色一眼可辨。

## 计划外的发现：gfx1250 的第四个别名

计划只点了 `D_C`。实际上 gfx1250 的两个 splitk pipeline
（`..._cluster_tdm_splitk_ws_gfx1250.cuh` / `..._clusterlaunch_tdm_splitk_ws_gfx1250.cuh`）
是用 **`DataAcc`** 去 cast workspace 的：

```cpp
DataAcc* ws_ptr = reinterpret_cast<DataAcc*>(kargs.ws_handle->ptr);   // 旧
D_WS*    ws_ptr = reinterpret_cast<D_WS*>(kargs.ws_handle->ptr);      // 新
```

`sizeof(DataAcc)` 算 gmem bounds 那处一并改了。不动这里的话，gfx1250 会原样留着
这一步要消灭的那个 bug——只不过换了个别名拼写。

## 计划与代码树的漂移

计划写的是 `gen_instances_gfx942.py:521`，树上实际在 `:538`。已按内容定位，不按行号。

## 逐 arch 改动

### gfx1250（splitK-only）
- traits 模板形参 `typename D_C_` → `typename D_WS_`；新增
  `static_assert(is_same<D_WS, float>)` 与 `static_assert(is_same<D_ACC, float>)`。
- `kCVec` 从 `sizeof(D_WS)` 算（这条 store 打的就是 workspace）。
- 两个 pipeline 的 workspace cast + bounds 换 `D_WS`（见上）。
- `splitk_reduce_gfx1250.cuh`：`using D_WS = float;`，reduce 输入按 `D_WS` 命名，
  `sizeof(float)` → `sizeof(D_WS)`。
- codegen traits slot 3 由 `D_C` 改成写死 `fp32_t`，并注明这是 D_WS 槽位、不是
  launcher 的 D_C dispatch dtype（两者今天恰好都是 fp32_t，但角色不同）。

### gfx950（1 个 splitK + 4 个非 splitK traits）
- splitK traits：`D_C` → `D_WS` + 两条 static_assert。
- 4 个非 splitK traits：`D_C` → `D_OUT`，每个前面标一句
  `// Non-split-K trait: no workspace, hence no D_WS.`
- 7 个非 splitK pipeline 整体重命名到 `D_OUT`；splitk pipeline 的 ws cast 换 `D_WS`。
- reduce 与 gfx1250 同样处理。

### gfx942（唯一有 bf16 workspace kid 的 arch）
- traits 导出 `D_OUT` 与 `D_WS` 两个名字（同一 slot，见上）+ `D_ACC` static_assert。
- helpers 按角色切：`epilogue_store_c_if` / `epilogue_store_c_lds_staged` → `D_OUT`；
  `epilogue_store_workspace_sc0nt` → `D_WS`。
- 3 个 dual-mode pipeline（kbuf1 / kbuf2v / kbuf2v_bk128）两个名字都取，`g_c` lambda
  两个分支各指各的。`kbuf2v_bk128` 的 `bk64_traits_view` 转发结构体要一并转发两个名字。
- `quad_mfma32_kbuf1` 需要特殊处理：它两种模式**共用同一条 store 路径**
  （LDS-staged drain + predicated tail，靠 `IS_SPLITK` 选 `stride_out`），
  逐处改名会把同一条语句劈成两份。改成在顶上一次性选定：
  ```cpp
  using D_STORE = std::conditional_t<IS_SPLITK, D_WS, D_OUT>;
  ```
- codegen `ws_bytes`：`sizeof(typename Traits::D_C)` → `sizeof(typename Traits::D_WS)`。

## 有意没动的

- 三个 codegen 里 launcher 侧的 `template <typename D_C>`。那是 **tune-dispatch 表
  选择器**，跟 traits 的元素类型别名是两回事，混为一谈正是这一步要防的。
- a8w8 全家（`gfx950/opus_gemm_traits_a8w8_{no,}scale_gfx950.cuh`、`gfx942/a8w8/`）。
  它没有 workspace，traits 独立，`D_C` 在那里只有一个角色，名副其实。
- gfx950 第一个非 splitK traits 里我一度加了 `using D_C = D_OUT;` 向后兼容别名，
  随即删掉——留着这个名字就等于留着这一步要消灭的那份歧义。

## 验证

### 1. 产物逐字节比对（沿用 A0 的口径）

```bash
for A in gfx942 gfx950 gfx1250; do
  (cd csrc/opus_gemm && GPU_ARCHS=$A python gen_instances.py -w /tmp/s6_$A --tune)
  diff -rq /tmp/gid_$A /tmp/s6_$A
done
```

| arch | 差异文件 | 内容 |
|---|---|---|
| gfx1250 | 6 | 只有 traits slot 行 + ws sizing 行 |
| gfx950 | 6 | 只有 ws sizing 行 |
| gfx942 | **7，全是 splitK kid** | 只有 ws sizing 行 + 新注释 |

gfx942 这 7 个逐个看过（含 2 个 bf16ws kid），diff 严格限于：

```diff
-                * (size_t)padded_M * (size_t)padded_N * sizeof(typename Traits::D_C);
+                * (size_t)padded_M * (size_t)padded_N * sizeof(typename Traits::D_WS);
```

**这一步语义上是 no-op**：`D_WS` 和旧的 `D_C` 指的是 traits 的同一个 slot，
所以 bf16ws kid 现在算出来的字节数和改之前一样（都是 bf16）。这一步只把"这个尺寸
该由谁决定"钉死，为第 7 步以后 python 侧按 kid 算容量铺路。

### 2. 编译

产物 diff 只覆盖默认 15 个 kid，**`quad_mfma32` 的 10006/10216 不在里面**——而那
恰好是唯一需要 `D_STORE` 特判的 pipeline。单独强制生成后两种模式各编一遍：

```bash
echo '[10006, 10216]' > /tmp/s6_q/compiled_kids.json
GPU_ARCHS=gfx942 python gen_instances.py -w /tmp/s6_q --compiled_kids_sidecar /tmp/s6_q/compiled_kids.json
```

| 目标 | device TU | host TU |
|---|---|---|
| gfx942（15 kid + quad 两模式，共 19 个 TU） | 全过 | 过 |
| gfx950（14 个 TU） | 全过 | 过 |
| gfx1250（8 个 TU） | 本机 toolchain 不支持该 target ID | 过（`--offload-host-only`） |

唯一的 warning 是既有的（`duplicate 'inline'`、`loop not unrolled`、occupancy），
都不在改动行上。gfx1250 编不了是环境限制（HIP 7.0 / clang 20 不认 `gfx1250`
target ID），不是改动引入的；host pass 覆盖了 traits 实例化和 ws_bytes 表达式。

### 3. 残留 `D_C` 清扫

a16w16 范围内的头文件已清空。剩下的三处注释同步改了措辞
（`opus_gemm_arch_gfx950.cuh:117,164`、`gen_instances.py:443,552`、
`opus_gemm_common.py:1288`），并复跑 codegen 确认**产物零变化**。
`opus_gemm_heuristic_dispatch_gfx942.cuh:22` 的 `template <typename D_C>` 保留——
那是 launcher dispatch dtype，见上。

---

# 第 6b 步 — 两套 launcher 签名、两张生成表

## 结论

`opus_gemm.cu` 里那套「按 kid 数值区间判断要不要 workspace」的谓词全部删除，
换成查 codegen 生成的 per-arch split-K 表。ABI 上非 split-K 一行未动。

## ABI

`include/opus_gemm_launcher_abi.cuh`（新建）里定义两个类型，不合并：

```cpp
using OpusA16W16Kernel          = void (*)(T&, T&, T&,     std::optional<T>, int);
using OpusA16W16WorkspaceKernel = void (*)(T&, T&, T&, T&, std::optional<T>, int);
```

`workspace` 是 `aiter_tensor_t&` 不是 `std::optional`：split-K kid 没它跑不了，
不存在「忘了传还能编过」的签名。

第三个类型 `OpusA16W16Dispatch` 是二选一容器，两个 implicit 转换构造函数。
它存在的唯一理由是 gfx942 heuristic 里那 ~20 条 `return <launcher><T>;` ——
其中 plain 和 split-K 混着返回，没有这个容器就得逐条改写，
连带 A0 那个靠正则抓 decl 的 parity oracle 一起失效。

原来 `OpusA16W16NoscaleKernel` 这个 typedef 在 gfx942/gfx950 两个 heuristic 头里
各写了一份，用 `#ifndef` 互相躲——两份函数指针 typedef 一旦漂移就是链接器抓不到的
ODR 违例。现在都 include 同一个头，名字也统一成 `OpusA16W16Kernel`。

## 生成的表：两族 × 分区

`gen_a16w16_tune_lookup()` 出两族宏：

| 宏 | 类型 | 分区依据 |
|---|---|---|
| `GENERATE_A16W16_TUNE_LOOKUP_{BF16,FP32}(CTYPE)` | `OpusA16W16Kernel` | output dtype（kid 缺 `<CTYPE>` 实例化会链接失败） |
| `GENERATE_A16W16_WS_TUNE_LOOKUP_{GFX942,GFX950,GFX1250}` | `OpusA16W16WorkspaceKernel` | **arch** |

WS 表不带 `(CTYPE)` 参数：split-K launcher 只有 `<fp32_t>` 一种实例化。

**按 arch 分区是 #4246 没做完的地方**——它只改了 gfx1250。三个 arch 现在都有自己的
宏，且**恒定发射**（空表只有哨兵），所以每个 `opus_gemm_arch_<arch>.cuh` 可以无条件
引用自己那张。fat binary 里 gfx942 的 kid 在 gfx950 设备上必须 miss，
而不是解析到一个这台设备跑不了的 launcher。

运行时 `(M,N,K)` 表同样按族劈开（`gen_lookup_dict()`），
出 `GENERATE_OPUS_LOOKUP_TABLE_*` + `GENERATE_OPUS_WS_LOOKUP_TABLE_*` 四张。
一个 shape 每种 outdtype 至多一个 winner，所以它只会在其中一张里，两张都查不歧义。

四张表全部以哨兵打头（kid 表 `{-1, nullptr}`，shape 表 `{{-1,-1,-1}, nullptr}`）：
劈开之后空表两个方向都可达——gfx1250-only 构建没有任何 plain 行（它每个 kid 都是
split-K），而 `Entry t[] = {};` 是 ill-formed。kid 和 extent 都非负，哨兵永不命中。

## 删掉的谓词

| 谓词 | 处置 |
|---|---|
| `opus_kid_is_splitk` | 删。改查 WS 表命中 |
| `opus_kid_is_persistent` | 删（grep 确认无调用者） |
| `opus_kid_is_mono_tile` | 删（同上） |
| `opus_kid_is_gfx1250_splitk` | 删（同上） |
| `opus_kid_is_gfx942_splitk` | 删。原来只用于「gfx942 split-K 不支持 bias」，现在写成 `needs_workspace && arch == Gfx942` |
| `opus_kid_is_a16w16_sb` | **保留** |
| gfx950 头里的 `kid_is_splitk` / `kSplitkKidMin` / `kSplitkKidMax` | 删 |

`opus_kid_is_a16w16_sb` 保留是有理由的：它回答的是 **bias 能力**，不是 launcher
arity。没有哪张生成表以「支不支持 bias」为键，且答错只是一条清晰的运行时报错，
不是打错 arity 的函数指针调用。

## raw ABI 与两向校验

```cpp
void opus_gemm_a16w16_tune(
    aiter_tensor_t& XQ, aiter_tensor_t& WQ, aiter_tensor_t& Y,
    std::optional<aiter_tensor_t> bias,
    std::optional<aiter_tensor_t> workspace,   // 无 C++ 默认值
    int kernelId, int splitK);
```

两个方向都拦：

- split-K kid 传了 `None` → 报错。
- 非 split-K kid 传了 tensor → 也报错。这不是洁癖：说明调用方替一个没人读的 buffer
  算了尺寸、分配了显存，它对这个 kid 属于哪一族的判断在别处同样是错的。

pybind 那层 `py::arg("workspace") = std::nullopt` 带默认值，只因为 Python 不允许
无默认参数排在有默认参数后面；C++ body 的两向校验让这个默认值没法掩盖漏传。

## python 公共 wrapper

`opus_gemm_a16w16_tune(XQ, WQ, Y, bias=None, kernelId=0, splitK=0, *, workspace=None)`

`workspace` 是 keyword-only 末位参数。旧的 positional 调用
`(XQ, WQ, Y, bias, kernelId, splitK)` 和那个 `(XQ, WQ, Y, kid, splitK)` 简写
（wrapper 里靠「第 4 个位置参数是 int」识别）都原样可用。

## `opus_gemm()` 入口（`:198-230`）

它拿不到 workspace（torch-free，自己分配不了），所以现在只能跑 plain 族，
拿到 split-K dispatch 就 `AITER_CHECK` 报错并指向正确入口。第 12 步 python 侧
会绕开这条路（自己定 kid + 分配 workspace 走 tune 入口），
`opus_gemm()` 本身的删除属于任务二。

## 顺手修的一个真 bug

kid **10204**（`a16w16_em3en4_lds1_pgr2_sk`）在 `GFX942_SPLITK_TAGS` 里
（所以 emitter 给了它 workspace arity），却不在 `gen_instances.py:SPLITK_TAGS` 里
（所以表生成器把它放进了 plain FP32 表）。查生成产物时发现的。

修法是把 `codegen/common.py:_GFX942_SPLITK_ONLY` 立为唯一源头，
`GFX942_SPLITK_TAGS = _SPLITK`，两个集合互相加了引用注释。
产物里 10204 现在只出现在 `GENERATE_A16W16_WS_TUNE_LOOKUP_GFX942`。

`opus_gemm_kid_symbols.json` 里这个 kid 的 `is_splitk` 从 `false` 翻成 `true`，
是三个 arch 产物 diff 中唯一一处非 arity 的语义变化。

## 记下的一个坑（第 12 步会撞上）

gfx942 bf16ws launcher 在 host 侧会改调 fp32ws 双胞胎
（`codegen/gen_instances_gfx942.py` 的 `bf16ws_host_redirect`）。
加了 workspace 参数之后，这个 redirect 把**我们的 buffer** 交给了一个元素宽度
4 字节的 launcher，而我们按 2 字节算的容量。

所以 `make_workspace_plan()` 对 bf16ws kid 必须按 **redirect 目标**（fp32）定容，
只要 shape 可能落在 `bf16ws_exact_reduce_shape` 之外。旧的按需增长 registry 里
callee 自己会 resize，这事看不见。注释已经写在 codegen 生成点上。

## 验证

### 1. 产物逐字节比对

三个 arch 全部重新生成并与基线 diff。差异只有三类：
step 6 的 `D_WS` sizing 行、本步的 `aiter_tensor_t &workspace,` arity 行 +
注释、以及表劈分 / 哨兵 / `<CTYPE>`→`<fp32_t>` 的宏内容。无夹带。
10204 的 `is_splitk` 翻转如上。

### 2. 编译

`-fsyntax-only`（本机 gfx950 toolchain）：

| TU | gfx942 | gfx950 | gfx1250 | 三 arch fat |
|---|---|---|---|---|
| `opus_gemm.cu` | 过 | 过 | 过 | 过 |
| `all_instances_host_<arch>.cu` | 过 | 过 | 过 | — |

只剩既有的 4 条 `aiter_tensor.h` nodiscard warning，不在改动行上。

### 3. parity 测试

`op_tests/test_opus_heuristic_parity.py` 13 项全过。

oracle 改了三处才过：`_WS_DECL_RE` 抓第二族 decl、`_WS_LAUNCHER_ARGS` 给它们
正确的 arity、`kid_of()` 收 `OpusA16W16Dispatch` 并按族取指针。
不改的话那 7 个 split-K launcher 会被静默漏掉，
所有返回它们的 heuristic 结果统一变成 `-1`——看起来像 parity 失败，
实际是 oracle 坏了。这条已写进测试注释。

---

# 第 7 步 — reduce ABI 单点改动

## 改了什么

`csrc/opus_gemm/gen_instances.py` 的 `SPLITK_REDUCE_ABI_MAP`：三个 arch 条目的
`ws_arg` 由 `const opus_splitk_ws_handle* ws_handle` 改成 `void* ptr_ws`，
`ws_type` 由 `const opus_splitk_ws_handle*` 改成 `void*`。这张表同时驱动
fused host TU 里的 forward decl（`:949`）和显式实例化行的 `ws_ptr_type`
（`:1061`），所以一处改动三个 arch 的签名一起变。

`void*` 而不是 `const D_WS*`：workspace 元素类型是 reduce kernel 的模板参数
（gfx942 把同一个 kernel 在 float 和 `__bf16` 两种 workspace 上都实例化了），
而这个字符串同时用在按 (D_OUT, HAS_OOB) 组合发出的显式实例化行上，各行必须逐字
一致。每个 kernel 在函数体开头 reinterpret_cast 回自己的 `D_WS`。

## 超出计划的那部分

只改这张表是不够的。同一个 ABI 字符串在 per-arch codegen 模块里还有 **五处硬编码
镜像**，不走这张表：

- `codegen/gen_instances_gfx1250.py:61,64`（额外实例化）、`:241`（forward decl）
- `codegen/gen_instances_gfx942.py:168,175`（forward decl）、`:187,190,198`（实例化）

计划把这些归在第 11 步。但留着的话，生成出来的文件里 forward decl 写 `void*`、
显式实例化写 `const opus_splitk_ws_handle*`——一个纯粹坏掉、没有任何好处的中间态。
所以这五处在第 7 步一起改了。

## 验证

三个 arch 重新生成，diff 只有 ABI 那几行。此时 device TU 还编不过
（`opus_splitk_ws_handle` 尚未删除，第 8 步才动），属于预期中间态。

---

# 第 8 步 — 删 handle 结构体，kargs 换 `void* ptr_ws`

顺序按计划 gfx1250 → gfx950 → gfx942。

## 改了什么

三份 traits 头里的 `opus_splitk_ws_handle` 定义（靠 `#ifndef
OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED` 互防）整体删除，guard 一起删；
kargs 的 `ws_handle` 字段换成 `void* __restrict__ ptr_ws`：

| 文件 | 删 | 改 |
|---|---|---|
| `gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh` | struct + guard | 头注释、kargs 注释、字段 |
| `gfx950/opus_gemm_traits_a16w16_gfx950.cuh` | struct + guard | 注释、字段 |
| `gfx942/a16w16/opus_gemm_traits_a16w16.cuh` | struct + guard | 字段 |

`include/opus_gemm_common.cuh:24` 的注释同步（去掉 "+ opus_splitk_ws_handle
（guarded; shared with gfx950）"）。

## 和计划不一致的一处：`opus_splitk_ws_ptr<>()` 留下了

计划说整个删掉，把 `readfirstlane` 广播就地写进各 pipeline。**没有照做**，
只改了参数类型：

```cpp
template <typename D_WS>
__device__ __forceinline__ D_WS* opus_splitk_ws_ptr(void* __restrict__ ptr_ws)
```

理由：这个 helper 有 **七个调用点**（五条 pipeline + 两条 reduce 路径）。
删掉它等于把同一段 intrinsic 序列抄七遍才能保住优化，而共享 helper 存在的意义
恰恰就是防这种抄写漂移。

优化本身的性质也在注释里写清楚了：参数原本是 `const opus_splitk_ws_handle*`，
编译器无法证明 `->ptr` 这次 load 是 uniform 的，所以需要 readfirstlane 对把结果
逼进 SGPR，让后续 buffer 寻址保持 scalar。`ptr_ws` 现在是普通 kernarg，
天然 uniform，在当前 LLVM 上这段已经不产生额外指令了——留着是因为**这个 codegen
保证本身是目的**：下游任何改动都不应该有办法把这个地址变成 vector 值。

## 验证

三份 traits 头单独 `-fsyntax-only` 各过；三份放进同一个 TU 加 `sizeof`
static_assert 一起编，0 error——这条是专门证明删 guard 是安全的
（原本 guard 就是为了三份定义共存）。

readfirstlane 用独立探针 kernel 验了一次：2 × `v_readfirstlane_b32`，没丢。
（真 kernel 上的对照见第 10 步。）

reduce 头此时报 `unknown type name 'opus_splitk_ws_handle'`，第 10 步解决。

---

# 第 9 步 — pipeline 侧的 cast

每文件一行，数值算法零改动。

| 文件 | 行 |
|---|---|
| `gfx1250/..._cluster_tdm_splitk_ws_gfx1250.cuh` | 387 |
| `gfx1250/..._clusterlaunch_tdm_splitk_ws_gfx1250.cuh` | 455 |
| `gfx950/..._flatmm_splitk_gfx950.cuh` | 265-266 |
| `gfx942/..._em3en4_lds1_pgr2_sk.cuh` | 584 |
| `gfx942/..._kbuf1.cuh` | 76 |
| `gfx942/..._kbuf2v.cuh` | 80 |
| `gfx942/..._kbuf2v_bk128.cuh` | 118 |
| `gfx942/..._quad_mfma32_kbuf1.cuh` | 284 |

gfx1250 / gfx950 三处：`reinterpret_cast<D_WS*>(kargs.ws_handle->ptr)` →
`reinterpret_cast<D_WS*>(kargs.ptr_ws)`。gfx950 那处顺带把注释
"Deref the handle slot at entry; survives a post-capture grow." 改成
"Caller-allocated workspace; the pointer is fixed for this launch."——
旧注释描述的是已经不存在的机制。

gfx942 五处：`opus_splitk_ws_ptr<D_WS>(kargs.ws_handle)` →
`opus_splitk_ws_ptr<D_WS>(kargs.ptr_ws)`。

## 顺手修的一个第 6b 引入的回归

这一步是这些 TU 第一次编到足够深，于是暴露出：

```
explicit instantiation of 'gemm_a16w16_em3en4_lds1_pgr2_sk_kernel' does not
refer to a function template ... candidate template ignored: substitution
failure: too many template arguments
```

第 6b 为了修 kid 10204 的分表，把 `a16w16_em3en4_lds1_pgr2_sk` 加进了
`codegen/common.py:_GFX942_SPLITK_ONLY`（这一步是对的），于是它进了 `_SPLITK`。
但 `_kargs_template_vars()` 当时是拿 `kernel_tag in _SPLITK` 来判断
"要不要把 kargs 结构体作为第二个显式模板实参发出去"，于是给一个声明成
`template<typename Traits>` 的 kernel 发了两个模板实参。

**"是不是 split-K" 和 "device kernel 模板有没有 Kargs 参数" 是两个问题。**
新增 `_KARGS_TEMPLATED`，只按 kernel 模板的 arity 定义，绝不从 `_SPLITK` 派生：

```python
_KARGS_TEMPLATED = tuple(W3_KERNEL_PAIRS.values()) + ("a16w16_kbuf1_sk",)
```

`_kbuf1_sk` / `_kbuf2v_sk` / `_kbuf2v_bk128_sk` / `_quad_mfma32_kbuf1_sk`
和各自的非 split-K 双胞胎共用一个 kernel 模板，两者只差 kargs 类型——这正是
Kargs 是模板参数的原因。`em3en4_lds1_pgr2_sk` 没有双胞胎，它的 kernel 是
`template<typename Traits>` 按名字收 `opus_gemm_splitk_kargs`。这段推理写进注释了。

## 验证

s8→s9 产物 diff 只动 3 行，全是 em3en4 的——证明修得干净，没有波及别的 kid。
19 个 split-K main kernel TU + 15 个非 split-K TU 编译通过。

ISA 抽取这一步做不了（要整个 TU 编得过，reduce 头还没迁移），推迟到第 10 步。

---

# 第 10 步 — reduce

| 文件 | 改动点 |
|---|---|
| `gfx950/splitk_reduce_gfx950.cuh` | 3 |
| `gfx1250/splitk_reduce_gfx1250.cuh` | 5 |
| `gfx942/a16w16/splitk_reduce_gfx942.cuh` | 9 |

gfx942 最重，如计划所说：baseline / OOB 两套 kernel（`fallback_body`、
`fallback`、`bf16ws_fallback`、`exact_n_rowblock` 四个签名）+ 两个转发层调用点，
外加两处 `opus_splitk_ws_ptr<const D_WS>(ws_handle)` → `(ptr_ws)`。

gfx1250 的头注释重写过：两个 arch 的 workspace ABI 现在完全一样
（都是 `void* ptr_ws`），**名字是唯一还在防止它们在 multi-arch build 里撞车的东西**。
旧注释把 "distinct ws_handle ABI" 也算作隔离手段，现在不成立了。

## 一个预测错误的更正

我此前告诉过你"到第 11 步之前一直编不过"。实际上第 10 步做完，**37 个 device TU
全部干净通过**——剩下的 `opus_splitk_ws_get` extern 住在生成的 host TU 里，
不在 device TU 里。

## 验证

### 编译

| arch | device TU 数 | error |
|---|---|---|
| gfx942 | 17 | 0 |
| gfx950 | 13 | 0 |
| gfx1250 | 7 | 0 |

`ws_handle` 在 `csrc/opus_gemm/include/` 下归零（唯一剩下的文本命中是
`opus_splitk_ws_ptr` 那段说明为什么留着它的注释散文）。

### readfirstlane 对照（补第 8/9 步欠的）

这次在真 kernel 上做，并且 `git stash` 出改动前的基线逐一对照：

| TU | 改后 | 基线 |
|---|---|---|
| `splitk_p1_..._Cfp32_t` | 5 | 5 |
| `splitk_em3en4_lds1_pgr2_...` | 5 | — |
| `splitk_legacy_bf16ws_...` | 5 | — |
| `splitk_reduce_gfx942` | 276 | 276 |

完全一致。stash 已 pop，`git stash list` 空。

### 产物

与基线逐字节相同——本步只动 header。这本身就是"没有东西漏进 codegen"的检查。

### parity

13/13。

---

# 第 11 步 — codegen 删 grow 块

## 改了什么

三个 per-arch emitter 里的 workspace 增长块整体删除：per-stream registry
的 `extern` 声明（`opus_splitk_ws_get` / `opus_splitk_ws_device_handle` /
`opus_splitk_ws_sync_to_device`）、`hipStreamIsCapturing` + `capturing` 判定、
`ws_bytes` 公式、4MiB 对齐的 `hipMalloc` / `hipDeviceSynchronize` / `hipFree`、
以及 gfx942 / gfx1250 的 device handle 镜像。

换成三行：算出所需元素数 → 校验传进来的 workspace tensor → 取 `data_ptr()`。

**三个 `ws_bytes` 公式一起消失。** 容量计算的唯一权威移到 python 侧的
`kid_workspace_plan()`（第 12 步落地）。

`auto stream = aiter::getCurrentHIPStream();` 原本在被删的块里面，但 kernel
launch 还要用，三个 arch 都把它单独留了下来。

## 每个 arch 原有的 split_k clamp 全部保留

按你的要求，只删 workspace 相关。三处 clamp 逻辑一字未动：

- gfx950：`prefetch_k_iter` 下界 clamp + `total_iters >= pfk` 的 AITER_CHECK
- gfx942：cu_num 自动挑 split_k + `min_iters_per_split=2` clamp + P1 的偶数
  loops parity 要求
- gfx1250：空尾 split 的 clamp（保证 `n_active == split_k`，reduce 才敢把全部
  split_k 片都加进去）

产物 diff 里 `split_k--;` 在全部 19 个 split-K kid 里都还在。

## 新增：`include/opus_gemm_workspace_check.cuh`

一个共享 helper，不是三份 emitter 各生成一份。校验内容完全一样，而让它们漂移的
后果是"哪份弱哪个 arch 就 OOB 写"。

`validate_workspace<D_WS>(ws, required_elems, Y, kid_name)` 检查：

1. **`ws.dtype() == D_WS 对应的枚举`** —— 这条是重点。
2. `ws.is_gpu()` 且 `ws.device_id == Y.device_id`
3. `ws.is_contiguous()`
4. `required_elems <= ws.numel()`
5. base pointer 16B 对齐（split-K kid 最宽的 workspace 访问是 dwordx4）

`required_elems` 由调用方在 `size_t` 里算好传进来，不在 helper 里重算：padding
规则是 per-kid 的（`B_M` / `B_N` 来自 traits），复制一份这个算术正是会漂的东西。
各调用点每个因子都先 widen 到 `size_t` 再相乘，理由写在注释里。

### 为什么 dtype 那条是重点

两个 arch 的 workspace 是 fp32，gfx942 有一族是 bf16。所以"元素数对、元素宽度错"
是从 python 那边**够得着**的错误，而它产生的 buffer 恰好只有 kernel 要写的一半字节。
那不是精度损失，是越过分配末尾的 device 端越界写——#4246 就是这么把机器写死到
需要断电重启的。只查 numel 抓不到；查 dtype 抓得到，代价是一次整数比较。

### 新写的 C++ 类型 → `AiterDtype` 反向映射

`aiter_dispatch.h` 里有正向的 `cpp_type_from_aiter_dtype`（枚举 → 类型），
反向的没有。launcher 只知道 `Traits::D_WS` 这个**类型**，tensor 报的是运行时
**枚举**，所以补了一个 `opus_ws_check::aiter_dtype_of<T>`。

故意不做成通用 trait：只列了 `fp32_t` 和 `bf16_t` 两个可能成为 `D_WS` 的类型。
将来加一种新的 workspace 元素类型，会在这里编译失败，而不是悄悄取到隔壁的枚举值。

## gfx942 bf16ws redirect：坑的性质变了，注释重写

第 6b 记的那个坑，现在看得更清楚了。redirect 把**我们的 tensor** 交给 fp32ws
双胞胎，而双胞胎会拿**它自己的** `D_WS` 去 validate。所以 bf16ws kid 的 workspace
**dtype 是 shape 相关的**：

- shape 命中 `bf16ws_exact_reduce_shape` → 本 kid 自己 reduce → 要 bf16
- 没命中 → 落到双胞胎 → 要 fp32

不只是"按 fp32 定容"那么简单，`make_workspace_plan()` 必须求同一个 N 谓词，
按**真正会消费这个 buffer 的那个 launcher** 来定 dtype 和容量。注释已经改写在
codegen 生成点上（`gen_instances_gfx942.py` 的 `bf16ws_host_redirect` 分支）。

（redirect 在 `validate_workspace` 之前就 return 了，所以校验点上的
`Traits::D_WS` 永远是真正写这块 buffer 的那个 launcher 的类型。）

## 验证

### 产物 diff

三个 arch 重新生成。改动文件数 = split-K kid 数，一个不多：

| arch | 改动文件 | split-K kid |
|---|---|---|
| gfx942 | 7 | 7 |
| gfx950 | 6 | 6 |
| gfx1250 | 6 | 6 |

diff 行只有三类：删掉的 grow 块、新增的 validate 三行 + 注释、
`ws_handle*` → `ptr_ws_` 的实参重命名（gfx942 469 处、gfx950 24 处、
gfx1250 30 处——数量大是因为 gfx942 的 exact-N rowblock reduce 分支是
按 (N, split_k) 组合展开的）。

`ws_handle` / `opus_splitk_ws_get` / `hipMalloc` / `kGrowAlign` 在三个 arch
的产物里全部归零。

### 编译

**host TU 这一步第一次可编**（`opus_splitk_ws_get` 的 extern 就住在那儿）：

| TU | gfx942 | gfx950 | gfx1250 |
|---|---|---|---|
| `all_instances_host_<arch>.cu` | 0 error | 0 error | 0 error |
| device TU（17 / 13 / 7） | 0 error | 0 error | 0 error |

只剩既有的 `aiter_tensor.h` nodiscard warning，不在改动行上。

### parity

13/13。

## 还没接线

C++ 侧现在收 workspace tensor 并校验，但**还没有人传给它**——`SplitkWsRegistry`
和 `opus_gemm_workspace_init` 仍在 `opus_gemm.cu` 里（现已无人调用），
python 侧的 `_workspace.py` 还没写。这是第 12 步。

---

# 第 12 步 — 入口 / 绑定 / 框架 / 删 registry

至此 A1 收尾：C++ 侧最后一块自有内存删掉，python 侧接上线。

## 删掉的东西

| 文件 | 删了什么 |
|---|---|
| `csrc/opus_gemm/opus_gemm.cu` | `SplitkWsRegistry` + `opus_splitk_ws_get/put/reset` + `opus_gemm_workspace_init`（106 行），以及随之变成死引用的 include |
| `csrc/opus_gemm/include/opus_gemm.h` | `opus_gemm_workspace_init` 声明 |
| `csrc/include/rocm_ops.hpp` | `OPUS_GEMM_WORKSPACE_INIT_PYBIND` 宏整段 |
| `csrc/pybind/opus_gemm_pybind.cu` | 该宏的调用 |
| `aiter/tuned_gemm.py` | prewarm 机制整套（`_OPUS_WS_ARCHS` / `_opus_ws_warmed_sigs` / `_opus_needs_ws_prewarm` / `_opus_graph_capture_stream` / `_opus_prewarm_capture_workspace`，约 65 行）+ 调用点 |
| `aiter/ops/opus/gemm_op_a16w16.py` | `opus_gemm_workspace_init` 的 `@compile_ops` binding |

`opus_gemm_a16w16_tune` 的 C++ 签名末位加 `std::optional<aiter_tensor_t> workspace`，
声明 / 宏 / binding 三处同步。

## 新增：`aiter/ops/opus/_workspace.py`

python 侧唯一的定容点。

- `make_workspace_plan(arch, family, kid, batch, M, N, K, split_k)` → `WorkspacePlan | None`。
  转手 sidecar 的 `kid_workspace_plan()`，只做 dtype 名 → torch dtype 的映射，
  映射表 `_WS_TORCH_DTYPE` 是**闭集**（只有 `fp32_t` / `bf16_t`），和 C++ 侧
  `opus_ws_check::aiter_dtype_of` 一一对应：新增一种 workspace 元素类型会在这里
  KeyError，而不是悄悄分配错宽度。
- `allocate_workspace(plan, device)` → `torch.empty`（不是 `zeros`，理由写在
  docstring：reduce 读的每个元素都是同一次 launch 里主 kernel 写的，
  clamp 保证没有空 split，`zeros` 只会把 clamp 的回归掩盖成"答案还对"）。
- `workspace_for_call(kid, XQ, WQ, split_k)` —— 所有调用点走的那一个入口。
- `current_arch()` —— 走和包导入 guard 同一个 `_detect_arch`，
  `GPU_ARCHS` 覆盖时定容和编译取到同一个 arch。

`None` 是正常答案（非 split-K kid、以及走原子累加直接进 Y 的 wave_k_coop_accum
一族），必须原样往下传：C++ 那边"给不需要的 kid 传了 workspace"和"给需要的 kid
没传"是同等严格的两条断言。

### `split_k_upper_bound()`：只预测 gfx942

buffer 要在 launcher 跑之前定容，而 `splitK <= 0` 意思是"launcher 自己挑"。
查过三个 arch：只有 gfx942 的 launcher 真的自己挑
（`gen_instances_gfx942.py:502`，目标约 1 WG/CU，clamp 到 [1,16]）；
gfx950 / gfx1250 都是 `int split_k = (splitK <= 1) ? 1 : splitK;`。所以：

- `split_k > 0` → 原样返回（tuner 定的 / 显式覆盖）
- 非 gfx942 → 1
- gfx942 → `auto_split_k_gfx942(...)`，**CU 数取真机的**
  `multi_processor_count`，不用 heuristic 模块里 64 的兜底值——那个函数是随
  cu_num 单调增的，猜小了就是分配不够。

**没有**去移植各 launcher 后续的 clamp 循环。它们只会把 split_k 往小调，
所以这里给的是上界，最坏是多分配。移植过去反而要求 python 和 C++ 逐位一致，
一旦漂了就是 device 端越界写。这个单向余量是故意留的。

## 谁来分配：wrapper 默认分配

计划表里"两个 tuner 的调用点走公共 wrapper，自动分配"这两行，实际做法是让
`opus_gemm_a16w16_tune()` 在 `workspace is None` 时自己调 `workspace_for_call()`。
结果是 `csrc/opus_gemm/opus_gemm_tune.py` 和 `csrc/gemm_a16w16/gemm_a16w16_tune.py`
**一行没改**（前者只改了和第 6b 的 N=384 有关的那处），六个调用点也不用各写一遍
定容算术——写两遍的东西就是会漂的东西。

显式传 `workspace=` 的口子留着，但 docstring 写明它是给"故意递一个错 buffer
看断言炸不炸"的测试用的，不是给正常调用方的。

## `_opus_gemm_bf16_dispatch` 不再是 python 的路径

`gemm_op_a16w16.py` 里 CSV miss 的那一支，原来落到 C++ 的 `opus_gemm()`；
现在先过 python heuristic 拿到 kid，再走 tune 入口。原因写在注释里：
`opus_gemm()` 是 torch-free 的，分配不了，也从来不把选中的 kid 报回来——
这正是当初非要一个 C++ 侧 workspace registry 不可的那个缺口。
`opus_gemm()` 本身按计划留着不删（属于任务二），binding 也还在，
只是正常 python 路径不再经过它。

### 顺带补的 4 GiB 保护

C++ 的 `check_shape_4g` 跑在 heuristic **之前**（gfx950 `:186` / gfx1250 `:77`），
且只在 fallback 路径上跑、tuned CSV 命中时不跑。既然 heuristic 搬到了 python，
这条也得跟着搬，否则等于悄悄放开了 >4 GiB 的 heuristic dispatch。
`_check_shape_4g()` 照抄语义，gfx942 的豁免也照抄（那边本来就没这个 guard）。

### mono-kid 改道改成递归进 tune 入口

`mono_kid_shape_ok()` 不过的时候，原来重投 `_opus_gemm_bf16_dispatch`，
现在重投 `opus_gemm_a16w16_tune` + heuristic kid。终止性写在注释里：
heuristic 永远不会返回 mono-tile kid，所以重试不可能再进这一支。

## `opus_gemm_workspace_init` 变成 deprecated no-op

`aiter/ops/opus/__init__.py` 里定义成模块级空壳，放在 `if _arch_ok:` **之前**——
它什么都不 launch，没有"arch 不支持"这回事，两个分支上都该存在。
带 `DeprecationWarning`，docstring 说明老 registry 是什么、以及为什么现在没东西
需要预热。留着是因为 vLLM / sglang 那类 TBO setup 会在 capture 前调它，
升级不该直接炸。

`aiter/ops/deepgemm.py` 的 shim 转发改成关键字形式——wrapper 第 4 位现在是
`bias`，位置传参会走进那条"第 4 位是 int 就当 kernelId"的兼容修补路径。

## 为什么 prewarm 可以直接删而不是重做

`hipMalloc` 在 capture 期间非法，所以老方案必须在 capture 前把每个 stream 的
workspace 先长出来。torch 的 caching allocator 在 capture 期间返回 replay-stable
的地址，captured graph 烤进去的指针在 replay 时仍然有效。这一条事实就是整套
prewarm 能删掉的全部理由，删除点上留了 8 行注释说明。

顺带：buffer 现在进 torch 的内存池，而不是待在一块别的 allocator 看不见也回收
不了的私有分配里。

## README

`## Splitk workspace and TBO` 整节重写成 direct-pointer ABI：launcher 签名、
**Sizing it**（指向 `_workspace.py::workspace_for_call()` → `kid_workspace_plan()`，
写清 padding 规则和 gfx942 bf16ws 那个 shape 相关的 dtype）、
**Validation**（六条检查，以及为什么 dtype 那条靠 #4246 撑着、为什么不能用
uint8 扁平 buffer）、**TBO and HIP graphs: nothing to do**。

## 验证

### 产物

三个 arch 重新生成，和第 11 步的 baseline **逐字节相同**。第 12 步是纯接线。

### 编译

46 个 TU 语法编译 0 error（gfx950 16 / gfx942 20 / gfx1250 10，
gfx1250 用 `--offload-arch=gfx950` 代编——本机 hipcc 7.0 打不了 gfx1250 target）。

### parity

`op_tests/test_opus_heuristic_parity.py` 13/13。

### 端到端（本机 gfx950，256 CU）

| 用例 | 结果 |
|---|---|
| `test_opus_a16w16_gemm.py` b=8 m=256 n=512 k=256 | err=0 |
| b=8 m=1 n=8192 k=4096（瘦长 M） | err=0 |
| b=4 m=128 n=2048 k=5120 | err=0 |
| b=8 m=4096 n=4096 k=4096 | err=0 |
| b=8 m=512 n=4096 k=4096 fp32 out | err=0 |
| b=8 m=33 n=1024 k=2048（非整齐 M） | err=0 |
| bias / fp32 out / bias+fp32 | relerr 2–3e-3 |
| 17 个已编译 split-K kid × splitK ∈ {1,4,8} | 51 ok / 0 fail |

### HIP graph 冷 capture

**不预热、不调 `opus_gemm_workspace_init`**，split-K shape 的第一次触碰就在
capture 里面——正是老方案会 abort 的那个用例——capture 成功，replay 正确
（relerr 3.13e-3），改输入后 replay 仍正确。这是这次迁移的核心断言，单独验了。

### 七条断言全部命中各自的负面用例

numel 对 / 元素宽度错（#4246 的签名）、太小、uint8 扁平 buffer、cpu tensor、
非 contiguous、split-K kid 传 None、非 split-K kid 传了 workspace。

### 两个 tuner + shim

`opus_gemm_tune.run_opus_gemm` / `run_opus_gemm_bench`（eager + graph-captured）、
`gemm_a16w16_tune.run_opus_gemm_bf16`（relerr 2.99e-3）、
deepgemm shim（relerr 3.19e-3 + DeprecationWarning）全部正常。

## 三个失败，都是既有问题（stash 后在 baseline 上复现过）

1. `import aiter` 撞 `triton>=3.6.0, found 3.4.0` 的 gate（`AITER_USE_SYSTEM_TRITON=1` 绕过）
2. `aiter.ops.flydsl` 缺失，`tgemm.mm` 起不来
3. `opus_gemm_tune.py` 命令行 tune 0 shapes（改调 `run_opus_gemm` 直连）

## A1 之外还欠的两笔（来自 A0）

- N=384 的行为变化要单独一个 commit 落，commit message 里写明两个 gfx942
  bf16ws kid 在 N=384 重新进入 tuning space。
- 两个 unreachable kid（gfx950 1208、gfx942 10003）不修，在 parity 测试里
  按精确集合断言住。

---

# 第 13 步：坑 4（splitK 上界）+ gfx1250 batch>1 拒绝 + 验收补测

前 12 步把 workspace 迁到了 torch，但计划里的**坑 4** 一直没落地，验收清单
里的**双 stream** 和 **gfx1250 batch>1 明确失败**两项也只有口头结论、没有
测试兜住。这一步补齐这三样，不动已有行为。

## 改的文件

| 文件 | 性质 |
|---|---|
| `csrc/opus_gemm/opus_gemm_common.py` | 改（`kid_workspace_plan` 两处 raise + docstring） |
| `op_tests/test_opus_splitk_workspace.py` | 新建（417 行，19 个用例） |

## 为什么两个 guard 都放在 `kid_workspace_plan`

`kid_workspace_plan()` 是全仓**唯一**算 workspace 尺寸的地方（grep 三处命中，
其中一处是 `aiter/ops/opus/_workspace.py:160` 这个唯一调用点）。放这里意味着
不管从哪个入口进来——两个 tuner、shape API、deepgemm shim——都绕不过去；
放 `_workspace.py` 或调用点就得复制 N 份，而复制的校验必然漂移。

### 坑 4：`split_k > max_split_k` 报错，不夹

```python
if split_k > inst.max_split_k:
    raise ValueError(...)
```

**为什么不能静默夹。** launcher 自己就会向下夹。所以这里夹一下再放行的话，
调用方拿到的是一次**正常跑完**的 GEMM，只不过 buffer 是按夹完的片数分配的，
比调用方以为自己要的小——而且和一次正确分配在外部完全无法区分，直到有人
读过界。相反，`splitK=100000` 本来就是调用方 bug（tuner 枚举到 16 封顶，
每个 launcher 也硬夹 16），把一次几十 GiB 的 `torch.empty` 换成一行错误信息
是划算的。

顺带补了 `split_k < 1` 的下界：0 在上游表示"让 launcher 决定"，必须在
`split_k_upper_bound()` 里解析成具体上界后才能进来，它不是一个合法 extent。

### 坑 2 的另一半：`ws_includes_batch=False` 的 kid 拒绝 `batch != 1`

```python
if inst.ws_includes_batch:
    shape = (split_k, batch, padded_M, padded_N)
else:
    if batch != 1:
        raise ValueError(...)
    shape = (split_k, padded_M, padded_N)
```

gfx1250 的 launcher 里写死 `kargs.batch = 1`（`gen_instances_gfx1250.py:337`），
C++ 侧只 check `batch >= 1`（`:282`），没人拦 batch>1。所以 batch=4 的调用会
只算 batch 0，Y 的其余三个 batch 保持分配时的内容——**答案错，但不报错**，
正是这个任务要消灭的那类失败。sizing 是最后一个同时看得到 kid 和 batch 的
地方，所以拦在这里，而不是指望调用方自觉。

按计划要求，判据是 `ws_includes_batch` 字段，**不是** `if arch == "gfx1250"`。

## 新测试 `op_tests/test_opus_splitk_workspace.py`

19 个用例。sizing 那批**不需要 GPU、不需要编译产物**，按路径 load
`opus_gemm_common.py`（和 parity 测试同一手法），因此一台 gfx950 机器也能测
gfx1250 的规则——否则 gfx1250 的 batch 规则在本机永远是盲区。

| 用例 | 拦的是什么 |
|---|---|
| `test_split_k_within_max_is_accepted[3 arch]` | 上界的下半边：全 refuse 也能过"拒绝 100000"，所以 1/2/max 必须通过且 extent 随 split_k 单调增 |
| `test_split_k_over_max_is_refused_not_clamped[3 arch]` | 每个 splitk kid × {max+1, 64, 100000} 都 raise，且错误信息里带 `max_split_k` |
| `test_split_k_below_one_is_refused[3 arch]` | 0 / -1 |
| `test_max_split_k_matches_the_tuner_ceiling` | 552 个 kid 的 `max_split_k` 集合 == {16}，和 tuner 枚举上限、gfx942 auto-pick 里的字面量 16 对齐 |
| `test_non_splitk_kid_is_unaffected_by_the_bound` | 非 split-K kid 传 splitK=100000 仍返回 None——上界故意放在 split-K 分支内 |
| `test_batchless_workspace_is_exactly_the_gfx1250_kids` | 精确集合断言：batchless kid 集合 == gfx1250 kid 集合（496 个）。新 arch 冒出来会在这里红，而不是变成 allocator 里的一个 `if arch ==` |
| `test_gfx1250_rejects_batch_gt_1` | 496 个 kid × batch ∈ {2,4,64} 全 raise；batch=1 返回 3D shape |
| `test_batched_arches_accept_batch_gt_1[gfx950/942]` | 反面：gfx950/942 的 batch 维随 batch 变。没这条的话"到处拒绝 batch"也能过上一条 |
| `test_two_streams_get_independent_workspaces` | 两个同时存活的 workspace `data_ptr()` 不同。这条挡的是未来给 workspace 加 cache——那正是被删掉的 per-stream registry 要绕的别名危险 |
| `test_concurrent_splitk_on_two_streams_is_correct` | **验收清单的"双 stream"**：两个 CPU 线程各驱一个 `torch.cuda.Stream()`，各跑 32 次 split-K，各自对自己的 `torch.bmm` 参考核对 |
| `test_wrapper_refuses_absurd_split_k_before_allocating` | 上界从公开入口 `opus_gemm_a16w16_tune` 可达，且 `memory_reserved()` 不变——证明是"拦下"不是"分配失败" |
| `test_splitk_matches_torch_across_the_legal_split_k_range` | split_k 从 1 扫到 16，每个值一个不同大小的 buffer，两端都得对 |

### 双 stream 那条为什么断言相对误差

手工跑的时候 max abs 误差是 0.5 / 1.0，看着吓人。单 stream 同 shape 基线一
对比也是 max abs 0.5、rel 0.0075、mean rel 2.3e-7——K=4096 的 bf16 累加本来
就这个量级，逐元素容差在这里没有意义。真正的 stream 间串扰会让相对误差差
几个数量级，所以断言 `sum|diff| / sum|ref| < 1e-2`；两个线程用不同 seed，
结果被调包不会蒙混过去。

## 验证

```
op_tests/test_opus_splitk_workspace.py    19 passed
op_tests/test_opus_heuristic_parity.py    13 passed
```

端到端那批确实跑到了真 kernel（不是空过）：本机 gfx950 选中 kid 200
`opus_gemm_flatmm_splitk_256x64x64x64_2x1_16x16x32_0x0x0_wgpcu2`，workspace
`[4, 2, 256, 512] float32`，rel 1.41e-3。

gfx1250 的 batch 规则只在 host 侧验（本机是 gfx950，且本机 hipcc 打不出
gfx1250 target），走 `kid_workspace_plan` 直连——这是这条规则在任何一台单
arch 机器上唯一可测的形式。

## 仍然欠的

- **本次改动尚未 commit**（用户指示先不管 commit）。
- N=384 行为变化的独立 commit（A0 遗留，见上一节）。

# 第 14 步 — 坑 1b 收敛到计划（删掉 C++ redirect）+ 验收清单补测

第 13 步末尾"仍然欠的"里那两笔中的一笔了结了：坑 1b 的实现一直和计划有出入
（计划要求删掉 C++ 侧的 bf16ws redirect，实际保留着），本步把它对齐；同时把
验收清单里那 7 行"只手工验过一次、没有回归测试"的补上。

先做坑 1b 再补测，顺序是故意的：测试要断言的是最终行为，redirect 还在的时候
写出来的 bf16ws 测试会把中间态固化成契约。

## 为什么现在删而不是继续保留

A0 当初留着 C++ redirect 的理由是"两份实现并存，好做 port 前后的 parity
对比"。A1 做完之后这个理由过期了——现在 workspace 是 python 先分配、再传进
launcher 的，launcher 在拿到 buffer *之后*把 kid 换成 twin，等于拿一个按
2 字节元素类型算出来的 buffer 去喂一个写 4 字节的 kernel。这就是 #4246
（`ea093a77`）那次的形状：numel 对、宽度错、2 倍欠分配、越界写。

redirect 本身没错，错的是它发生的时机。它现在必须发生在**分配之前**，也就是
kid 选择那一步，所以它整体上移到了 `resolve_bf16ws_kid()`。

## 改的文件

| 文件 | 改动 |
|---|---|
| `csrc/opus_gemm/codegen/gen_instances_gfx942.py` | 删掉 `bf16ws_fallback_decl`（fp32ws twin 的前向声明块）及其模板槽；原先 `if is_quad_mfma32_splitk` 的两分支合并成一个无条件 `AITER_CHECK` |
| `csrc/opus_gemm/opus_gemm_common.py` | `kid_workspace_plan()` 对越界 N 的 gfx942 bf16ws kid 从"改 dtype"改成 `raise ValueError`；docstring / `_GFX942_BF16WS_REDIRECT` 注释块 / `resolve_gfx942_bf16ws` docstring 重写 |
| `aiter/ops/opus/gemm_op_a16w16.py` | 注释（~611 行）：说明这里是 substitution 唯一发生的地方，以及为什么 launcher 不再自己做 |

launcher 里现在是：

```cpp
const bool bf16ws_exact_reduce_shape = <N 的 exact-N 判据>;
AITER_CHECK(bf16ws_exact_reduce_shape,
    "...: bf16 workspace reduce is only generated for exact-N rowblock "
    "shapes, got N=", N,
    ". This kid is not silently redirected to its fp32-workspace twin: "
    "the workspace was already sized for a 2-byte element type and the "
    "twin writes 4. Select the kid through "
    "aiter.ops.opus.common.resolve_bf16ws_kid(), which performs the "
    "substitution before the buffer is allocated.");
```

sizing 侧同样从"悄悄换 dtype"变成拒绝：

```python
if arch.lower() == "gfx942" and ws_dtype == "bf16_t":
    if not gfx942_bf16ws_exact_n_ok(N):
        raise ValueError(... "Resolve the kid through resolve_gfx942_bf16ws() "
                             "before sizing -- {hint}.")
```

`hint` 区分 10210/10213（有 twin）和 10216（无 twin，严格拒绝），因为这两种
情况调用方要做的事不一样。

## 产物是逐字节核对过的

删 codegen 里的代码最容易顺手删多。做法是把改动前的 `gen_instances_gfx942.py`
重建到 `/tmp/pre_edit_gfx942.py`，换进去生成一遍，再换回来生成一遍，diff 两组
产物：

- gfx950 / gfx1250 产物 **byte-identical**（这两个 arch 本来就不该被碰）
- gfx942 的差异只落在 7 个 split-K 文件里，内容只有 AITER_CHECK 的替换
  （外加非 bf16ws kid 上多出的一个空行）
- 融合 host TU 对 gfx942 语法编译 exit 0

## 补测：验收清单那 7 行

测试文件从 19 个用例扩到 44 个。模块 docstring 里列了 7 条关切，和验收清单
一一对应。

**bf16ws 一节**（模块级 `_BF16WS_KIDS = (10210, 10213, 10216)`、
`_BF16WS_TWIN = {10210: 10200, 10213: 10203}`）：

| 用例 | 拦的是什么 |
|---|---|
| `test_bf16ws_kid_set_has_not_moved` | 三个 kid 的身份三元组还在原位；新增 bf16ws kid 会在这里红 |
| `test_exact_n_set_includes_384` | A0 那笔行为变化（384 重新进 tuning space）不会被悄悄回退 |
| `test_bf16ws_legal_n_sizes_as_bf16[64/384/2048]` | 合法 N 仍按 bf16 算——只会拒绝也能过"拒绝非法 N" |
| `test_bf16ws_illegal_n_is_refused_by_sizing[600/777/1536]` | sizing 抛 ValueError，且信息里带 bf16 和实际 N |
| `test_bf16ws_illegal_n_substitutes_at_kid_selection[600/777/1536]` | 替换上移到 kid 选择：10210→10200、10213→10203，且 tuned splitK 被丢弃（换了 kid，原来的 splitK 是对另一个 reduce 调的）；10216 rejected 且 kid 不变 |
| `test_no_path_feeds_a_bf16_buffer_to_an_fp32_kernel` | 穷举 3 kid × N∈[64,2112] step 64：sizing 报的 dtype 永远属于 resolve **之后**的 kid。redirect 被重新引入到这两步之间的任何位置，都在这里红 |
| `test_tuner_and_sizing_agree_on_which_n_are_legal` | tuner 的准入判据和 sizing 判据同源（并断言两个函数对象不同一，否则比较是空的） |

**C++ 契约一节**（8 条负例，逐条在真设备上探过再写断言，不是照着 header 猜的）：
`test_contract_good_workspace_is_accepted` / `rejects_wrong_dtype_same_numel` /
`rejects_one_element_short` / `rejects_flat_uint8_buffer` / `rejects_cpu_tensor` /
`rejects_noncontiguous` / `rejects_unaligned_base_pointer` /
`splitk_kid_rejects_missing_workspace` / `non_splitk_kid_rejects_supplied_workspace`。

`rejects_wrong_dtype_same_numel` 是这批里最重要的一条：numel 对、dtype 错，
正是 #4246 的形状。

**其余三条**：
- `test_cold_hip_graph_capture_needs_no_prewarm` — 冷进程直接 capture，不调
  任何 prewarm。这是删 `opus_gemm_workspace_init` 的唯一依据（caching
  allocator 在 capture 期间给的地址是 replay-stable 的）的可执行形式
- `test_bias_dtype_rules_are_intact` — bias/out dtype 组合矩阵
- `test_workspace_does_not_accumulate_across_shapes` — shape 扫描后稳态不涨

## 三条测试自身的坑（第一轮 5 red，都是测试写错不是产品错）

- `test_bf16ws_illegal_n_substitutes_at_kid_selection` ×3：拿 `shape` 和
  `kid_workspace_plan()` 的整个返回值 `(shape, dtype)` 比，忘了解包
- `test_tuner_and_sizing_agree_on_which_n_are_legal`：`opus_gemm_tune.py` 里是
  平铺的 `from opus_gemm_common import ...`（tuner 以 `csrc/opus_gemm` 为 cwd
  跑时的形态），按路径 load 时 `ModuleNotFoundError`。补 `sys.path` 而不是
  往 `sys.modules` 里塞 `common` 已经持有的那份——塞进去两个判据就是同一个
  函数对象，比较变成恒真
- `test_workspace_does_not_accumulate_across_shapes`：断言写成了精确相等，
  实测内存**减少**了 7 MB（进程里别的缓存张量释放）就红了。改成非增长上界：
  留住 workspace 是逐 pass 累加的，任何上界都抓得到；减少从来不是这条测试要
  找的失败

## 验证

```
op_tests/test_opus_splitk_workspace.py    44 passed
op_tests/test_opus_heuristic_parity.py    13 passed
```

（44 那批换 `-p no:randomly` 固定顺序又跑了一遍，排除用例间顺序耦合。）

## 仍然欠的

- **本次改动尚未 commit**（用户指示先不管 commit）。
- N=384 行为变化仍需独立 commit，message 里注明两个 gfx942 bf16ws kid 在
  N=384 重新进入 tuning space。
- `aiter/ops/opus/_workspace.py:119` 的 `sys.path.insert` 相对计划的 grep 块
  没有清零（理由已记在第 12 步）。
- 本文"累计改动文件"表里的行数已经过时（`opus_gemm_common.py` 记的是 +253，
  实际更多）。

# 第 15 步 — review 两条意见：workspace plan 按最终 kid + 砍掉 `_workspace.py`

来自 0811 的两条 review：

1. 「这个 workspace plan 有考虑 heuristic 的处理么？」（红框圈在"按最终 kid
   生成 WorkspacePlan"那行）
2. 「不要增加专门的 workspace python 文件。这个太重了。还有这里的 dtype，
   需要都跟 942 一样，由 kid 决定 fp32 或者 bf16。」「层级太多了。我理解应该
   只需要一个 init 包含 shape 处理就可以了。」

核下来第 1 条牵出一个**真 bug**，第 2 条照做。

## 意见 1：顺序是对的，但解析在运行时根本没生效

调用链本身没问题：三条路径（explicit kid / tuned CSV / heuristic）都是先
`resolve_bf16ws_kid()` 定最终 kid，再把它传进 `opus_gemm_a16w16_tune`，那里才
分配。**先定 kid 再算 plan** 这个次序是成立的。

问题在解析本身。实测：

```
common._bf16ws_resolver()            -> None
common.resolve_bf16ws_kid(10210,600,8) -> (10210, 8, False)   # 应为 (10200, None, False)
```

`common.py:219` 的 `_bf16ws_resolver()` 直接 `from opus_gemm_common import ...`，
但 `csrc/opus_gemm` 在运行时**不在 sys.path 上**（`aiter.jit.core` 只插
AITER_META_DIR，即仓库根）。except 分支静默 `return None`，`resolve_bf16ws_kid`
退化成恒等映射。

那个 except 的注释写着"A0 阶段 C++ 侧 redirect 还在，降级到今天的行为"——
**这个前提在第 14 步被我自己删掉了**。所以现在 gfx942 + N=600 + kid 10210
会：kid 不变 → 按 bf16 分配 → 撞 launcher 的 AITER_CHECK。第 14 步删 redirect
把一个"静默降级"变成了"必挂"，而降级路径没跟着改。

**为什么 44 个测试没抓到**：它们全部按路径 load sidecar，直接测
`opus_gemm_common.resolve_gfx942_bf16ws`。那是测**规则**的正确方式（任何单
arch 机器上都能测三个 arch），但它绕过了 `aiter/ops/opus/common.py` 这层包装，
于是"规则对不对"绿着，"规则有没有被走到"没人看。

修法：sidecar 的导入收敛成 `common.py::sidecar()` 一个入口，自带 sys.path 处理，
**失败即 raise**。没有 None 可返回了——C++ 侧 redirect 没了，sizing 也没有 C++
allocator 可退，静默猜一个尺寸就是猜一个 kernel 即将写多少显存。

## 意见 2：删掉 `_workspace.py`，收敛成一个函数

原来 285 行、公开 5 个符号（`WorkspacePlan` 类 + `make_workspace_plan` +
`allocate_workspace` + `split_k_upper_bound` + `workspace_for_call`），但外部
实际只用到 2 个：`workspace_for_call`（1 处）和 `current_arch`（1 处）。其余
三个加那个类都是自己内部串起来的层级。

现在：整个文件删除，落到 `aiter/ops/opus/common.py`：

| 原 | 现 |
|---|---|
| `WorkspacePlan` 类（shape/dtype/kid/arch + numel + repr） | 删。plan 是 sidecar 返回的 `(shape, dtype)` 二元组，用完即弃，不需要一个类 |
| `make_workspace_plan()` | 并进 `workspace_for_call()` |
| `allocate_workspace()` | 并进 `workspace_for_call()`（就是一行 `torch.empty`） |
| `split_k_upper_bound()` | `_split_k_upper_bound()`，私有 |
| `workspace_for_call()` | 保留，是唯一公开入口：sidecar 问 shape+dtype → `torch.empty` |
| `_sidecar()` | 并进 `common.py::sidecar()`，和 bf16ws 解析共用同一个入口（见意见 1） |
| `current_arch()` | 挪到 `common.py` |

`aiter/ops/opus/_workspace.py` 已删除。

### dtype 早就是"由 kid 决定"

`splitk_workspace_dtype` 是 kid 的字段，`workspace_for_call` 直接读它，没有
`if arch ==` 分支。实测三 arch 的 kid 数分布，正好对上 review 那张表：

| arch | ws dtype 分布 | batch |
|---|---|---|
| gfx942 | fp32 × 5，bf16 × 3 | 支持 |
| gfx950 | fp32 × 48 | 支持 |
| gfx1250 | fp32 × 496 | 只允许 batch=1（`ws_includes_batch=False`） |

唯一带 `arch == "gfx942"` 字样的是 `kid_workspace_plan` 里 exact-N 那个拒绝
分支，那管的是 **reduce kernel 的生成范围**，不是 dtype 规则——bf16ws kid 在
合法 N 上照样按 bf16 分配。

## 新增 3 条测试（补的正是上面那个盲区）

| 用例 | 拦的是什么 |
|---|---|
| `test_public_resolver_actually_resolves` | 从 `aiter.ops.opus.common` 这个**运行时真正走的入口**断言精确三元组。同时断言合法 N（512）**不**替换，否则"永远 redirect"也能过 |
| `test_public_resolver_agrees_with_the_sidecar` | 3 kid × N∈[64,2112]，wrapper 和 sidecar 逐点相等。写成等式而不是第二份期望值，这样它不会和规则各自漂移 |
| `test_missing_sidecar_raises_instead_of_degrading` | `sidecar()` 里不许再出现 `return None`。静默降级分支重新长出来会在这里红 |

**做了变异验证**：把 `resolve_bf16ws_kid` 改回旧的恒等映射行为，重跑——

```
FAILED test_public_resolver_actually_resolves
FAILED test_public_resolver_agrees_with_the_sidecar
2 failed, 45 passed
```

也就是说这两条确实抓得住已经发生过的这个 bug，不是摆设。

## 验证

```
op_tests/test_opus_splitk_workspace.py   47 passed（19 -> 44 -> 47）
op_tests/test_opus_heuristic_parity.py   13 passed
op_tests/test_opus_a16w16_gemm.py        4 shape 全 passed，err=0
  b=8 m=256 n=512 k=4096 bf16 | 56.32 TFLOPs
  b=2 m=128 n=600 k=2048 bf16 | 23.01 TFLOPs   <- 走 bf16ws 转跳的那条
  b=1 m=1024 n=2048 k=4096 bf16 | 116.76 TFLOPs
  b=8 m=256 n=512 k=4096 fp32 | 56.31 TFLOPs
```

注：`test_opus_a16w16_gemm.py` 是脚本不是 pytest 用例（`test_` 前缀会被 pytest
误收集成 fixture 缺失），按它自己的 argparse 用法跑；且本机需要
`PYTHONPATH=<repo>`，否则会 import 到 `/sgl-workspace/aiter` 那份装好的包。

## 仍然欠的

- **本次改动尚未 commit**（用户指示先不管 commit）。
- N=384 行为变化仍需独立 commit。
- **三 arch 各跑一遍**：本机只有 gfx950。gfx942 / gfx1250 的端到端从未在真
  设备上跑过——gfx1250 的 batch 规则、gfx942 的 bf16 ws dtype 目前都只在 host
  侧验过。这条在本机关不掉。
- `gen_instances_gfx942.py` 里 `bf16ws_host_redirect` / `bf16ws_exact_reduce_shape`
  两个变量名还在（语义已换成 AITER_CHECK），计划的 grep 要求归零。纯改名。
