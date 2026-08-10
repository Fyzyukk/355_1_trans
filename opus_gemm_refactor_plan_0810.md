# Opus GEMM 两大任务 — 修改架构方案（基于 0810 树）

**任务一**：splitK workspace 切换到 torch 分配方式
**任务二**：重构 opus gemm 的各种入口

> 目标仓库：`/root/workspace/0810/aiter`，`main` @ `35696023`
> 参考实现（两个 PR 分别对应两个任务，不是同一件事的两种做法）：
> - **任务一** → ROCm/aiter PR **#4246** `gfx1250 opus gemm splitk fuse`（open）中的 commit `4ac1fa8`：gfx1250 的 workspace 从 hipMalloc registry 切到 python `torch.empty`
> - **任务二** → ROCm/aiter PR **#4320** `Add opus fp8 mxscale BMM kernels for gfx950`（yzhou103，open，即「zoe 那个」）：已经把分层做对了的样板，详见 §2.3
> 任务来源：Lyu, Binglue 的两段需求（`11.png` / `download.png`）
>
> **本文取代 `opus_gemm_refactor_plan.md`**。那份基于另一个更旧的 checkout，与本树有三处实质差异（见 §0.1），照那份做会踩空。

---

## 0. 需求拆解

### 任务一 — splitK workspace 切换到 torch

> 「还有 splitk 的 workspace 开辟需要切换到 torch 的方式」「按这个方式（PR #4246），或者 zoe 那个也可」

**独立可交付**。只改 workspace 的内存所有权：C++ 内部 `hipMalloc` registry → python `torch.empty` 传指针进来。不触碰 dispatch / 入口逻辑。

> **两个参考 PR 的分工（一处常见误读）**
>
> #4246 捆了**两件独立的事**，只有第一件属于任务一：
>
> | | 内容 | 与任务一 |
> |---|---|---|
> | (a) commit `4ac1fa8` | gfx1250 的 `SplitkWsRegistry`/hipMalloc → python `torch.empty`；kargs `ws_handle*` → `void* ptr_ws` | **这才是要抄的**。注意上游只切了 gfx1250，gfx942/gfx950 仍留在 registry 上——**任务一要做的是把它推广到三个 arch** |
> | (b) 新 pipeline `..._tdm_splitk_fuse_gfx1250.cuh` | cluster barrier 在单 kernel 内做 split-K 归约，省掉独立 reduce kernel 的 launch + 一趟显存往返 | 无关。kernel 层改动 |
>
> 两个澄清：
> - **TDM 本来就在树里**。`opus_gemm_pipeline_a16w16_{cluster,clusterlaunch}_tdm_splitk_ws_gfx1250.cuh` 从 #3517（`90ee2737`）就有了。#4246 加的是 `_fuse` 变体，不是 TDM 本身。
> - **fuse 并没有消掉 workspace**。非最后一个 split 的 workgroup 仍要把 fp32 partial cast 成 workspace dtype 存进 GL2（默认 bf16，可选 fp32）。省的是 reduce kernel 的 launch，不是 buffer。这反而**加重**了 §3.1 坑 1：#4246 的 commit `86d1eba` 就是因为「写 fp32 的 kernel 写进按 bf16 算大小的 buffer」把机器挂了，才改成按 kid 自己的 ws dtype 定尺寸。
>
> **zoe 的 #4320 不是 workspace 参考——它整个没有 workspace。** `opus_bmm.cu` 的入口是 `opus_bmm_a8w8_mxscale(O, wo_a, Y, x_scale, w_scale, int splitK, int kernelId)`，无 workspace 参数、无 hipMalloc、无 registry，tuned CSV 里 78 行 `splitK` 全是 1。它的价值在任务二，见 §2.3。

### 任务二 — 重构 opus gemm 的各种入口

> 「重构一下现在 opus gemm 的各种入口之类的」
> 「需要把之前 gfx950 的部分也切到这上面来」
> 「tune lookup 简化一下……python 有一套，c++ 还有一套。现在只留 python 这套。主要是 heuristic 的部分」
> 「c++ 只维持目前的 kernel 相关部分，以及根据 kid dispatch 的功能。其余所有的工作都交给 aiter 这里的 torch 做」
> 「各种 dtype 和 layout 也需要在 opus_gemm.cu 那个文件以及 python interface 整理一下」

| # | 子需求 | 归类 |
|---|---|---|
| R1 | 重构各种入口 | 入口收敛 |
| R2 | gfx950 部分切到新机制上 | 跨 arch 统一 |
| R3 | tune lookup 只留 python 一套（主要是 heuristic） | 职责下沉 |
| R4 | `opus_gemm.cu` + python interface 的 dtype / layout 整理 | 接口清理 |

「c++ 只维持 kernel + kid dispatch」是四项共同的总纲。

### Bonus — opus 新 kernel 接入 skill

> 「搞个 skill 来作为 opus 新 kernel 接入的方式，减少 agent 吐冗余代码和 comment，维持代码结构扁平简洁」

### 0.1 本树与旧分析的三处关键差异

| 项 | 旧 checkout | **本树（0810）** | 影响 |
|---|---|---|---|
| arch 数 | gfx950 + gfx942 | **+ gfx1250**（`90ee2737` 已合入） | 所有改动 ×3 arch；gfx1250 的 workspace 公式还和另两个**不一样**（§1.3） |
| C++ 顶层入口 | 5 个（含 3 个 a8w8） | **3 个**（`opus_gemm` / `opus_gemm_a16w16_tune` / `opus_gemm_workspace_init`） | a8w8 已无独立顶层入口、python 侧也没有 `gemm_op_a8w8_*.py`。R1 的收敛目标从 5→3 变成 **3→2** |
| workspace handle | 单层（host handle） | **两层**（host pinned handle + hipMalloc device mirror + `sync_to_device`） | 任务一要删的东西比预想多一个函数、一份 device 副本 |

另外 gfx1250 的 heuristic 已经是「`(M,N,K,has_bias) → int kid`」的干净形态（`opus_gemm_heuristic_dispatch_gfx1250.cuh:38`），和 gfx950 一致；**只有 gfx942 还在直接返回 launcher 符号名**。

### 0.2 两个任务的关系

任务一是任务二总纲在「内存所有权」这一维上的实例，但**技术上独立**：任务一只动 workspace ABI 与分配点，任务二只动 dispatch 与入口签名。

**建议先做任务一**：规模小、边界清晰、有 PR #4246 参照，且它删掉的 `opus_gemm.cu:325-429` 正好是任务二要瘦身的同一文件里最容易剥离的一段。

---

## 1. 现状盘点

### 1.1 入口与 dispatch 路径

**C++**（`csrc/opus_gemm/include/opus_gemm.h`，实现在 `opus_gemm.cu`，431 行）：

| 入口 | 干什么 |
|---|---|
| `opus_gemm`（7 参） | 万能入口：按 `XQ.dtype` 分 fp8 / bf16；fp8 支硬编码单个 launcher，bf16 支做 **C++ 内部 (M,N,K) 查表 + heuristic** |
| `opus_gemm_a16w16_tune`（+kid +splitK） | 唯一「纯 kid dispatch」的入口，最健康的一个 |
| `opus_gemm_workspace_init`（无参） | 给 splitK 注册 per-stream `hipMalloc` workspace |

**Python**（`aiter/ops/opus/`）：`gemm_a16w16_opus`、`opus_gemm_a16w16_tune`、`opus_gemm_workspace_init`；外加 `aiter/tuned_gemm.py` 的 `solMap["opus"]` 调用点。

**同一个 bf16 GEMM 有三条到达 kernel 的路径**：

```
路径 A  gemm_a16w16_opus → python lookup_tuned(CSV) 命中 → opus_gemm_a16w16_tune(kid)
路径 B  gemm_a16w16_opus → CSV miss → _opus_gemm_bf16_dispatch → C++ opus_gemm
                                        → opus_gemm_lookup.h 查表（codegen 时从同一批 CSV 烘焙）
                                        → miss → opus_a16w16_heuristic_kid_gfx950/942/1250()
路径 C  aiter/tuned_gemm.py solMap["opus"] → opus_gemm_a16w16_tune(kid)
```

路径 A 与路径 B 的第一段**读的是同一批 CSV**（`aiter/configs/bf16_tuned_gemm.csv` + `model_configs/*_bf16_tuned_gemm.csv`，均按 `libtype=='opus'` 过滤），只是一个运行时用 pandas 读、一个 codegen 时烘焙成头文件。**这就是「python 一套、C++ 还有一套」的具体所指。**

> **⚠️ 实现者必读：codegen 产出的是两张表，只能删一张。**
>
> | 头文件 | 键 | 生成于 | 处置 |
> |---|---|---|---|
> | `opus_gemm_lookup.h` | **(M, N, K)** → launcher | `gen_instances.py:430 gen_lookup_dict()` | **删**（这就是 R3 要消掉的那套） |
> | `opus_gemm_a16w16_tune_lookup.h` | **kid** → launcher | `gen_instances.py:537 gen_a16w16_tune_lookup()` | **保留**（这就是「根据 kid dispatch」本身） |
>
> 两个文件名只差 `_a16w16_tune`，`gen_instances.py` 里两个生成函数也紧挨着（430 / 537）。删错就把 kid dispatch 一起端了。

### 1.2 C++ 侧越界承担的 framework 职责

| 越界项 | 位置 | 规模 |
|---|---|---|
| (M,N,K) 调优表查找 | `gen_instances.py:430 gen_lookup_dict()` → `opus_gemm_lookup.h` → 3 个 arch 头各两处 `lower_bound` | ~108 行 codegen |
| heuristic 选核 | gfx950 `opus_gemm_heuristic_dispatch_gfx950.cuh`（127 行）<br>gfx942 同名（157 行，**返回 launcher 符号名**）<br>gfx1250 同名（48 行，返回 kid，形态最好） | 332 行 |
| splitK workspace 生命周期 | `opus_gemm.cu:325-429`（registry + 3 个导出函数 + host/device 双 handle）<br>+ 3 份 codegen 内联的 grow 块（gfx950:1327-1357 / gfx942:495-528 / gfx1250:289-321）<br>+ `aiter/tuned_gemm.py:51-125` 的 75 行 prewarm 补丁 | **~275 行** |
| kid 语义分类 | `opus_gemm.cu:186-265`（7 个常量 + 6 个谓词）+ `opus_gemm_arch_gfx950.cuh:100-112` **重复** + `gemm_op_a16w16.py:289-290` **再重复** + `opus_gemm_common.py:1033-1161` **权威版本在这** | 4 处副本 |
| 4 GiB buffer-resource 守卫 | `opus_gemm_arch_gfx950.cuh:196` / `:233`（两个特化各一份）+ `opus_gemm_arch_gfx1250.cuh:79 check_shape_4g()` + `opus_gemm_tune.py:391` | **4 处副本** |
| shape 合法性检查 | 各 launcher 里的 `AITER_CHECK`（K 偶数、tile 对齐…）与 `opus_gemm_tune.py:364 kid_rejects_shape()` 重叠 | 2 处副本 |

**结论：C++ 侧现在同时是 kernel 库、调优数据库、启发式策略引擎和显存分配器。要砍的是后三个。**

### 1.3 splitK workspace 现状（**任务一**的靶子）

`opus_gemm.cu:341` 的 `SplitkWsRegistry`：

- 进程级 `unordered_map<hipStream_t, Owner*>` + `std::mutex`，**永不释放**（`new Owner` 泄漏、`hipMalloc` 永久持有，只在 grow 时 free 旧的）。
- 每个 stream **两份 handle**：`hipHostMalloc(Coherent)` 的 host shadow（`opus_splitk_ws_get`）+ `hipMalloc` 的 device mirror（`opus_splitk_ws_device_handle`），靠 `opus_splitk_ws_sync_to_device()` 手工同步。gfx942/gfx1250 的 kernel 读 device mirror（注释解释：读 host pinned 是 PCIe coherent read，慢 ~5×）。多这一跳纯粹是为了让 HIP graph 烘焙一个稳定地址。
- grow：4 MiB 对齐 + `hipDeviceSynchronize()` + `hipFree`；capture 期间 grow 直接抛错，于是被迫暴露 `opus_gemm_workspace_init()` 让用户在 capture 前手动预热，**再于 `tuned_gemm.py` 里补 75 行猜测 torch 默认 capture stream 的代码**去替用户调它。
- `opus_splitk_ws_handle` 结构体在 3 个 traits 头里**各定义一遍**（`gfx950:593` / `gfx942:106` / `gfx1250:45`），靠 `#ifndef` guard 防冲突。

**三个 arch 的 workspace 尺寸公式互不相同**——这是任务一最容易出错的地方：

| arch | 公式 | 出处 |
|---|---|---|
| gfx950 | `split_k * batch * padded_M * padded_N * sizeof(float)` | `gen_instances_gfx950.py:1335` |
| gfx942 | `split_k * batch * padded_M * padded_N * sizeof(Traits::D_C)` ← **D_C 可能是 bf16** | `gen_instances_gfx942.py:505` |
| gfx1250 | `split_k * padded_M * padded_N * sizeof(float)` ← **无 batch**（该 pipeline 只支持 batch==1） | `gen_instances_gfx1250.py:299` |

**问题**：与 torch caching allocator 完全脱节（同一进程两套显存池，OOM 时互不感知）、TBO 下靠 mutex map 打补丁、graph capture 需要框架猜 stream 去预热、二级指针多一次 device 端 load。

### 1.4 dtype / layout 现状（**任务二 R4** 的靶子）

- `opus_gemm.cu:289-322`：手写 if-else 树 `is_splitk ? force fp32_t : (Y.dtype 分支)`。
- `opus_gemm.cu:133-181`：fp8 支再来一遍 `has_scale × Y.dtype` 分支，且只允许 fp32 输出——规则散在三处。
- Python 侧 `gemm_op_a16w16.py:_validate_and_reshape`（~130 行）+ `_check_a16w16_tune_layout`（~77 行）：A/B/Y/bias 的 dim / shape / stride / dtype 校验，与 C++ `AITER_CHECK` 和 codegen 的 `BIAS_HOST_VALIDATE` 三处重叠。
- gfx1250 的 **batch==1 硬约束只写在 codegen 注释和 C++ 里**，python 侧没有对应的门（`gemm_op_a16w16.py` 的 batch 校验是通用的 stride 检查，不含 per-kid 约束）。

---

## 2. 目标架构

### 2.1 分层契约

> **C++ = 「kid + tensors + workspace → launch」。别的什么都不做。**

```
┌─ Python / torch 层（aiter 框架） ──────────────────────────────────┐
│  1. dtype / layout / shape 校验（唯一一处）                        │
│  2. tuned CSV 查表（唯一一处，pandas）                             │
│  3. heuristic 选 kid（唯一一处，从 C++ 搬过来）                    │
│  4. splitK workspace: torch.empty 分配（唯一一处，走 caching alloc）│
│  5. 输出 tensor 分配 / reshape                                     │
└───────────────────────────────┬────────────────────────────────────┘
                                │ opus_gemm_<family>_tune(tensors…, kid, splitK, ws)
┌───────────────────────────────▼────────────────────────────────────┐
│  C++ 层                                                            │
│  6. kid → launcher 函数指针（opus_gemm_a16w16_tune_lookup.h）      │
│  7. launcher: 填 kargs、算 grid、launch main (+ reduce) kernel      │
│  8. AITER_CHECK 只保留「kernel 物理约束」（最后一道保险）           │
└────────────────────────────────────────────────────────────────────┘
```

### 2.2 单一事实来源（SSOT）表

| 事实 | 唯一归属 | 消费方 |
|---|---|---|
| kid → 完整 kernel 元信息（tile / tag / has_oob / bias / ws dtype / arch） | `csrc/opus_gemm/opus_gemm_common.py`（已经是，但被 3 处副本削弱） | codegen、python dispatch、python tuner |
| tuned 结果 | `aiter/configs/**/*_tuned_gemm.csv` | 只有 `aiter/ops/opus/common.py` 读 |
| heuristic 策略 | 新建 `aiter/ops/opus/heuristic.py` | 只有 python dispatch |
| shape 合法性规则 | `opus_gemm_tune.py:kid_rejects_shape()`（提取到共享模块） | python dispatch + tuner |
| workspace 尺寸公式 | 从 `opus_gemm_common.py` 的 kid 元信息推导 | python 分配器 |
| kid → launcher 函数指针 | `opus_gemm_a16w16_tune_lookup.h`（codegen 产物，**保留**） | C++ |

**关键前提（已验证）**：`csrc/opus_gemm/opus_gemm_common.py` 是纯 python 模块，运行时可 import。所以 kid 元信息**不需要额外 codegen 一份 JSON、也不需要新建一个 catalog 模块**——直接 import 现有的。C++ 唯一需要的 kid 知识就是「kid → 函数指针」，这本来就是 codegen 产物。

> **为什么不新建 `aiter/ops/opus/_kernel_catalog.py`**（另一版方案的提议）
>
> 动机是「runtime 不该反向 import `csrc/`」。但这个顾虑不成立：
> - `aiter/jit/core.py:408` 已经 `sys.path.insert(0, AITER_META_DIR)`，集中做一次，不是运行时临时改 path；
> - `from csrc.X import Y` 是仓内既有模式：`aiter/ops/sampling.py:7`、`aiter/ops/attention.py:11`、`aiter/aot/pa.py:4`；
> - #4320 正是这么做的：`from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bmm_kernel_lists`；
> - `setup.py:169` 会把 `csrc/` 整个 copytree 进 `aiter_meta/`，装完的 wheel 里也在。
>
> 新建 catalog 的实际后果是：为了消灭 metadata 副本，先造一个副本，然后要把 codegen、tuner 的所有调用点逐个迁过去。**该做的是把字段补进现有的 `OpusGemmInstance`。**
>
> 不过那版提的字段设计值得抄进来——现在 `OpusGemmInstance` 缺的：`supports_m_tail` / `supports_n_tail` / `supports_k_tail`（现在散在 `kid_rejects_shape` 的 if 里）、`supports_batch`（gfx1250 的 batch==1 约束现在只写在注释里）、`max_tensor_bytes`（4 GiB 守卫，见 §4.1）、`ws_includes_batch`（§3.1 坑 2）、`bias_dtypes`。

### 2.3 §2.1 的分层在 #4320（zoe）里已经落地了

上面的契约不是纸上设计——PR #4320 的 opus BMM 模块（新增，与现有 opus GEMM 平行）**已经完全按这个形态写了**。任务二本质上是把 GEMM 侧对齐到 BMM 侧已经达到的水平。逐条对照：

| §2.1 契约 | #4320 的做法 |
|---|---|
| C++ 只做 kid → 函数指针 | `opus_bmm.cu` 唯一入口 `opus_bmm_a8w8_mxscale(O, wo_a, Y, x_scale, w_scale, splitK, kernelId)`，实现是 `kTune.find(id)` 单次表查找 + 一个默认 kid 兜底。注释原话：「Single table lookup instead of a ~40-case switch」。**无 CSV、无 heuristic、无 hipMalloc** |
| heuristic 在 python | `bmm_op.py::_heuristic_mxscale_kid(g, m, n, k)`，纯 python if-else |
| CSV 查表在 python，且只有一处 | 在 `batched_gemm_a8w8_mxscale` 里；`bmm_op.py` 的 docstring 明说 tuned id 是「read one layer up … which hands the tuned id down」 |
| kid 元信息单一来源 | **直接 import codegen 模块反查，不手写副本**：<br>`from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bmm_kernel_lists`<br>`return {int(kid): int(inst.m_align) for fam in ... for kid, inst in fam.items()}`（`functools.cache` 修饰） |
| CSV 命中的 kid 也要过 shape 门 | `if kernelId is not None and not _kid_runs_m(int(kernelId), m): kernelId = splitK = None` — 拒绝时 kid 和 splitK **一起丢**，然后落回 heuristic |

最后一条尤其值得抄。#4320 的注释记了它为什么这么做：那张 `m_align` 表是用来替换「已经漂移了的手工维护副本」的，漂移的后果是 kid326 在从未调优过的非对齐 M 上被派发，损失 ~9%。**这正是 §4.1 要消灭的东西、和 §4.2b 要修的那个「CSV 命中后无条件信任 solidx」缺陷，人家已经踩过并修好了。**

⚠️ **排期冲突**：0810 树里**完全没有 bmm**（无 `opus_bmm.cu`、无 `aiter/ops/opus/bmm_op.py`）。#4320 是即将进来的新模块，而它要改的 `gen_instances.py` / `opus_gemm_common.py` / `opus_gemm_pybind.cu` / `rocm_ops.hpp` / `optCompilerConfig.json` **和任务二是同一批文件**。见 §8.6。

---

## 3. 任务一 — splitK workspace 切换到 torch

按 #4246 commit `4ac1fa8` 的形态一次到位，不重演它中间那版 `ws_handle` 间接层。**但覆盖三个 arch**——上游只切了 gfx1250。

**边界**：只动 workspace 的分配者与传递方式。不碰 dispatch、不碰选核逻辑、不碰 dtype/layout 校验。改完后 `gemm_a16w16_opus` / `opus_gemm_a16w16_tune` 的外部行为完全不变（除 `opus_gemm_workspace_init` 被废弃）。

### 3.1 前置：kid workspace 元信息

python 要能算出「这个 kid 这次调用需要多少字节 workspace」。数据已在 `opus_gemm_common.py`（`OpusGemmInstance` 有 `B_M` / `B_N` / `kernel_tag` / `splitk_workspace_dtype`），只是没有统一出口。加两个函数：

```python
def kid_workspace_dtype(kid: int) -> str | None:
    """splitK kid 返回 'fp32_t' / 'bf16_t'；非 splitK kid 返回 None。"""

def kid_workspace_plan(kid: int, batch: int, M: int, N: int, K: int,
                       split_k: int) -> tuple[tuple[int, ...], str] | None:
    """返回 (shape, dtype_str)；None 表示该 kid 不需要 workspace。

    返回 shape 而非 byte 数：typed tensor 才能让 C++ 侧校验 dtype，
    而 #4246 挂机的根因就是 dtype 错配（见坑 1）。
    """
```

> **为什么是 shape+dtype 而不是 byte 数**：`torch.empty(nbytes, dtype=uint8)` 的扁平 buffer 传到 C++ 只剩一个 `void*`，§3.3 那条尺寸断言就只能比字节数，查不出「按 bf16 大小分配、却被写 fp32」这类错。typed tensor 让 C++ 能同时断言 dtype 和 numel，把坑 1 从「靠 python 算对」变成「算错也会被拒」。

**三个坑，逐一说明：**

**坑 1 — ws dtype 必须读 kid 自己的属性。** gfx942 有 `bf16ws` 变体（`opus_gemm_common.py:569` / `:598` 把 `splitk_workspace_dtype` 设为 `"bf16_t"`）。硬编码 `sizeof(float)` 会给这些 kid 分配 2× 的量（浪费，不出错）；反过来若哪天有 kid 需要更宽的 ws，就是越界写——PR #4246 踩的正是这个坑（"corrupting adjacent device memory and hanging the machine"）。

**坑 2 — batch 项 per-arch 不同。** gfx1250 的公式**不含 batch**（§1.3 表）。`kid_workspace_bytes` 必须按 `kid_arch(kid)` 分支，不能写一个统一公式。建议直接把公式挂在 kid 元信息上（给 `OpusGemmInstance` 加一个 `ws_includes_batch: bool = True`，gfx1250 的构造函数里置 False），而不是在分配器里 `if arch == "gfx1250"`——后者又是一处 arch 特判副本。

**坑 3 — launcher 会 clamp split_k。** 三个 arch 的 launcher 各有一段 `while (split_k > 1) { ...; split_k--; }`（gfx950:1302 / gfx942:481 / gfx1250:277），所以 python 传进去的 `splitK` ≠ kernel 实际用的 `split_k`。两个选项：

- (a) 在 python 复刻 clamp 循环——精确，但又多一份要同步的逻辑；
- (b) **按未 clamp 的上界分配**——clamp 只会让 `split_k` 变小，上界恒安全。

**推荐 (b)。** 复刻 clamp 恰恰是这次重构要消灭的那种重复。代价是最多多分配 `(splitK_requested / splitK_actual)` 倍，而 clamp 通常只减 1~2。

### 3.2 ABI 变更

```python
# gen_instances.py:65 SPLITK_REDUCE_ABI_MAP —— 三个 arch 条目统一改
# 旧
"ws_arg":  "const opus_splitk_ws_handle* ws_handle",
"ws_type": "const opus_splitk_ws_handle*",
# 新
"ws_arg":  "void* ptr_ws",
"ws_type": "void*",
```

kargs 里 `ws_handle` 字段改为 `void* ptr_ws`；kernel 侧 `opus_splitk_ws_ptr<D_C>(kargs.ws_handle)` 改为直接 `reinterpret_cast<D_WS*>(kargs.ptr_ws)`（少一次 device 端 load）。

**顺带把 `D_C` 拆开——这是坑 1 的病根。** 现在 traits 里的 `D_C` 同时被当成「累加类型」和「workspace 元素类型」用，gfx942 的 `sizeof(typename Traits::D_C)`（`gen_instances_gfx942.py:506`）就是这个歧义的直接产物。改名为三个各自明确的别名：

```cpp
using D_ACC = fp32_t;   // 寄存器内累加，恒 fp32
using D_WS  = ...;      // workspace 元素类型（bf16 或 fp32，由 kid 决定）
using D_OUT = ...;      // 最终输出类型
```

改名本身是机械的，但它把「workspace 该多大」从一个需要读注释才知道的问题变成类型系统里的显式事实——比在分配器里加校验治本。这条独立于任务一的其余部分，可以先做。

删除 `opus_splitk_ws_handle` 的**三份**结构体定义（`gfx950/opus_gemm_traits_a16w16_gfx950.cuh:593`、`gfx942/opus_gemm_traits_a16w16.cuh:106`、`gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh:45`）及其 `#ifndef` guard。

**验收：全仓 `grep -rn ws_handle` 零命中。**

### 3.3 两侧改造

**C++ 入口签名**——workspace 作为**末位默认参数**，现有 5 参调用点（`tuned_gemm.py:625`、`op_tests/*`）零改动：

```cpp
void opus_gemm_a16w16_tune(aiter_tensor_t& XQ, aiter_tensor_t& WQ, aiter_tensor_t& Y,
                           std::optional<aiter_tensor_t> bias,
                           int kernelId, int splitK,
                           std::optional<aiter_tensor_t> workspace = std::nullopt);
```

C++ 侧保留**三个** `AITER_CHECK`：

```cpp
AITER_CHECK(!kid_needs_ws || workspace.has_value(), "kid ", kernelId, " requires a workspace");
AITER_CHECK(ws.dtype() == kD_WS_dtype, "workspace dtype mismatch: kid ", kernelId,
            " writes ", kD_WS_dtype, ", got ", ws.dtype());
AITER_CHECK(ws_elems_needed <= ws.numel(),
            "workspace too small: need ", ws_elems_needed, " got ", ws.numel());
```

后两条是防 §3.1 三个坑的最后一道防线，且**不违反分层**——launcher 本来就已经算出了 `padded_M * padded_N * split_k`（而且是 clamp 之后的精确值），两次比较零成本。

**dtype 那条是关键**：#4246 挂机（"corrupting adjacent device memory and hanging the machine"）的直接原因就是「写 fp32 的 kernel 拿到按 bf16 算大小的 buffer」。只比 numel 查不出这个——numel 是对的，元素宽度是错的。

> `aiter_tensor_t` 没有 `nbytes()`，只有 `numel()`（`csrc/include/aiter_tensor.h:22`）和 `element_size()`（`:25`）。

**Python 侧分配**——用 typed tensor，不用 `uint8` 扁平 buffer：

```python
def _alloc_splitk_workspace(kid, batch, M, N, K, split_k, device):
    plan = opus_gemm_common.kid_workspace_plan(kid, batch, M, N, K, split_k)
    if plan is None:
        return None
    shape, dtype_str = plan
    return torch.empty(shape, dtype=_TORCH_DTYPE[dtype_str], device=device)
```

kernel 确实是按 `void*` + 自算 stride 访问的，扁平 `uint8` 也能跑。但 typed tensor 让上面那条 dtype 断言成立——这个收益压过「typed shape 可能暗示 stride 有语义」的顾虑。#4246 也是 typed。

**HIP graph**：不需要任何特殊处理。torch caching allocator 在 capture 期间返回 replay-stable 的地址——这正是 PR #4246 能删掉 prewarm 的原因。

### 3.4 删除旧机制

| 位置 | 内容 | 行数 |
|---|---|---|
| `opus_gemm.cu:325-429` | `SplitkWsRegistry` + `opus_splitk_ws_get` + `opus_splitk_ws_device_handle` + `opus_splitk_ws_sync_to_device` + `opus_gemm_workspace_init` + 17 行 rationale 注释 | 105 |
| `opus_gemm.cu:27-29` | `<mutex>` / `<unordered_map>` include | 2 |
| `opus_gemm.h:44-45` | `opus_gemm_workspace_init` 声明 | 2 |
| `rocm_ops.hpp:296-302` | `OPUS_GEMM_WORKSPACE_INIT_PYBIND` | 7 |
| `opus_gemm_pybind.cu:18` | 宏调用 | 1 |
| `gemm_op_a16w16.py:520-529` | python binding + `__init__.py:44/50/56` 导出 | ~12 |
| `gen_instances_gfx950.py:1317-1357` | extern 声明 + capture 探测 + grow 块 | 41 |
| `gen_instances_gfx942.py:495-528` | 同上 | 34 |
| `gen_instances_gfx1250.py:289-326` | 同上 + device mirror 获取 | 38 |
| **`aiter/tuned_gemm.py:41-125`** | **整个 prewarm 机制**（`_opus_needs_ws_prewarm` / `_opus_graph_capture_stream` / `_opus_prewarm_capture_workspace`）+ 调用点 `:623` | **~85** |

**净删除 ~325 行 C++/python**，新增 ~80 行。

`tuned_gemm.py` 那 85 行是意外收获——它整段存在的唯一理由是「猜 torch 会在哪个 stream 上 capture，好在那儿提前调 `workspace_init`」。torch 分配器天然 capture-aware，这个猜测游戏直接消失。

`opus_gemm_workspace_init` 是**公开 API 的破坏性删除**。仓内调用点只有 `tuned_gemm.py`（同批删除），但下游 vLLM/sglang 可能有。处理：python 保留空壳 + `DeprecationWarning` 一个 release，C++ 侧直接删。

### 3.5 任务一验收

| 项 | 要求 |
|---|---|
| 机械检查 | 全仓 grep `ws_handle` / `opus_splitk_ws_` / `SplitkWsRegistry` / `workspace_init` 零命中（除 deprecation 空壳） |
| 数值 | `python3 op_tests/test_opus_a16w16_gemm.py` 全绿 |
| **三 arch 覆盖** | gfx950 / gfx942 / gfx1250 各跑一遍。**gfx1250 的 batch 项不同、gfx942 的 ws dtype 不同**，只测一个 arch 必漏 |
| **CUDA graph** | 新增 capture/replay 用例 + golden 对拍。这是原机制存在的全部理由，必须证明新机制也成立 |
| **双 stream 并发** | 新增用例，覆盖原 TBO 场景（两 CPU 线程各驱动一 stream 同时跑 splitK） |
| **ws dtype 覆盖** | 必须同时覆盖 fp32-ws kid 与 bf16-ws kid（gfx942 的 bf16ws 变体）——§3.1 坑 1 只在后者暴露 |
| **workspace 契约负例** | 新增 `op_tests/test_opus_workspace.py`：正好够大、**少 1 个元素**、错 dtype、错 device、noncontiguous、caller 自带 workspace 复用。§3.3 那三条 `AITER_CHECK` 每条都要有对应的负例，否则等于没写 |
| 性能 | splitK shape sweep 前后对比，不应回退（少一次 device 端 handle load，理论上略好） |

后六项**都不能省**。

---

## 4. 任务二 — 重构 opus gemm 的各种入口

| 阶段 | 内容 | 覆盖需求 |
|---|---|---|
| 4.1 | kid 元信息去重（无行为变化） | —（前置） |
| 4.2 | dispatch 下沉到 python + 入口收敛 | R1, R2, R3 |
| 4.3 | dtype / layout 整理 | R4 |
| 4.4 | 三 arch 遗留差异清理 | R2 |

§4.2 是核心（R1 + R3 拆不开：删掉 C++ dispatch 的同时必须在 python 补上，中间态不可用）。§4.3 / §4.4 可并行、可拆独立 PR。

### 4.1 kid 元信息去重（纯 plumbing）

在 `opus_gemm_common.py` 补全公开查询 API（数据已存在，只是没有统一出口）：

```python
def kid_info(kid: int) -> OpusGemmInstance | None
def kid_is_splitk(kid) -> bool          # 取代 4 处硬编码区间判断
def kid_supports_bias(kid) -> bool      # 取代 opus_gemm.cu:255 + kid_rejects_bias()
def kid_arch(kid) -> str                # "gfx950" / "gfx942" / "gfx1250"
def kid_exceeds_4g(kid, M, N, K) -> bool
# kid_workspace_dtype / kid_workspace_bytes 已在任务一加入
```

删除副本：

- `gemm_op_a16w16.py:289-290` 的 `_SPLITK_KID_MIN/MAX`（注释里已经承认「Kept here only as a documentation anchor」——那就删）
- `aiter/ops/opus/common.py:192-199` 的 `_MONO_TILE_KID_TILES`（改为从 `opus_gemm_common` 推导；它自己的注释就写着「Must stay in lock-step with…」）
- `opus_gemm_arch_gfx950.cuh:100-112` 的 `kSplitkKidMin/Max` + `kid_is_splitk`
- `opus_gemm.cu:186-265` 全部 kid 分类常量与谓词（80 行）

后两条依赖 §4.2 完成（那时 C++ 不再需要知道 kid 语义）；本阶段先加 python API 并把 python 副本切过去，C++ 副本标 TODO。

**4 GiB 守卫合并**：`opus_gemm_arch_gfx950.cuh:196` + `:233` + `opus_gemm_arch_gfx1250.cuh:79` + `opus_gemm_tune.py:391` **四处**合成 `kid_exceeds_4g()`。

**验证**：op_test 全绿；`gen_instances.py` 产物 diff 为空（这一步不应改变任何生成代码）。

### 4.2 dispatch 下沉（R3）+ 入口收敛（R1）

#### a. heuristic 搬到 python

新建 `aiter/ops/opus/heuristic.py`：

- **gfx1250**（`opus_gemm_heuristic_dispatch_gfx1250.cuh:38`，48 行）——已是纯 `(M,N,K,has_bias)→kid`，机械翻译，**先做这个**（最小、最容易验证）。
- **gfx950**（同名头 `:83`，127 行）——同样是 kid 形态，机械翻译。
- **gfx942**（同名头，157 行）——**直接返回 launcher 符号名**，落后一代。翻译前先改写成 kid 形态：符号名→kid 的映射从 `opus_gemm_common.py` 的 `gfx942_*_kernels_list` **反查生成**，不要手写。

```python
def heuristic_kid(arch: str, M, N, K, *, has_bias=False, out_dtype=None) -> int:
    return _HEURISTIC_BY_ARCH[arch](M, N, K, has_bias)
```

#### b. 统一 dispatch 函数

```python
# aiter/ops/opus/dispatch.py（新建）
@functools.cache
def resolve_kid(M, N, K, *, bias, dtype, outdtype, arch) -> tuple[int, int]:
    """返回 (kid, splitK)。两级：CSV → heuristic。"""
    cfg = common.lookup_tuned(M, N, K, bias=bias, dtype=dtype, outdtype=outdtype)
    if cfg is not None and not kid_rejects_shape(cfg["solidx"], M, N, K):
        return cfg["solidx"], cfg["splitK"]
    return heuristic.heuristic_kid(arch, M, N, K, has_bias=bias), 0
```

两条选择规则，都是修现存缺陷：

**(i) CSV 命中的 kid 也要过 shape 门。** `gemm_op_a16w16.py:498` 现在命中 CSV 后**无条件信任** solidx，只对 mono-tile 做了 `mono_kid_shape_ok` 特判。统一走 `kid_rejects_shape` 后，任何 kid 的 shape 不匹配都降级到 heuristic，而不是靠一堆 kid-family 特判打补丁。#4320 已经这么做了，且拒绝时 **kid 和 splitK 一起丢**（`kernelId = splitK = None`）——tuned row 的 splitK 是配着那个 kid 调出来的，单独留着没意义。

**(ii) explicit kid 不兼容时抛错，不静默换 kernel。** 用户显式传 `kernelId=` 是在表达「我就要这个」，多半是在做性能对比或复现。静默换成另一个 kernel 会让 benchmark 结果无法解释。只有 CSV/heuristic 选出的 kid 才允许降级。

#### b2. runtime CSV 查询范围（一个现存 bug）

`aiter/ops/opus/common.py:_load_tuned_dict()` 把 `model_configs/*_bf16_tuned_gemm.csv` **全部 glob 进来**，concat 后按 `us` 升序 first-wins。也就是说：跑 A 模型时，如果 B 模型的 CSV 里有同 shape 且 `us` 更小的行，**会命中 B 的配置**。key 是 9 元组 `(cu_num, M, N, K, bias, dtype, outdtype, scaleAB, bpreshuffle)`，不含模型标识，撞车无从察觉。

build-time 和 runtime 对 CSV 的用途必须分开：

| | 用途 | 该读哪些 CSV |
|---|---|---|
| build-time（`gen_instances.py --tune_files`） | 决定编译哪些 kid 进 .so | **取并集**。多编几个 kid 只是编译时间代价 |
| runtime（`lookup_tuned`） | 选这次用哪个 kid | **只读框架当前明确选定的配置文件**（`AITER_CONFIG_*` 指定的那个），不 glob |

另外 key 里应加上 `gfx`。CSV schema 已有该列（`common.py:23` 的注释确认），只是 `_KEY_COLUMNS` 没用。旧 CSV 缺 `gfx` 时，仅在当前 arch 唯一无冲突的情况下兼容；新写出的行必须带。

这条**不阻塞**任务二的其余部分，但既然要重写 lookup 层，顺手修掉比留着划算。

#### c. C++ 侧删除

| 位置 | 内容 | 行数 |
|---|---|---|
| `gen_instances.py:430-535` | `gen_lookup_dict()`（**不是** `gen_a16w16_tune_lookup`，见 §1.1 警告） | 108 |
| `gen_instances.py:855+` | `get_tune_dict()` | ~45 |
| `gen_instances.py:1129-1172` | CSV → `opus_gemm_lookup.h` 烘焙块 | ~45 |
| `opus_gemm_lookup.h` | 整个（生成物） | — |
| `opus_gemm_heuristic_dispatch_gfx1250.cuh` | 整个文件 | 48 |
| `opus_gemm_heuristic_dispatch_gfx942.cuh` | 整个文件 | 157 |
| `opus_gemm_heuristic_dispatch_gfx950.cuh` | 只删 heuristic 函数体。文件里还有 `OpusA16W16NoscaleKernel` typedef 和 `opus_gfx950_detail::*`，被 gfx942/gfx1250 复用——**趁机把它们挪到 `opus_gemm_common.cuh`**，然后整个文件也删掉 | 127 |
| `opus_gemm_arch_gfx950.cuh:163-248` | `opus_dispatch_a16w16_gfx950<>` 两个特化 | 86 |
| `opus_gemm_arch_gfx1250.cuh:76-134` | `check_shape_4g` + `opus_dispatch_a16w16_gfx1250<>` | 59 |
| `opus_gemm_arch_gfx942.cuh` | 对应的 `opus_dispatch_a16w16_gfx942<>` | ~50 |
| `opus_gemm.cu:54-81` | arch router `opus_dispatch_a16w16<>` | 28 |
| `opus_gemm.cu:160-177` | `opus_gemm()` 的 bf16 分支 | 18 |

**同时把保留的那张表改名**：`opus_gemm_a16w16_tune_lookup.h` → **`opus_gemm_a16w16_kid_dispatch.h`**，生成函数 `gen_a16w16_tune_lookup()` → `gen_a16w16_kid_dispatch()`。§1.1 那个「两张 lookup 表只能删一张」的风险，与其靠 review 警惕，不如让剩下那张的名字不再含 "lookup" —— 改完之后 `grep -rn lookup csrc/opus_gemm` 的命中就都是该删的。纯改名，零行为变化，建议**在删除动作之前**单独一个 commit。

> **重要副作用（务必确认）**：`gen_instances.py:1044` 的 subset-compile 集合是
> `S = (csv_kids | sidecar_kids | HEURISTIC_DEFAULT_KIDS) & valid_kids`。
> 删掉 C++ lookup 后 **`csv_kids` 仍必须保留**——它决定的是「编译哪些 kid 进 .so」，不是「运行时怎么查表」。这两件事被现有代码耦合在同一段逻辑里（CSV 既喂 `S` 又喂 `gen_lookup_dict`），拆时容易连坐删掉，后果是 CSV 里的 kid 编不进 .so、运行时 `AITER_CHECK(kid not found)`。
> `HEURISTIC_DEFAULT_KIDS` 同理保留——它保证的不变式从「C++ heuristic 返回的 kid 一定在 S 里」变成「python heuristic 返回的 kid 一定在 S 里」，需要在 `heuristic.py` 加 import-time 自检断言两者一致。

#### d. `opus_gemm()` 入口的处置

删掉 bf16 分支后，`opus_gemm()` 只剩 fp8 分支（`opus_gemm.cu:133-158`，且那里也是硬编码单 launcher、无 tuned 表、gfx950-only）。

- **推荐**：整个删掉，fp8 路径改造成 `opus_gemm_a8w8_tune(XQ, WQ, Y, x_scale, w_scale, kid)`，python 侧负责选 kid（现在 C++ 里那点 `has_scale` 判断照搬即可）。
- 保守：保留 `opus_gemm()` 作为 deprecated 薄壳，只转发 fp8。

推荐前者：R1 要的就是入口收敛，留一个只服务半个 dtype 的万能入口是反效果。

#### e. 最终 C++ 入口清单（3 → 2）

```cpp
void opus_gemm_a16w16_tune(XQ, WQ, Y, bias, kernelId, splitK, workspace);
void opus_gemm_a8w8_tune  (XQ, WQ, Y, x_scale, w_scale, kernelId);
```

两个都是纯「kid → 函数指针 → launch」。`opus_gemm.cu` 从 431 行降到 **~150 行**。

### 4.3 dtype / layout 整理（R4）

#### a. C++ 侧：表驱动取代 if-else 树

`opus_gemm.cu:289-322` 的嵌套分支替换为显式表：

```cpp
// (in dtype, out dtype, needs_ws) -> 模板实参 CDataType
// splitk kid 的 main kernel 硬编码 D_C=float（traits static_assert），
// Y 的实际 dtype 由 reduce kernel 在 launch 时决定 —— 这是唯一的非对角项，
// 值得一行注释而不是散在三个 if 里。
static constexpr OpusDtypeRule kA16W16Rules[] = {
    // in    out    needs_ws  ctype
    {bf16,   bf16,  false,    CType::BF16},
    {bf16,   fp32,  false,    CType::FP32},
    {bf16,   bf16,  true,     CType::FP32},   // splitk: ws=fp32, reduce 出 bf16
    {bf16,   fp32,  true,     CType::FP32},
};
```

配 `resolve_ctype(rules, in, out, needs_ws)`，命中失败给出含全部三元组的错误信息。a8w8 同构。

#### b. Python 侧：抽公共校验层

`_validate_and_reshape`（~130 行，面向 2D/3D 用户输入）与 `_check_a16w16_tune_layout`（~77 行，面向已 reshape 的 3D）语义大量重叠。抽 `aiter/ops/opus/layout.py`：

```python
@dataclass(frozen=True)
class GemmLayout:
    XQ: Tensor; WQ: Tensor; Y: Tensor
    batch: int; M: int; N: int; K: int
    squeeze_out: bool

def normalize_gemm_layout(A, B, *, out=None, out_dtype, bias=None,
                          in_dtypes, out_dtypes) -> GemmLayout
```

一处实现三条规则（源头都是 codegen 里硬编码的 `stride_*_batch`）：

1. A/B/Y 必须 contiguous 3D，stride 严格等于 `(M*K, K, 1)` / `(N*K, K, 1)` / `(M*N, N, 1)`
2. B 的 batch broadcast view（`stride(0)==0`）必须拒绝
3. bias 是 `[N]` 或 `[batch, N]`，dtype == out dtype

**顺带补一个现在缺的门**：gfx1250 的 cluster_tdm kid 只支持 `batch == 1`（`gen_instances_gfx1250.py:337` 注释确认），python 侧目前不检查，靠 C++ 或直接算错。加进 `kid_rejects_shape`。

现有的长篇错误信息（写得很好，解释了*为什么*拒绝）**原样保留**，只是搬到一处。

#### c. codegen 侧

`BIAS_HOST_VALIDATE` 在 python 已校验后是冗余的，但它是**最后一道保险**（tuner 和 C++ 直调不走 python 校验）。保留，但收敛成一个 codegen 共享片段而不是每个 emit 函数各贴一份。

### 4.4 三 arch 遗留差异清理（R2）

严格说 §4.2 做完 gfx950 就已经切到新机制上了。这里单列是为了强调遗留差异：

| 差异项 | gfx950 | gfx942 | gfx1250 | 处理 |
|---|---|---|---|---|
| heuristic 返回值 | int kid | **launcher 符号名** | int kid | gfx942 改成 kid，见 §4.2a |
| reduce kernel 名 | `splitk_reduce_kernel` | `..._fallback` | `..._gfx1250` | 保留（实现确实不同），ABI 统一为 `void* ptr_ws` |
| workspace dtype | 恒 fp32 | **有 bf16ws 变体** | 恒 fp32 | 统一走 `kid_workspace_dtype()` |
| workspace 含 batch | 是 | 是 | **否** | 挂到 kid 元信息（§3.1 坑 2） |
| batch 支持 | 多 batch | 多 batch | **仅 batch==1** | 进 `kid_rejects_shape`（§4.3b） |
| 4g_safe kid | 有（+5000） | 无 | 无 | 保持现状 |
| `SPLITK_REDUCE_ABI_MAP` | 三份 | | | 任务一后三份除 kernel 名外完全相同，合并为共享默认 + per-arch override |
| `OpusA16W16NoscaleKernel` typedef | 定义在 gfx950 heuristic 头，被另两 arch 复用 | | | 挪到 `opus_gemm_common.cuh`，解开反向依赖 |

### 4.5 任务二验收

| 项 | 要求 |
|---|---|
| 结构 | `opus_gemm.cu` ≤ 160 行；文件内不出现 CSV / heuristic / hipMalloc / shape 策略 |
| 入口数 | C++ 顶层入口 3 → 2，全部形如 `(tensors…, kernelId[, splitK][, workspace])` |
| 副本数 | kid 语义常量全仓仅 `opus_gemm_common.py` 一份；`opus_gemm_lookup.h` 不存在；改名后的 `opus_gemm_a16w16_kid_dispatch.h` **仍存在** |
| **codegen 回归** | 在 `op_tests/test_gemm_codegen.py` 加断言，把「不许倒退」写成测试而非 review 纪律：<br>① 生成产物中不含 `hipMalloc` / `hipFree` / `hipHostMalloc`<br>② 不生成任何 runtime shape lookup（产物里没有 `(M,N,K)` 键的表）<br>③ kid → launcher 映射唯一（无重复 kid）<br>④ 二次生成 diff 为空 |
| **kid 选择等价性** | 重构前后对同一 shape sweep（三 arch × 全 CSV 行 + 一批 CSV-miss shape）逐条对比选出的 (kid, splitK)，要求**完全相同**。这是任务二唯一能证明「没改行为」的手段，必须写成脚本而非抽查 |
| 编译集合 | 对每个 CSV opus 行断言 `solidx ∈ S`，防 §4.2c 那个连坐删除 |
| 数值 | `op_tests/test_opus_a16w16_gemm.py` 三 arch 全绿 |
| 性能 | 小 shape（M≤64）端到端 host 开销对比，确认 dispatch 下沉没引入可观测的 CPU 回退 |

---

## 5. Bonus — 新 kernel 接入 skill

**目标**（用户原话）：减少 agent 吐冗余代码和 comment，维持代码结构扁平简洁。

仓内已有两个 opus skill（`opus-kernel-best-practice` 编译期优化、`opus-module-build-optimization` build wall），**缺的正是「kernel 写完之后，怎么接进 aiter」这一段**——而这恰恰是 agent 最容易吐垃圾的地方（同时改 6~8 个文件，每处都有既定 pattern）。

新建 `.claude/skills/opus-kernel-integration/SKILL.md`：

1. **接入清单（按顺序）**——两个任务完成后的稳定版本：
   ```
   1. csrc/opus_gemm/include/<arch>/opus_gemm_pipeline_<name>.cuh   新 pipeline
   2. csrc/opus_gemm/include/<arch>/opus_gemm_traits_*.cuh          新 traits + kargs
   3. csrc/opus_gemm/opus_gemm_common.py                            新 kid（唯一元信息登记处）
   4. csrc/opus_gemm/codegen/gen_instances_<arch>.py                emit 函数 + register_emit
   5. aiter/ops/opus/heuristic.py                                   仅当要进 heuristic 兜底
   6. op_tests/opus/test_<name>.py                                  golden 对拍
   ```
   明确「**不需要改**」的文件（这是减少冗余输出的关键）：`opus_gemm.cu`、`opus_gemm.h`、`rocm_ops.hpp`、`opus_gemm_pybind.cu`、任何 python 入口——重构后新 kid 走既有 tune 入口，零 C++ 入口改动。

2. **kid band 分配表**——写死，避免 agent 自己发明编号：
   ```
   gfx950  [0, 10000)      splitk [200,300) / persistent [300,316) / mono [1400,1500)
                           nooob 镜像 +1000, cpol +2000/3000/4000, 4g_safe +5000
   gfx942  [10000, 20000)
   gfx1250 [20000, 21000)  cluster/TDM splitk（plain 段从 20000 连续分配）
   ```

3. **注释预算（硬约束）**：
   - codegen emit 函数：只在偏离既有 pattern 处注释，不重复解释 pattern 本身
   - **禁止**注释掉的旧代码
   - **禁止** AI 味的 changelog 式注释（"Previously this was X, now it's Y because…"）——这类信息属于 commit message。现状反例：`opus_gemm.cu:325-340` 那 16 行 registry rationale
   - 新增文件头注释 ≤10 行

4. **反模式清单**（每条指向仓内真实位置）：
   - 在 C++ 里加 shape → kernel 的策略判断（应去 python）
   - 在 C++ 里 `hipMalloc`（应 torch 分配后传入）
   - 复制 kid 区间常量到新文件（应 import `opus_gemm_common`）
   - 为新 kernel 加新的顶层 pybind 入口（应复用 `*_tune`）
   - 新 arch 复制一份 `check_shape_4g`（现状反例：已经有 4 份）

5. **验收命令**：`python3 op_tests/opus/test_<name>.py` + `ruff check` + `clang-format-18`。

Skill 应在**任务二完成之后**写——写在重构前只会固化现在这套要被删掉的 pattern。

---

## 6. 风险与缓解

### 任务一

| 风险 | 影响 | 缓解 |
|---|---|---|
| **workspace 尺寸算错**（PR #4246 踩过） | 越界写、机器挂死 | `kid_workspace_bytes` 必须读 kid 自己的 ws dtype 与 batch 语义（§3.1 坑 1/2）；单测覆盖三 arch；C++ 保留一次尺寸 `AITER_CHECK`（§3.3） |
| **gfx1250 无 batch 项被漏掉** | 分配 batch× 过大（浪费）或反向漏算 | 把 `ws_includes_batch` 挂到 kid 元信息，别在分配器里 arch 特判 |
| **split_k clamp 导致 python 与 kernel 不一致** | 分配偏小 → 越界 | 按未 clamp 上界分配（§3.1 方案 b） |
| **ABI 变更漏改某 arch 的 launcher** | 编译错，或更糟：静默用错地址 | ABI 集中在 `SPLITK_REDUCE_ABI_MAP`；改完全仓 grep `ws_handle` 必须零命中 |
| **HIP graph 行为变化** | capture/replay 出错 | 验收强制含 graph 用例；torch caching allocator 的 capture-aware 行为是 PR #4246 已验证的前提 |
| **`opus_gemm_workspace_init` 是下游公开 API** | 下游 vLLM/sglang 崩 | python 留空壳 + `DeprecationWarning` 一个 release |

### 任务二

| 风险 | 影响 | 缓解 |
|---|---|---|
| **删错 lookup 头**（`opus_gemm_lookup.h` vs `opus_gemm_a16w16_tune_lookup.h`） | 把 kid dispatch 一起删了，全线崩 | 见 §1.1 表。两个生成函数在 `gen_instances.py` 里紧挨着（430 / 537） |
| **subset-compile 集合 S 被连坐删除** | CSV 里的 kid 编不进 .so，运行时 "kid not found" | 见 §4.2c 引用块。加 CI：对每个 CSV opus 行断言 `solidx ∈ S` |
| **`HEURISTIC_DEFAULT_KIDS` 与 python heuristic 脱钩** | heuristic 返回未编译的 kid | `heuristic.py` import 时自检：所有可能返回值 ⊆ `HEURISTIC_DEFAULT_KIDS` |
| **gfx942 符号名→kid 翻译出错** | 选错 kernel，结果仍对但性能回退（最难发现） | 用 `opus_gemm_common.py` 的 kernels_list 反查**生成**，不手写；跑 §4.5 等价性对比 |
| **dispatch 下沉后每次调用多一次 python 开销** | 小 shape CPU-bound 吃亏 | `lookup_tuned` 已 `lru_cache`；`resolve_kid` 加 `functools.cache`，稳态一次 dict 查找。基线：现状 CSV 命中路径**本来就是** python 查表，只有 miss 路径从 C++ 变 python |
| **删 `opus_gemm()` 是 breaking change** | 下游直调 pybind 的会崩 | 见 §8；若确认有下游，退回 deprecated 薄壳 |

---

## 7. 交付顺序与规模估算

**任务一**（建议先做，单 PR）

| 步骤 | 内容 | 净变化 |
|---|---|---|
| §3.1 | kid workspace 元信息 | +45 |
| §3.2–3.4 | ABI 改 `void*` + python 分配 + 删 registry + 删 `tuned_gemm.py` prewarm | +35 / −325 |

小计约 **−245 行**。

**任务二**

| 步骤 | 内容 | 净变化 | 依赖 |
|---|---|---|---|
| §4.1 | kid 元信息 SSOT | +80 / −90 | 任务一 |
| §4.2 | dispatch 下沉 + 入口收敛 | +280 / −780 | §4.1 |
| §4.3 | dtype / layout 整理 | +150 / −290 | §4.2 |
| §4.4 | 三 arch 差异清理 | +40 / −130 | §4.2 |

小计约 **−740 行**。§4.3 / §4.4 可并行拆 PR。

**Bonus** §5：+200 行文档。

**合计约 −985 行**，`opus_gemm.cu` 431 → ~150 行，C++ 入口 3 → 2，bf16 dispatch 路径 3 → 1。

---

## 8. 待确认问题

**影响任务一：**

1. **workspace 参数加在现有入口尾部（默认 `nullopt`）还是另开入口**——前者对现有调用点零改动，本文按前者写。
2. **和 #4246 的关系怎么处理。** 该 PR 仍 open 且已经在 gfx1250 上做了任务一的 (a) 部分。三条路：
   - (i) 等 #4246 合入，任务一只补 gfx950/gfx942 两个 arch 并删掉 registry —— 工作量最小，但被上游 review 进度卡住（目前 valarLip 的 review 未到）；
   - (ii) 任务一独立做全三个 arch，与 #4246 冲突时由后者 rebase —— 任务一是纯 workspace 改动、#4246 的主体价值是 fuse pipeline，rebase 代价可控；
   - (iii) 把 (a) 从 #4246 拆出来单独先合。
   **倾向 (iii)**：所有权切换和 fuse pipeline 本来就该是两个 PR，拆开对双方 review 都更清晰，也解开任务一的依赖。

**影响任务二：**

3. **`opus_gemm()` 是删还是留薄壳**（§4.2d）——取决于下游有没有直接调 pybind 的 `opus_gemm`。
4. **a8w8 要不要一并改造成 `opus_gemm_a8w8_tune`**（§4.2d/e）——它现在没有 tuned 表、只有一个硬编码 launcher、gfx950-only。若近期不打算给 a8w8 做 tuning，保留现状、只把它从 `opus_gemm()` 里挪出来也够。
5. **既然 #4320 已经把 BMM 侧的分层做对了（§2.3），GEMM 侧要不要直接照抄它的模块布局**——即新建 `aiter/ops/opus/dispatch.py` + `heuristic.py`（本文 §4.2 的写法），还是像 BMM 那样把 heuristic 和 kid 校验都放进单个 op 文件。BMM 的 kernel 家族少所以单文件够用；GEMM 有三个 arch，我倾向拆文件。

**跨任务：**

6. **#4320 的落地时机与任务二的文件冲突。** 两者都改 `gen_instances.py` / `opus_gemm_common.py` / `opus_gemm_pybind.cu` / `rocm_ops.hpp` / `optCompilerConfig.json`。建议 **#4320 先合、任务二后做**：#4320 是新增模块（改动集中在新文件 + 少量注册点），任务二是删改存量（大面积改现有文件）；反过来会让 #4320 整个重写一遍适配。同时任务二应把 BMM 一并纳入 §4.5 的「入口清单」验收，避免刚合进来的 BMM 又漂移出去。
