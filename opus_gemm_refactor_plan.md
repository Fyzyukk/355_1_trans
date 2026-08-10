# Opus GEMM 两大任务 — 修改架构方案

**任务一**：splitK workspace 切换到 torch 分配方式
**任务二**：重构 opus gemm 的各种入口

> 生成日期：2026-08-10
> 目标仓库：`/root/workspace/aiter`（`main` @ `44715057b`）
> 参考实现：ROCm/aiter PR #4246 `gfx1250 opus gemm splitk fuse`（branch `binglue/gfx1250-opus-gemm-splitk-fuse`）
> 任务来源：Lyu, Binglue 的两段需求（见 `11.png` / `download.png`）

---

## 0. 需求拆解：两个大任务

### 任务一 — splitK workspace 切换到 torch 分配方式

> 「还有 splitk 的 workspace 开辟需要切换到 torch 的方式」
> 「按这个方式（PR #4246），或者 zoe 那个也可」

**独立可交付**。改的是 workspace 的内存所有权：从 C++ 内部 `hipMalloc` registry 换成 python `torch.empty` 传指针进来。不触碰任何 dispatch / 入口逻辑。

### 任务二 — 重构 opus gemm 的各种入口

> 「重构一下现在 opus gemm 的各种入口之类的」
> 「需要把之前 gfx950 的部分也切到这上面来」
> 「需要把之前的 tune lookup 简化一下……python 有一套，c++ 还有一套。现在只留 python 这套就行了。主要是 heuristic 的部分」
> 「c++ 只维持目前的 kernel 相关部分，以及根据 kid dispatch 的功能。其余所有的工作都交给 aiter 这里的 torch 做」
> 「各种 dtype 和 layout 也需要在 opus_gemm.cu 那个文件以及 python interface 整理一下」

拆成 4 项子需求：

| # | 子需求 | 归类 |
|---|---|---|
| R1 | 重构各种入口 | 入口收敛 |
| R2 | gfx950 部分切到新机制上 | 跨 arch 统一 |
| R3 | tune lookup 只留 python 一套（主要是 heuristic） | 职责下沉 |
| R4 | `opus_gemm.cu` + python interface 的 dtype / layout 整理 | 接口清理 |

「c++ 只维持 kernel + kid dispatch，其余交给 torch」是这四项共同的总纲，不单列。

### Bonus — opus 新 kernel 接入 skill

> 「如果可能的话可以帮忙搞个 skill 来作为 opus 新 kernel 接入的方式。希望能减少 agent 吐冗余代码和 comment 还有维持代码结构的扁平简洁」
> 「这个属于是 bonus 部分了」

### 两个任务的关系

任务一是任务二总纲在「内存所有权」这一维上的一个实例（workspace 是 framework 的事，不是 kernel 的事），但**技术上完全独立**：任务一只动 workspace ABI 与分配点，任务二只动 dispatch 与入口签名，两者改的文件有交集（`opus_gemm.cu`、`gen_instances*.py`）但改的是不同区段。

**建议先做任务一**：规模小、边界清晰、有 PR #4246 作参照，且它删掉的 `opus_gemm.cu:414-593` 那 180 行正好是任务二要瘦身的同一个文件里最容易剥离的部分 —— 先剥掉，任务二的 diff 会干净很多。

下文 §3 是任务一的方案，§4 是任务二的方案，§5 是 bonus。

---

## 1. 现状盘点（问题在哪）

### 1.1 入口现状：5 个 C++ 入口 / 4 个 Python 入口 / 3 条 dispatch 路径

**C++ 侧**（`csrc/opus_gemm/include/opus_gemm.h`，实现在 `opus_gemm.cu`）：

| 入口 | 签名要点 | 干什么 |
|---|---|---|
| `opus_gemm` | 7 参（含 group_layout / x_scale / w_scale / bias） | 万能入口：按 XQ.dtype 分 fp8 / bf16 两大支；bf16 支再做 **C++ 内部 (M,N,K) 查表 + heuristic** |
| `opus_gemm_a8w8_bpreshuffle` | 5 参 | 硬编码 tile 选择（`M%256`/`M%128` if-else） |
| `opus_gemm_a8w8_bpreshuffle_tune` | +kernelId | switch(kid) 里只有 2 个 case（3 / 9002） |
| `opus_gemm_a16w16_tune` | +kernelId +splitK | 唯一"纯 kid dispatch"的入口，也是最健康的一个 |
| `opus_gemm_workspace_init` | 无参 | 给 splitK 注册 per-stream `hipMalloc` workspace |

**Python 侧**（`aiter/ops/opus/`）：`gemm_a16w16_opus`、`opus_gemm_a16w16_tune`、`gemm_a8w8_bpreshuffle_opus`、`opus_gemm_workspace_init`；外加第三方调用点 `aiter/tuned_gemm.py:501 opus_gemm()`（注册在 `solMap["opus"]`）。

**同一个 bf16 GEMM 有三条到达 kernel 的路径**：

```
路径 A  gemm_a16w16_opus → python lookup_tuned(CSV) 命中 → opus_gemm_a16w16_tune(kid)
路径 B  gemm_a16w16_opus → CSV miss → _opus_gemm_bf16_dispatch → C++ opus_gemm
                                        → C++ opus_gemm_lookup.h 查表（codegen 时从同一批 CSV 烘焙）
                                        → miss → opus_a16w16_heuristic_kid_gfx950/942()
路径 C  aiter/tuned_gemm.py solMap["opus"] → opus_gemm_a16w16_tune(kid)
```

路径 A 和路径 B 的第一段**读的是同一批 CSV**（`aiter/configs/bf16_tuned_gemm.csv` + `model_configs/*_bf16_tuned_gemm.csv`，都按 `libtype=='opus'` 过滤），只是一个在运行时用 pandas 读、一个在 codegen 时烘焙成 `opus_gemm_lookup.h`。**这就是"python 一套、C++ 还有一套"的具体所指。**

### 1.2 C++ 侧越界承担的 framework 职责

| 越界项 | 位置 | 规模 |
|---|---|---|
| (M,N,K) 调优表查找 | `gen_instances.py:413 gen_lookup_dict()` → `opus_gemm_lookup.h` → `opus_gemm_arch_gfx950.cuh:178/220` | ~110 行 codegen + 每 TU 两张 `lower_bound` flat array |
| heuristic 选核 | `opus_gemm_heuristic_dispatch_gfx950.cuh:83`（127 行）<br>`opus_gemm_heuristic_dispatch_gfx942.cuh`（157 行，且**直接引用 launcher 符号名**，比 gfx950 的 kid 版本更差） | 284 行 |
| splitK workspace 生命周期 | `opus_gemm.cu:430-593`（`SplitkWsRegistry` + `hipMalloc`/`hipFree` + per-stream mutex map + graph-capture 守卫），另有 ~70 行注释掉的历史版本 | ~165 行 |
| kid 语义分类 | `opus_gemm.cu:289-355`（`OPUS_SPLITK_KID_MIN/MAX`、`opus_kid_is_splitk` / `_is_persistent` / `_is_mono_tile` / `_supports_bias`）+ `opus_gemm_arch_gfx950.cuh:100-112` **重复一遍** + `gemm_op_a16w16.py:274` **再重复一遍** + `opus_gemm_common.py:737-834` **权威版本在这** | 4 处副本 |
| 4 GiB buffer-resource 守卫 | `opus_gemm_arch_gfx950.cuh:196-206/233-243` 与 `opus_gemm_tune.py:225-243` **同一条规则两处实现** | 2 处副本 |
| shape 合法性检查 | 各 launcher 里的 `AITER_CHECK`（K 必须偶数、mono-tile 对齐…）与 `opus_gemm_tune.py:197 kid_rejects_shape()` 重叠 | 2 处副本 |

结论：**C++ 侧现在同时是 kernel 库、调优数据库、启发式策略引擎和显存分配器**。要砍的是后三个。

### 1.3 splitK workspace 现状（**任务一**的靶子）

`opus_gemm.cu:439` 的 `SplitkWsRegistry`：

- 进程级 `std::unordered_map<hipStream_t, Owner*>` + `std::mutex`，永不释放（`new Owner` 泄漏、`hipMalloc` 永久持有，只在 grow 时 free 旧的）。
- 每个 stream 一个 device 端 `opus_splitk_ws_handle{void* ptr; unsigned long bytes;}`，kernel 通过**二级指针**读 workspace 地址（`kargs.ws_handle` → device 上再解引用）。多这一跳纯粹是为了让 HIP graph 能烘焙一个稳定地址。
- grow 用 4 MiB 对齐 + `hipDeviceSynchronize()` + `hipFree`；capture 期间 grow 直接抛错，于是被迫暴露 `opus_gemm_workspace_init()` 让用户在 capture 前手动预热。
- codegen 侧 `gen_instances_gfx950.py:1425-1449` 每个 splitK launcher 都内联一段 `extern` 声明 + capture 探测 + `ensure_capacity` 调用。

**问题**：与 torch caching allocator 完全脱节（同一进程两套显存池，OOM 时互不感知）、TBO 下靠 mutex map 打补丁、graph capture 需要用户配合调 init、二级指针增加一次 device 端 load。

### 1.4 dtype / layout 现状（**任务二 R4** 的靶子）

- `opus_gemm.cu:379-411`：手写 if-else 树 `XQ.dtype==bf16 × (is_splitk ? force fp32_t : Y.dtype)`。
- `opus_gemm.cu:158-206`：fp8 支再来一遍 `has_scale × Y.dtype` 分支，且 a8w8 只允许 fp32 输出、bpreshuffle 允许 fp32/bf16 —— 规则散在三个函数里。
- Python 侧 `gemm_op_a16w16.py:_validate_and_reshape`（124 行）+ `_check_a16w16_tune_layout`（60 行）：A/B/Y/bias 的 dim / shape / stride / dtype 校验，与 C++ 侧 `AITER_CHECK` 和 codegen 生成的 `BIAS_HOST_VALIDATE` 三处重叠。
- `gemm_op_a8w8_bpreshuffle.py` 里另有一套 `_reshape_3d` / `bpreshuffle_kernel_shape_ok` / `_fallback_kernel_id` / `_lookup_tuned_kernel_id`，与 a16w16 那套**结构相同但代码不共享**。

---

## 2. 目标架构

### 2.1 分层契约（一句话版）

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
│  6. kid → launcher 函数指针（flat array + lower_bound）            │
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
| shape 合法性规则 | `csrc/opus_gemm/opus_gemm_tune.py:kid_rejects_shape()`（提取到共享模块） | python dispatch + tuner |
| workspace 尺寸公式 | 从 `opus_gemm_common.py` 的 kid 元信息推导 | python 分配器 |

**关键前提（已验证可行）**：`csrc/opus_gemm/opus_gemm_common.py` 是纯 python 模块，运行时可 import（`opus_gemm_tune.py:69` 已经这么用了）。所以 kid 元信息不需要额外 codegen 一份 JSON —— 直接 import 即可。C++ 侧唯一需要的 kid 知识就是"kid → 函数指针"，这本来就是 codegen 产物。

---

## 3. 任务一 — splitK workspace 切换到 torch

按 PR #4246 第 2/3 步的**最终形态**做，一次到位 —— 不重演它中间那版 `ws_handle` 间接层（PR 里是 registry → ws_handle 指针 → 直接 torch 指针，走了三步；我们直接落到第三步）。

**边界**：只动 workspace 的分配者与传递方式。不碰 dispatch、不碰入口选核逻辑、不碰 dtype/layout 校验。改完后 `gemm_a16w16_opus` / `opus_gemm_a16w16_tune` 的外部行为完全不变（除了 `opus_gemm_workspace_init` 被废弃）。

四步：§3.1 kid workspace 元信息 → §3.2 ABI → §3.3 两侧改造 → §3.4 删除旧机制，§3.5 是验收标准。

---

### 3.1 前置：kid workspace 元信息（唯一一处）

python 要能算出「这个 kid 这次调用需要多少字节 workspace」。数据已经在 `csrc/opus_gemm/opus_gemm_common.py` 里（`OpusGemmInstance` 有 `B_M`/`B_N`/`kernel_tag`/`splitk_workspace_dtype`），只是没有统一出口。加两个函数：

```python
def kid_workspace_dtype(kid) -> str | None:
    """"fp32_t" / "bf16_t"；非 splitK kid 返回 None。"""

def kid_workspace_bytes(kid, batch, M, N, K, split_k) -> int:
    """batch * split_k * padded_M * padded_N * sizeof(该 kid 自己的 ws dtype)。
    非 splitK kid 返回 0。"""
```

**这是任务一最容易出事的地方**。必须按 *该 kid 自己声明的* ws dtype 算，不能全局假设 fp32 或 bf16：gfx942 有 `bf16ws` 变体（`opus_gemm_common.py:566 _with_bf16_splitk_workspace`），gfx950 现在恒 fp32。PR #4246 正是在这里踩了坑 —— fp32-writing 的 kid 拿到按 bf16 尺寸算的 buffer，"corrupting adjacent device memory and hanging the machine"。

`padded_M` / `padded_N` 的公式要与 launcher 里的 `num_tiles_m * B_M` / `num_tiles_n * B_N`（`gen_instances_gfx950.py:1420-1423`）逐字一致。建议加一个 codegen 期自检：emit launcher 时把它算出的 `ws_bytes` 表达式与 python 侧公式对拍。

顺带把 `split_k` 的钳位逻辑也搬过来 —— launcher 里有一段 `while (split_k > 1) { ... split_k--; }` 的 prefetch-k 钳位（`gen_instances_gfx950.py:1407-1414`），python 传进来的 `splitK` 与 kernel 实际用的可能不同。**workspace 必须按钳位后的值分配，否则分配偏大（浪费）或偏小（越界）**。两个选项：
- (a) python 复刻钳位逻辑（从 `kid_info(kid).prefetch_k_iter` 推导），按钳位后的值分配 —— 精确，但多一份需要同步的逻辑；
- (b) python 按**未钳位**的 `split_k` 分配（钳位只会让 split_k 变小，所以未钳位值是上界）—— 保守但绝不越界，代价是最多浪费几 MiB。

**推荐 (b)**，并在注释里写明「按上界分配，钳位后 kernel 只会用其中一部分」。理由：钳位逻辑是 kernel 的实现细节，复刻到 python 就等于又造了一份要同步的副本 —— 那正是这次重构要消灭的东西。

### 3.2 ABI 变更

`gen_instances.py:65 SPLITK_REDUCE_ABI_MAP`：
```python
# 旧
"ws_arg":  "const opus_splitk_ws_handle* ws_handle",
"ws_type": "const opus_splitk_ws_handle*",
# 新
"ws_arg":  "void* ptr_ws",
"ws_type": "void*",
```
kargs 里 `ws_handle` 字段 → `void* ptr_ws`；device 侧 `opus_splitk_ws_ptr<D_WS>()`（`gfx942/opus_gemm_traits_a16w16.cuh:113`、`gfx950/..._gfx950.cuh:593`）从"解引用 handle"退化为一次 `reinterpret_cast`，**少一次 device 端 load**。

`struct opus_splitk_ws_handle` 本身（`gfx950/opus_gemm_traits_a16w16_gfx950.cuh:593`、`gfx942/opus_gemm_traits_a16w16.cuh:106`，两处重复定义）可以整个删掉。

改完后 grep `ws_handle` 应为零命中 —— 这是本步的机械验收条件。

### 3.3 两侧改造

#### C++ 入口签名

```cpp
void opus_gemm_a16w16_tune(
    aiter_tensor_t& XQ, aiter_tensor_t& WQ, aiter_tensor_t& Y,
    std::optional<aiter_tensor_t> bias,
    int kernelId, int splitK,
    std::optional<aiter_tensor_t> workspace);   // 新增，尾参默认 nullopt
```
尾参 + 默认值 → pybind 层与既有 5 参调用（`tuned_gemm.py:537`、`op_tests/*`）二进制兼容，不需要一次性改所有调用点。

C++ 侧只做两件事：
1. `AITER_CHECK(!kid_needs_ws || workspace.has_value())` —— 需要 workspace 却没给，报清楚的错而不是 nullptr 解引用；
2. `AITER_CHECK(ws_bytes_needed <= ws.numel() * ws.element_size())` —— launcher 内部本来就要算 `padded_M * padded_N * split_k` 来填 stride，顺手比一下 python 给的 buffer 够不够。（`aiter_tensor_t` 没有 `nbytes()`，用 `numel() * element_size()`，见 `csrc/include/aiter_tensor.h:22/25`。）**这一条不能省**：它是 §3.1 那个「算错尺寸 → 挂机器」风险的最后一道防线，成本是每次 launch 一次整数比较。

注意这与「C++ 不做 framework 的事」不矛盾：launcher 是在用它**自己已经算出来**的值做校验，没有引入任何新的 kid 元信息副本。

#### Python 侧分配

`aiter/ops/opus/gemm_op_a16w16.py` 新增：
```python
def _alloc_splitk_workspace(kid, batch, M, N, K, split_k, device):
    nbytes = _common.kid_workspace_bytes(kid, batch, M, N, K, split_k)
    if nbytes == 0:
        return None
    return torch.empty(nbytes, dtype=torch.uint8, device=device)
```
用 `uint8` 扁平 buffer 而不是 `[batch, split_k, padded_M, padded_N]` 的 typed tensor：kernel 反正是按 `void*` + 自算 stride 访问的，typed shape 只会诱导读者以为 stride 有语义。（PR #4246 用的是 typed tensor；这里做个小改进，代价是 debug 时少了一点自描述性 —— 如果 review 更看重可读性，用 typed tensor 也行，两者对 kernel 无差别。）

**HIP graph**：不需要任何特殊处理。torch caching allocator 在 capture 期间返回 replay-stable 的地址，PR #4246 也因此删掉了 `tuned_gemm` 里的 eager prewarm。

### 3.4 删除旧机制

- `opus_gemm.cu:414-593` 全部（`SplitkWsRegistry`、`opus_splitk_ws_get`、`opus_splitk_ws_ensure_capacity`、`opus_gemm_workspace_init`，以及 ~70 行注释掉的历史代码）
- `opus_gemm.h:45` 的 `opus_gemm_workspace_init` 声明
- `rocm_ops.hpp:317 OPUS_GEMM_WORKSPACE_INIT_PYBIND`
- `opus_gemm_pybind.cu:19` 的调用
- `aiter/ops/opus/gemm_op_a16w16.py:510` 的 python binding 与 `__init__.py` 导出
- codegen 里 `gen_instances_gfx950.py:1425-1449` / `gen_instances_gfx942.py:500-520` 的 extern 声明 + capture 探测块

`opus_gemm_workspace_init` 是**公开 API 的破坏性删除**。当前仓内无调用点（已 grep 确认），但下游 vLLM/sglang 集成可能有。处理：保留一个 python 空壳 + `DeprecationWarning`，一个 release 后删；C++ 侧直接删。

**净删除量估算**：C++ ~230 行，codegen ~45 行。

### 3.5 任务一验收

| 项 | 要求 |
|---|---|
| 机械检查 | 全仓 grep `ws_handle` / `opus_splitk_ws_` / `SplitkWsRegistry` 零命中 |
| 数值 | `python3 op_tests/test_opus_a16w16_gemm.py` 全绿（含 gfx950 + gfx942 两 arch） |
| **CUDA graph** | 新增 capture/replay 用例，golden 对拍。这是原机制存在的全部理由，必须证明新机制也成立 |
| **双 stream 并发** | 新增用例，覆盖原 TBO 场景（两个 CPU 线程各驱动一个 stream 同时跑 splitK） |
| **ws dtype 覆盖** | 单测必须同时覆盖 fp32-ws kid 与 bf16-ws kid（gfx942 的 10210/10211/10213）—— §3.1 那个坑只在 bf16-ws kid 上暴露 |
| 性能 | splitK shape sweep 前后对比，不应回退（少一次 device 端 load，理论上略好） |

前两项之外的四项**都不能省**，尤其是 CUDA graph 和 ws dtype 覆盖。

---

## 4. 任务二 — 重构 opus gemm 的各种入口

四个阶段。§4.2 是核心（R1 + R3 必须一起做，拆不开：删掉 C++ dispatch 的同时必须在 python 补上，否则中间态不可用），§4.3 / §4.4 可并行，§4.1 是纯 plumbing 前置。

| 阶段 | 内容 | 覆盖需求 |
|---|---|---|
| 4.1 | kid 元信息去重（无行为变化） | — （前置） |
| 4.2 | dispatch 下沉到 python + 入口收敛 | R1, R2, R3 |
| 4.3 | dtype / layout 整理 | R4 |
| 4.4 | gfx950 / gfx942 遗留差异清理 | R2 |

---

### 4.1 kid 元信息去重（无行为变化，纯 plumbing）

**动机**：后面每步都要用 kid 元信息，先把 4 份副本收成 1 份。任务一的 §3.1 已经开了个头（加了两个 workspace 相关函数），这里补齐其余的。

**改动**：

1. `csrc/opus_gemm/opus_gemm_common.py` 补全公开查询 API（数据已存在，只是没有统一出口）：
   ```python
   def kid_info(kid: int) -> OpusGemmInstance | None
   def kid_is_splitk(kid) -> bool          # 取代 4 处硬编码区间判断
   def kid_supports_bias(kid) -> bool      # 取代 opus_gemm.cu:347 + kid_rejects_bias()
   def kid_arch(kid) -> str                # "gfx950" / "gfx942" / "gfx1250"
   def kid_exceeds_4g(kid, M, N, K) -> bool
   # kid_workspace_dtype / kid_workspace_bytes 已在任务一中加入
   ```

2. 删除副本：
   - `aiter/ops/opus/gemm_op_a16w16.py:274-275` 的 `_SPLITK_KID_MIN/MAX`
   - `aiter/ops/opus/common.py:197-203` 的 `_MONO_TILE_KID_TILES`（改为从 `opus_gemm_common` 推导）
   - `csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh:100-112` 的 `kSplitkKidMin/Max` + `kid_is_splitk`
   - `csrc/opus_gemm/opus_gemm.cu:289-355` 的全部 kid 分类常量与谓词

   后两条的删除前提是 §4.2 完成（那时 C++ 不再需要知道 kid 语义）；本阶段先加 python API 并把 python 副本切过去，C++ 副本标 `// TODO: remove after dispatch moves to python`。

3. 4 GiB buffer-resource 守卫：`opus_gemm_arch_gfx950.cuh:196/233` 与 `opus_gemm_tune.py:225-243` 两处实现合并为 `kid_exceeds_4g()`。

**验证**：两个 op_test 全绿；`gen_instances.py` 产物 diff 为空（这一步不应改变任何生成代码）。

---

### 4.2 dispatch 下沉到 python（R3）+ 入口收敛（R1）

#### a. heuristic 搬到 python

新建 `aiter/ops/opus/heuristic.py`，把 `opus_a16w16_heuristic_kid_gfx950()`（`opus_gemm_heuristic_dispatch_gfx950.cuh:83`，127 行）逐条翻成 python。这个函数已经是"纯 `(M,N,K,has_bias) → int kid`"的形态，翻译是机械的。

gfx942 那个（`opus_gemm_heuristic_dispatch_gfx942.cuh`）麻烦一些：它**直接返回 launcher 符号名**而非 kid（比 gfx950 落后一代）。翻译前先把它改写成 kid 形态 —— 符号名到 kid 的映射从 `opus_gemm_common.py` 的 `gfx942_*_kernels_list` 反查即可，机械可验证。

统一后：
```python
# aiter/ops/opus/heuristic.py
def heuristic_kid(arch: str, M, N, K, *, has_bias=False, out_dtype=None) -> int:
    return _HEURISTIC_BY_ARCH[arch](M, N, K, has_bias)
```

#### b. 统一 dispatch 函数

```python
# aiter/ops/opus/dispatch.py（新建，a16w16 / a8w8 共用）
def resolve_kid(family, M, N, K, *, bias, dtype, outdtype, arch) -> tuple[int, int]:
    """返回 (kid, splitK)。三级：CSV → heuristic → 抛错。"""
    cfg = common.lookup_tuned(M, N, K, bias=bias, dtype=dtype, outdtype=outdtype, ...)
    if cfg is not None and not common.kid_rejects_shape(cfg["solidx"], M, N, K):
        return cfg["solidx"], cfg["splitK"]
    return heuristic.heuristic_kid(arch, M, N, K, has_bias=bias), 0
```
注意这里顺手修掉了一个现存缺陷：`gemm_op_a16w16.py:471` 命中 CSV 后**无条件信任** solidx，只对 mono-tile 做了 `mono_kid_shape_ok` 特判（`gemm_op_a16w16.py:209`）。统一走 `kid_rejects_shape` 后，任何 kid 的 shape 不匹配都会优雅降级到 heuristic，而不是靠一堆 kid-family 特判打补丁。

#### c. C++ 侧删除

- `opus_gemm_lookup.h` 整个文件 + `gen_instances.py:413 gen_lookup_dict()`（~110 行）+ `gen_instances.py:1114-1166` 的 CSV 烘焙块（~55 行）+ `get_tune_dict()`（~40 行）
- `opus_gemm_heuristic_dispatch_gfx950.cuh` 的 `opus_a16w16_heuristic_kid_gfx950()`（保留文件，因为 `OpusA16W16NoscaleKernel` typedef 还在用 —— 应该趁机把 typedef 挪到 `opus_gemm_common.cuh`，然后整个文件删掉）
- `opus_gemm_heuristic_dispatch_gfx942.cuh` 整个文件
- `opus_gemm_arch_gfx950.cuh:169-248` 的 `opus_dispatch_a16w16_gfx950<>`（两个特化，~80 行）；gfx942 同理
- `opus_gemm.cu:88-134` 的 arch router `opus_dispatch_a16w16<>`
- `opus_gemm.cu:138-207` 的 `opus_gemm()` bf16 分支

**重要副作用（务必确认）**：`gen_instances.py:1028` 的 subset-compile 集合是
`S = (csv_kids | sidecar_kids | HEURISTIC_DEFAULT_KIDS) & valid_kids`。
删掉 C++ lookup 后 **`csv_kids` 这一项仍必须保留** —— 它决定的是"编译哪些 kid 进 .so"，不是"运行时怎么查表"。这两件事被现有代码耦合在同一段逻辑里，拆的时候容易连坐删掉，导致 CSV 里的 kid 编不进 .so、运行时 `AITER_CHECK(kid not found)`。`HEURISTIC_DEFAULT_KIDS` 同理保留（现在它保证的不变式从"C++ heuristic 返回的 kid 一定在 S 里"变成"python heuristic 返回的 kid 一定在 S 里"，需要在 `heuristic.py` 里加一个 import-time 自检断言这个 frozenset 与 python heuristic 的可能返回值集合一致）。

#### d. `opus_gemm()` 入口的处置

删掉 bf16 分支后，`opus_gemm()` 只剩 fp8 分支（且那里也是硬编码 tile 选择）。选项：

- **推荐**：`opus_gemm()` 整个删掉，fp8 路径按同样的模式改造成 `opus_gemm_a8w8_tune(XQ, WQ, Y, x_scale, w_scale, kid)`，python 侧 `gemm_a8w8_opus()` 负责选 kid（现在 C++ 里那点 `M%256`/`M%128` 逻辑照搬到 python 即可）。
- 保守：保留 `opus_gemm()` 作为 deprecated 薄壳，只转发 fp8。

推荐前者：R1 要的就是"入口收敛"，留一个只服务半个 dtype 的万能入口是反效果。

#### e. 最终 C++ 入口清单（5 → 3）

```cpp
void opus_gemm_a16w16_tune(XQ, WQ, Y, bias, kernelId, splitK, workspace);
void opus_gemm_a8w8_tune  (XQ, WQ, Y, x_scale, w_scale, kernelId);   // 合并 scale / noscale / bpreshuffle
void opus_gemm_a8w8_bpreshuffle_tune(...);  // 若 B 的 preshuffle 布局无法与上者共用 kargs，则保留；
                                            // 能共用就并入 opus_gemm_a8w8_tune，靠 kid 区分
```
每个都是纯 `kid → 函数指针 → launch`。`opus_gemm.cu` 从 595 行降到 **~120 行**。

---

### 4.3 dtype / layout 整理（R4）

#### a. C++ 侧：表驱动取代 if-else 树

`opus_gemm.cu:379-411` 的嵌套分支替换为一张显式表：

```cpp
// (kid_family, XQ.dtype, Y.dtype) -> 模板实参 CDataType
// splitk kid 的 main kernel 硬编码 D_C=float（traits static_assert），
// Y 的实际 dtype 由 reduce kernel 在 launch 时决定 —— 这是唯一的非对角项，
// 值得一行注释而不是散在三个 if 里。
static constexpr OpusDtypeRule kA16W16Rules[] = {
    // in        out        needs_ws  ctype
    {bf16,       bf16,      false,    CType::BF16},
    {bf16,       fp32,      false,    CType::FP32},
    {bf16,       bf16,      true,     CType::FP32},   // splitk: ws=fp32, reduce 出 bf16
    {bf16,       fp32,      true,     CType::FP32},
};
```
配一个 `resolve_ctype(rules, in, out, needs_ws)`，命中失败给一条包含全部三元组的错误信息。a8w8 同构（`{fp8, fp32, has_scale}`）。

#### b. Python 侧：抽公共校验层

`gemm_op_a16w16.py::_validate_and_reshape`（124 行）与 `_check_a16w16_tune_layout`（60 行）语义大量重叠（前者面向 2D/3D 用户输入，后者面向已 reshape 的 3D）；`gemm_op_a8w8_bpreshuffle.py::_reshape_3d` 是第三份近似实现。

抽 `aiter/ops/opus/layout.py`：
```python
@dataclass(frozen=True)
class GemmLayout:
    XQ: Tensor; WQ: Tensor; Y: Tensor
    batch: int; M: int; N: int; K: int
    squeeze_out: bool

def normalize_gemm_layout(A, B, *, out=None, out_dtype, bias=None,
                          in_dtypes, out_dtypes) -> GemmLayout
```
一处实现三条规则（这三条是所有 opus launcher 共有的硬约束，源头是 codegen 里硬编码的 `stride_*_batch`）：
1. A/B/Y 必须是 contiguous 3D，stride 严格等于 `(M*K, K, 1)` / `(N*K, K, 1)` / `(M*N, N, 1)`
2. B 的 batch broadcast view（`stride(0)==0`）必须拒绝
3. bias 是 `[N]` 或 `[batch, N]`，dtype == out dtype

现有的长篇错误信息（写得很好，解释了 *为什么* 拒绝）**原样保留**，只是搬到一处。

#### c. codegen 侧

`gen_instances*.py` 里的 `BIAS_HOST_VALIDATE`（`gen_instances.py:838` 附近的模板）在 python 已经校验过之后是冗余的，但它是**最后一道保险**（tuner 和 C++ 直调不走 python 校验）。保留，但收敛成一个 codegen 共享片段而不是每个 emit 函数各贴一份。

---

### 4.4 gfx950 / gfx942 遗留差异清理（R2）

严格说 §4.2 做完 gfx950 就已经切到新机制上了。这里单列是为了强调**遗留差异清理**：

| 差异项 | gfx950 | gfx942 | 处理 |
|---|---|---|---|
| heuristic 返回值 | int kid（好） | launcher 符号名（差） | gfx942 改成 kid，见 §4.2a |
| reduce kernel 名 | `splitk_reduce_kernel` | `splitk_reduce_kernel_fallback` | 保留（实现确实不同），但 ABI 统一为 `void* ptr_ws` |
| workspace dtype | 恒 fp32 | 有 `bf16ws` 变体（`splitk_workspace_dtype` 属性） | 统一走 `kid_workspace_dtype()`；gfx950 未来加 bf16 ws 时零成本 |
| 4g_safe kid | 有（+5000 band） | 无 | 保持现状 |
| `SPLITK_REDUCE_ABI_MAP` | 两 arch 各一份 | | 任务一做完后两份内容除 kernel 名外完全相同，合并为共享默认 + per-arch override |
| `opus_gemm_arch_gfx950.cuh` 承载的 typedef | `OpusA16W16NoscaleKernel` 定义在 gfx950 的 heuristic 头里，被 gfx942 复用 | | 挪到 `opus_gemm_common.cuh`，解开这个反向依赖 |

另外为 gfx1250 预留：`OpusGfxArch` enum（`opus_gemm_arch.cuh:13`）加 `Gfx1250`；`aiter/ops/opus/__init__.py:13` 的 `_SUPPORTED = {"gfx950", "gfx942"}` 加 `"gfx1250"`。PR #4246 的 fuse kid band 是 `[20000, 30000)`，与现有 band（gfx950 `[0,10000)`、gfx942 `[10000,20000)`）不冲突，直接沿用。

### 4.5 任务二验收

| 项 | 要求 |
|---|---|
| 结构 | `opus_gemm.cu` ≤ 150 行；文件内不出现 CSV / heuristic / hipMalloc / shape 策略 |
| 入口数 | C++ 顶层入口 5 → 3，全部形如 `(tensors..., kernelId[, splitK][, workspace])` |
| 副本数 | kid 语义常量全仓仅 `opus_gemm_common.py` 一份（grep `SPLITK_KID_MIN` / `kSplitkKidMin` 零命中）；`opus_gemm_lookup.h` 不存在 |
| **kid 选择等价性** | 重构前后对同一 shape sweep（两 arch × 全 CSV 行 + 一批 CSV-miss shape）逐条对比选出的 (kid, splitK)，要求**完全相同**。这是任务二唯一能证明"没改行为"的手段，必须写成脚本而不是抽查 |
| 编译集合 | 对每个 CSV opus 行断言 `solidx ∈ S`（subset-compile 集合），防止 §4.2c 那个连坐删除 |
| 数值 | `op_tests/test_opus_a16w16_gemm.py` + a8w8 bpreshuffle 测试全绿 |
| 性能 | 小 shape（M≤64）上对比端到端 host 开销，确认 dispatch 下沉没有引入可观测的 CPU 回退 |

---

## 5. Bonus — 新 kernel 接入 skill（B1）

**目标**（用户原话）：减少 agent 吐冗余代码和 comment，维持代码结构扁平简洁。

仓内已有三个 opus skill：`opus-kernel-authoring`（写 kernel 本身，`.claude/skills/`）、`opus-kernel-best-practice`（编译期优化）、`opus-module-build-optimization`（build wall）。**缺的正是"kernel 写完之后，怎么接进 aiter"这一段** —— 而这恰恰是 agent 最容易吐垃圾的地方（要同时改 6~8 个文件，每处都有既定 pattern）。

新建 `.claude/skills/opus-kernel-integration/SKILL.md`，内容骨架：

1. **接入清单（checklist，按顺序）** —— 两个任务完成后的稳定版本：
   ```
   1. csrc/opus_gemm/include/<arch>/opus_gemm_pipeline_<name>.cuh   新 pipeline
   2. csrc/opus_gemm/include/<arch>/opus_gemm_traits_*.cuh          新 traits + kargs
   3. csrc/opus_gemm/opus_gemm_common.py                            新 kid（唯一的元信息登记处）
   4. csrc/opus_gemm/codegen/gen_instances_<arch>.py                emit 函数 + register_emit
   5. aiter/ops/opus/heuristic.py                                   （仅当要进 heuristic 兜底）
   6. op_tests/opus/test_<name>.py                                  golden 对拍
   ```
   明确"**不需要改**"的文件（这是减少冗余输出的关键）：`opus_gemm.cu`、`opus_gemm.h`、`rocm_ops.hpp`、`opus_gemm_pybind.cu`、任何 python 入口 —— 重构后新 kid 走既有 tune 入口，零 C++ 入口改动。

2. **kid band 分配表** —— 明确写死，避免 agent 自己发明编号：
   ```
   gfx950  [0, 10000)      splitk [200,300) / persistent [300,316) / mono [1400,1500)
                           nooob 镜像 +1000, cpol +2000/3000/4000, 4g_safe +5000
   gfx942  [10000, 20000)
   gfx1250 [20000, 30000)  splitk-fuse
   ```

3. **注释预算（硬约束）** —— 复用 `opus-kernel-authoring` 的 Rule 2（关键逻辑 ≤2-3 行注释），并追加接入层的约束：
   - codegen emit 函数：只在偏离既有 pattern 处注释，不重复解释 pattern 本身
   - **禁止**：注释掉的旧代码（现状反例：`opus_gemm.cu:455-542` 有 ~70 行注释掉的历史实现）
   - **禁止**：AI 味的 changelog 式注释（"Previously this was X, now it's Y because…"）—— 这类信息属于 commit message。现状反例：`opus_gemm.cu:414-429` 15 行的 rationale 注释块
   - 新增文件头注释 ≤10 行

4. **反模式清单**（从这次重构里提炼，每条都指向仓内真实位置）：
   - 在 C++ 里加 shape → kernel 的策略判断（应该去 python）
   - 在 C++ 里 `hipMalloc`（应该 torch 分配后传入）
   - 复制 kid 区间常量到新文件（应该 import `opus_gemm_common`）
   - 为新 kernel 加新的顶层 pybind 入口（应该复用 `*_tune`）

5. **验收命令**：`python3 op_tests/opus/test_<name>.py` + `ruff check` + `clang-format-18`。

Skill 应在**任务二完成之后**写 —— 写在重构前只会固化现在这套要被删掉的 pattern。

---

## 6. 风险与缓解

### 任务一的风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| **workspace 尺寸算错**（PR #4246 踩过） | 越界写、机器挂死 | `kid_workspace_bytes` 必须读 kid 自己的 ws dtype（§3.1）；单测覆盖 fp32-ws 与 bf16-ws 两类 kid；C++ 侧保留一次 `AITER_CHECK` 尺寸比对（§3.3） |
| **split_k clamp 导致 python 与 kernel 不一致** | 分配偏小 → 越界 | 按未 clamp 的上界分配（§3.1 方案 b）；clamp 只会让 split_k 变小，上界恒安全 |
| **ABI 变更漏改某个 arch 的 launcher** | 编译错，或更糟：静默用错地址 | ABI 集中在 `SPLITK_REDUCE_ABI_MAP`；改完全仓 grep `ws_handle` 必须零命中 |
| **HIP graph 行为变化** | capture/replay 出错 | 验收强制包含 graph 用例；torch caching allocator 的 capture-aware 行为是 PR #4246 已验证的前提 |
| **`opus_gemm_workspace_init` 是下游公开 API** | 下游 vLLM/sglang 集成崩 | python 留空壳 + `DeprecationWarning` 一个 release，C++ 侧直接删 |

### 任务二的风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| **subset-compile 集合 S 被连坐删除** | CSV 里的 kid 编不进 .so，运行时报 "kid not found" | 见 §4.2c 加粗段。加 CI 检查：对每个 CSV opus 行断言 `solidx ∈ S` |
| **`HEURISTIC_DEFAULT_KIDS` 与 python heuristic 脱钩** | heuristic 返回未编译的 kid | `heuristic.py` import 时自检：枚举所有可能返回值 ⊆ `HEURISTIC_DEFAULT_KIDS` |
| **gfx942 heuristic 符号名→kid 翻译出错** | 选错 kernel，结果仍对但性能回退（最难发现的一类） | 翻译用 `opus_gemm_common.py` 的 kernels_list 反查**生成**，不手写；跑 §4.5 的 kid 选择等价性对比，要求逐条相同 |
| **dispatch 下沉后每次调用多一次 python 开销** | 小 shape 上 CPU-bound 可能吃亏 | `lookup_tuned` 已经 `lru_cache`；给 `resolve_kid` 加 `functools.cache`（key=`(family,M,N,K,bias,dtypes,arch)`），稳态是一次 dict 查找。基线：现状 CSV 命中路径**本来就是** python 查表，只有 miss 路径会从 C++ 变 python |
| **删 `opus_gemm()` 是 breaking change** | 下游直调 pybind 的会崩 | 见 §8 待确认问题 2；若确认有下游，退回"保留 deprecated 薄壳" |

---

## 7. 交付顺序与规模估算

**任务一（建议先做，可独立合入）**

| 步骤 | 内容 | 净代码变化 |
|---|---|---|
| §3.1 | kid workspace 元信息（2 个函数） | +40 |
| §3.2–3.4 | ABI 改 `void*` + python 分配 + 删 registry | +40 / −275 |

小计约 **−195 行**，一个 PR 的体量。

**任务二**

| 步骤 | 内容 | 净代码变化 | 依赖 |
|---|---|---|---|
| §4.1 | kid 元信息 SSOT | +80 / −60 | 任务一（复用其已加的两个函数） |
| §4.2 | dispatch 下沉 + 入口收敛 | +250 / −700 | §4.1 |
| §4.3 | dtype / layout 整理 | +150 / −280 | §4.2 |
| §4.4 | gfx950/942 差异清理 | +40 / −120 | §4.2 |

小计约 **−640 行**。§4.3 与 §4.4 可并行、可拆成独立 PR。

**Bonus** §5：+200 行文档，在任务二之后。

**两任务合计约 −835 行**，`opus_gemm.cu` 595 → ~120 行，C++ 入口 5 → 3 个，bf16 dispatch 路径 3 → 1 条。

---

## 8. 待确认问题

**影响任务一：**

1. **"或者 zoe 那个也可"** —— 除 PR #4246 外还有一个参考实现，我没有找到对应的 PR/分支。如果 zoe 那版在 workspace 所有权上有不同取舍（比如走 `torch.empty` 之外的 aiter 内部 buffer pool），需要先看一眼再定 §3.3 的具体形态。**只影响实现细节，不影响分层架构。**
2. **workspace 参数是加在现有 `opus_gemm_a16w16_tune` 尾部（默认 `std::nullopt`），还是另开入口** —— 前者对现有 5 参调用点（`tuned_gemm.py:537`、`op_tests/*`）零改动，本文按前者写。

**影响任务二：**

3. **`opus_gemm()` 万能入口是删还是留薄壳**（§4.2d）—— 取决于下游有没有直接调 pybind 的 `opus_gemm`。
4. **a8w8 三个入口能否合并成一个**（§4.2e）—— 取决于 bpreshuffle 的 B 布局能否与普通 a8w8 共用同一套 kargs；需要读一遍 `gen_scale_bpreshuffle_instance`（`gen_instances_gfx950.py:794`）确认。
5. **gfx1250 是这次重构的目标 arch 之一，还是只是预留** —— 影响 §4.4 要不要把 PR #4246 的 fuse pipeline 一并 rebase 进来。
