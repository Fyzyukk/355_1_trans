# OPUS GEMM Split-K 与接口重构设计

> 状态：设计提案，尚未实施；已合并第三轮方案评审与上游 PR 实证
>
> 基线：当前工作区 `35696023`
>
> 上游参考（截至 2026-08-10 均为 open）：
>
> - [ROCm/aiter PR #4246 - gfx1250 opus gemm splitk fuse](https://github.com/ROCm/aiter/pull/4246)：任务 A 的直接参考。workspace 所有权切换始于 commit `4ac1fa86ea452a4289aa0751aa07e8d306e3f3ed`，按 kid 的实际 dtype/shape 修正容量并删除 prewarm 的版本为 `86d1eba4c2b9de9649fe36228b3b8bba578ab0bd`。
> - [ROCm/aiter PR #4320 - Add opus fp8 mxscale BMM kernels for gfx950](https://github.com/ROCm/aiter/pull/4320)：需求中“Zoe 那个”，任务 B 的分层参考；本文核验的 head 为 `ec93aadf2596052eac7cb6f3ddc01993fb1faa4c`。
>
> 两个 PR 解决的问题不同。#4246 还包含可选 fused pipeline；#4320 的 Python 选核/C++ kid dispatch 值得复用，但其生成 launcher 仍有 raw HIP split-K workspace grow，不能作为任务 A 的完成形态。

## 1. 任务拆分

这项工作应明确拆成两个必做任务和一个可选 kernel 任务。它们共享 kernel metadata 和低层 launcher ABI，但交付物、风险和验收标准不同，不应放在一个大 PR 中完成。

### 任务 A：现有 split-K workspace Torch 化

任务 A 只改变 workspace 的所有权和传递方式：

- gfx942、gfx950、gfx1250 的现有 split-K workspace 统一由 Python 使用 typed `torch.empty` 分配。
- C++ launcher接收 workspace Tensor并把直接指针写入 kargs。
- 删除 raw HIP allocator、per-stream registry、host/device双 handle和 graph prewarm。
- 不修改现有 kernel 数值流水线，不在本任务新增 fused kernel。

#4246 只把 gfx1250 路径切成 Torch workspace；本任务要把同一所有权模型推广到当前树已有的 gfx942、gfx950、gfx1250 三套 a16w16 two-stage split-K。#4246 的 `_fuse` pipeline不在本任务内，且 fuse仍保留 partial workspace，只是省掉独立 reduce launch。

任务 A 不立即删除 C++ lookup/heuristic源码，但它依赖一个最小的 Python `LaunchConfig` 解析前置：Python必须在分配 workspace 前知道最终 kid。旧 C++ selector暂时只用于 differential test，不再作为 production兼容 fallback。

### 任务 B：重构 OPUS GEMM 接口

任务 B 将当前混合了 a16w16、a8w8、lookup、heuristic、workspace registry 和兼容逻辑的入口拆开：

- Python 负责输入规范化、tuned lookup、heuristic、split-K 解析、output/workspace 分配和 fallback。
- C++ 仅负责运行时安全检查、arch 校验、`kid -> launcher` dispatch、kargs 构造和 kernel launch。
- a16w16、a8w8、blockscale 等 kernel family 使用各自的低层 ABI，不再共用一个 mega entry。

#4320 已展示可复用的基本形态：Python做 heuristic和 tuned-kid shape gate，`opus_bmm.cu` 做公共安全检查及 kid table dispatch。本文复用这种职责边界，但不照搬其 unknown-kid默认回退、raw workspace grow或单文件 heuristic布局。

### 可选任务 C：gfx1250 fused split-K kernel

PR #4246 的标题是 fused split-K，但原始需求明确提到的是“workspace 开辟切换到 torch”。除非需求方另外确认要新增 fused kernel，否则 fused pipeline不属于任务 A/B 的必做范围，应单独立项。

fused split-K 消除的是独立 reduce kernel，不是 workspace：非最后 split仍需把 partial写入 `ptr_ws`，最后 split读取前 `SplitK - 1` 个 partial并完成融合规约。

### 公共前置和真实依赖

两个必做任务依赖同一份权威 `KernelSpec` 投影和一个 Python `resolve_launch_config()`。不能让任务 A 在 `gemm_op_a16w16.py` 增加 kid-range/workspace 特判，同时任务 B 又重新设计 metadata。

任务 A 也不能在所有路径上完全独立于选核逻辑：当前 CSV miss路径进入 C++ 后才通过 heuristic得到 kid；Python事前不知道 workspace dtype、tile和容量。完整迁移只能从以下两种方案中选择：

1. 推荐：先把 heuristic等价移植到 Python并保留旧 C++ selector做 golden对照；随后所有路径都转成显式 kid launch。
2. 不推荐：Python分配跨所有候选 kernel的最坏情况 byte buffer。该方案浪费显存、丢失 dtype校验，而且会固化临时 ABI。

建议依赖关系如下：

```text
PR0：authoritative KernelSpec query + Python selector（旧 C++ selector仍保留）
                              │
                              ▼
任务 A：workspace Torch 化、direct ptr、删除 registry/prewarm
                              │
                              ▼
任务 B：删除 C++ lookup/heuristic，拆分 family API
                              │
                              └────► 可选任务 C：gfx1250 fused split-K
```

## 2. 当前架构和主要问题

当前 a16w16 调用链是：

```text
gemm_a16w16_opus()
  ├─ Python tuned CSV lookup
  │    └─ hit -> opus_gemm_a16w16_tune(kid, splitK)
  └─ miss -> C++ opus_gemm()
              ├─ C++ baked (M,N,K) lookup
              ├─ C++ heuristic
              └─ kid launcher
                    └─ C++ workspace registry/hipMalloc
```

由此产生以下问题：

- Python 和 C++ 都拥有 tuned lookup/heuristic，选择可能漂移。
- `opus_gemm()` 同时根据 dtype 路由 a16w16/a8w8，又携带 scale、group layout 和 bias，接口边界不清晰。
- `opus_gemm_a16w16_tune()` 通过 kid 数值区间判断 split-K、bias capability 和 output dtype 分支。
- gfx942、gfx950、gfx1250 各自的生成 launcher 管理 workspace，存在 `hipMalloc`、handle 和 graph prewarm。
- runtime lookup 表和真正需要保留的 kid dispatch 表都由 `gen_instances.py` 生成，语义混在一起。
- split-K 的 workspace dtype、accumulation dtype 和最终输出 dtype 没有被独立表达。

### 2.1 两张生成表不能混淆

| 生成物 | key | 运行时职责 | 最终处置 |
|---|---|---|---|
| `opus_gemm_lookup.h` | `(M,N,K)` | C++ runtime shape lookup | 删除 |
| `opus_gemm_a16w16_tune_lookup.h` | `kid` | C++ kid -> launcher dispatch | 保留并改名为 `opus_gemm_a16w16_kid_dispatch.h` |

删除第一张表时不能连带删除 CSV kid集合：build-time仍需从 tuned CSV提取 `csv_kids`，保证 runtime可能选中的 kid被编进 `.so`。

### 2.2 当前 workspace 机制

当前 `SplitkWsRegistry` 以 `hipStream_t` 为 key，维护 host pinned handle和 device mirror，并通过手工同步保持二者一致。launcher grow路径还包含 4 MiB对齐、设备同步和 raw HIP allocation。由此引出了 `opus_gemm_workspace_init()` 以及 `tuned_gemm.py` 中针对 graph capture stream的 prewarm补丁。

现有 two-stage split-K 的容量语义为：

| arch | 元素数量 | 元素 dtype | batch语义 |
|---|---|---|---|
| gfx950 | `split_k * batch * padded_M * padded_N` | fp32 | per-batch workspace |
| gfx942 | `split_k * batch * padded_M * padded_N` | kid决定，存在 bf16 workspace | per-batch workspace |
| gfx1250 | `split_k * padded_M * padded_N` | fp32 | 当前 pipeline只支持 `batch == 1` |

gfx1250 不能被建模成永久的 `ws_includes_batch=False`；正确事实是当前 kernel的 `batch_mode="single_only"`，对应 workspace的 `batch_storage="single"`。未来若支持逐 batch顺序复用或并行 batch，应分别声明 `host_serial/shared_serial` 或 `grid_parallel/per_batch`，而不是只改一个乘数。

## 3. 目标架构

### 3.1 控制流

```text
Public Python API
  gemm_a16w16_opus(A, B, ...)
          │
          ▼
normalize_and_validate()
          │
          ▼
select_launch_config()
  explicit kid > tuned lookup > Python heuristic > fallback
          │
          ▼
KernelSpec + LaunchConfig
          │
          ├─ torch.empty(out)
          └─ torch.empty(workspace) / caller workspace
          │
          ▼
_opus_gemm_a16w16_launch(A, B, out, bias, workspace, kid, split_k)
          │
          ▼
C++: arch + kid + output dtype -> generated launcher
          │
          ▼
main kernel [-> separate reduce kernel]
```

### 3.2 职责边界

| 能力 | Python | C++ |
|---|---:|---:|
| shape/device/dtype/layout 规范化 | 是 | 只做安全复核 |
| tuned CSV lookup | 是 | 否 |
| heuristic | 是 | 否 |
| explicit/tuned/heuristic 优先级 | 是 | 否 |
| effective split-K 解析 | 是 | 否 |
| output/workspace 分配 | 是 | 否 |
| arch 与已编译 kid 校验 | 是 | 是 |
| kid -> function dispatch | 否 | 是 |
| kernel 参数构造和 launch | 否 | 是 |
| raw HIP workspace allocator/registry | 否 | 否 |

C++ 中仍保留防止非法 launch 或越界的检查；“C++ 只做 dispatch”不代表移除安全检查。

### 3.3 #4320 可复用的模式与不可照搬的部分

#4320 新增的 mxscale BMM 与本方案目标的对照如下：

| #4320 已实现 | 本方案如何使用 |
|---|---|
| `bmm_op.py` 在 Python执行 heuristic | 作为任务 B 将三架构 GEMM heuristic迁出 C++ 的直接结构参考。 |
| Python从 `csrc.opus_gemm.opus_gemm_common` 读取 `m_align`，不再手写 kid allowlist | 证明安装/JIT形态下可以复用现有 kernel metadata；本方案进一步提供稳定、只读的 `KernelSpec` 查询层。 |
| tuned kid不满足 `m_align` 时将 `kernelId` 和 `splitK` 一起置空，再走 heuristic | 原样吸收“tuned配置原子失效”规则；不能把旧 tuned split-K带给新 heuristic kid。 |
| `opus_bmm.cu` 做 dtype/arch安全检查和一次 kid table lookup | 作为 family-specific C++ launch入口的参考。 |
| generated kid -> launcher map | 与必须保留的 `opus_gemm_a16w16_tune_lookup.h` 同类；后续统一使用 `*_kid_dispatch` 命名。 |

以下细节不能照搬：

- #4320 的 `opus_bmm_a8w8_mxscale_tune_dispatch()` 对 unknown kid静默回退到默认 kid。显式 kid是调试、tuner和可复现性接口，目标 GEMM/BMM launch binding都应严格报错；只有 Python selector可以有可观测的 fallback。
- #4320 的顶层 `opus_bmm.cu` 没有 workspace参数，不代表整条路径没有 allocator。其 `gen_instances_gfx950.py` 生成的 mxscale BMM split-K/fused launcher仍调用 `opus_splitk_ws_get()`、`hipMalloc()`、`hipFree()`，并在 graph capture时禁止 grow。因此 #4320 不是任务 A 的 workspace参考。
- #4320 的零参数 `functools.cache` 只缓存一份静态 `kid -> m_align` 字典，不等于允许对动态 selector结果或 Tensor workspace使用无界 cache。
- #4320 只有一个 gfx950 BMM family，把 heuristic放在单个 op文件尚可；三架构 a16w16仍应拆成共享 selector加 per-arch heuristic模块。

如果 #4320 在任务 A 前合入，必须明确选择范围：要维持“OPUS 中无 raw workspace allocator”的 Definition of Done，就要把它新增的 gfx950 mxscale BMM split-K/fused launcher也纳入 Torch workspace迁移；若本期只承诺当前 a16w16，必须在 PR描述和检查命令中显式排除 BMM，不能误报全模块已经清理完毕。

## 4. 公共数据模型

### 4.1 权威事实源与稳定 runtime 查询层

不再新增一份承载 kernel数据的 `aiter/ops/opus/_kernel_catalog.py`。本树已经把所有实例登记在 `csrc/opus_gemm/opus_gemm_common.py`，codegen和 tuner都直接消费它；把几百个 kid迁到 runtime目录只会制造一次高风险搬家，迁移期间还会出现两个权威源。

运行时读取 `csrc` 在本项目中是受支持的安装契约，而不是仅在源码 checkout中碰巧可用：

- `aiter/__init__.py` 在导入 OPUS 前先导入 `aiter.jit.core`；`core.py` 将 `AITER_META_DIR` 放入 `sys.path`。
- editable/source模式的 `AITER_META_DIR` 是仓库根；非 editable安装时 `setup.py:prepare_packaging()` 将整个 `csrc` 复制到 `aiter_meta/csrc`，`MANIFEST.in` 包含 `aiter_meta`。
- 仓内 attention、sampling和 AOT模块已经使用 `from csrc...`；#4320 的 `bmm_op.py` 也用同一路径读取 `opus_gemm_common.py`。

因此采用“一个权威源、两个接口面”：

1. `csrc/opus_gemm/opus_gemm_common.py` 继续保存 codegen authoring record和全部实例登记，并新增稳定查询函数 `get_kernel_spec(arch, family, kid)`、`iter_kernel_specs(...)` 与 `catalog_fingerprint()`。
2. 现有 `OpusGemmInstance` 是可变的 codegen对象，且部分 family会在构造后附加属性，不能直接暴露给 runtime。查询函数必须返回下面的 frozen `KernelSpec` 投影；构建投影时若 `D_ACC/D_WS/D_OUT` 或 capability含糊，应立即失败，不能靠默认值猜。
3. 新增 `aiter/ops/opus/_catalog.py` 作为很薄的 runtime facade。它先显式导入 `aiter.jit.core` 以建立 `AITER_META_DIR`，再导入上述查询函数；facade不保存或重写任何 kid数据，也不泄露可变 instance dict。
4. codegen和 tuner继续直接导入 `opus_gemm_common.py`，不反向 import `aiter`，因此无 GPU的生成流程不会触发包级 arch/JIT副作用。

CI必须分别验证 source、editable install和 wheel install三种形态：在不加载 OPUS extension、无需 GPU的子进程中导入 `_catalog.py`，读取同一个代表性 spec并比较 fingerprint。禁止各模块自行修改 `sys.path`；路径建立只允许由 `aiter.jit.core` 集中完成。

任务 A 只消费 `KernelSpec` 中与 workspace、tile、split-K和batch有关的字段；任务 B 再消费完整 capability。不要为任务 A 另建一份简化 kid表，否则任务 B会再次迁移同一批事实。

```python
@dataclass(frozen=True)
class WorkspaceSpec:
    dtype: str
    layout: str                 # split_major_padded / tile_major_fused
    batch_storage: str          # single / per_batch / shared_serial
    alignment: int


@dataclass(frozen=True)
class ScaleSpec:
    mode: str                   # per_tensor / block
    a_dtype: str
    b_dtype: str
    block_shape: tuple[int, ...] | None
    b_layout: str               # plain / preshuffled


@dataclass(frozen=True)
class SplitKSpec:
    mode: str                   # none / runtime / static
    static_value: int | None = None
    max_splits: int = 1
    min_k_iters_per_split: int = 1
    require_even_k_iters: bool = False
    auto_target_wg_per_cu: int | None = None


@dataclass(frozen=True)
class KernelSpec:
    arch: str
    family: str
    kid: int
    kernel_tag: str
    launcher_symbol: str

    block_m: int
    block_n: int
    block_k: int

    a_dtypes: tuple[str, ...]
    b_dtypes: tuple[str, ...]
    output_dtypes: tuple[str, ...]
    acc_dtype: str
    workspace: WorkspaceSpec | None
    scales: ScaleSpec | None

    a_layout: str
    b_layout: str
    output_layout: str
    split_k: SplitKSpec
    launch_kind: str           # direct / two_stage / fused

    # bias存在时允许的精确 (output_dtype, bias_dtype) 组合；空 tuple表示不支持bias
    output_bias_dtype_pairs: tuple[tuple[str, str], ...]
    batch_mode: str             # single_only / grid_parallel / host_serial
    m_alignment: int            # 1表示任意M；比 supports_m_tail 布尔值更精确
    n_alignment: int
    k_alignment: int

    cluster_dims: tuple[int, int, int] | None = None
    cluster_axes: tuple[str, str, str] | None = None
    addressing_mode: str = "legacy_full_extent_buffer_resource"
```

这里的 dtype 使用稳定字符串枚举，例如 `bf16`、`fp32`，runtime 再映射到 `torch.dtype`，codegen 映射到 C++ 类型。不要让权威 metadata依赖 PyTorch初始化。a16w16的 `scales=None`；a8w8 no-scale和blockscale/preshuffle差异由 `ScaleSpec` 与 family共同表达，不再靠某个 optional scale是否为 `None` 猜 kernel family。

alignment必须记录 launcher真正执行的倍数，而不是 `supports_*_tail` 布尔近似。#4320 的 `m_align` 已证明某些 family要求 `2 * B_M`，只记录“没有 M tail”仍不足以判断 shape。bias也必须记录精确 `(output,bias)` pair，不能把两个 dtype列表做笛卡尔积后错误放宽 gfx950/gfx942或错误收紧 gfx1250。

`SplitKSpec` 足以表达当前三种 runtime clamp：gfx950的每 split最小 prefetch迭代数、gfx942的最小迭代数/偶数循环/按 CU自动补 TLP，以及 gfx1250的非空 split约束。Python和generated launcher必须调用同一个纯算法或由同一模板生成，不能分别手写三份近似逻辑。

`kid` 的逻辑身份是 `(arch, family, kid)`。数值区间只可以作为 ID 分配规则，不能再作为 runtime capability API。

4GiB能力不能压缩成 `kid_exceeds_4g(kid, M, N, K)`：实际合法性还与 tensor dtype、stride、batch launch方式和 addressing mode有关。应使用互斥的 `addressing_mode`（例如 `legacy_full_extent_buffer_resource`、`per_workgroup_buffer_resource`、`flat_64bit`）表达能力，再对实际 tensor的 storage span执行校验；不要再额外保存一个可能与 mode矛盾的 `is_4g_safe` 布尔副本。

### 4.2 动态选择结果

```python
@dataclass(frozen=True)
class LaunchConfig:
    spec: KernelSpec
    requested_split_k: int
    effective_split_k: int | None
    source: str                # explicit / tuned / heuristic
```

静态属性和动态选择结果必须分开：

- fused split-K 的静态值来自 `KernelSpec.split_k.static_value`。
- 任务 A 期间，two-stage runtime split-K 的 `requested_split_k` 已知，但 clamp仍在 generated launcher中，因此 `effective_split_k=None`，workspace按 requested上界分配。
- 任务 B 完成后，two-stage 的 `effective_split_k` 由 Python统一解析，并按精确值分配。
- 非 split-K kernel 的 requested/effective split-K均固定为 1；static fused kernel二者均取 `spec.split_k.static_value`。
- selector只返回已通过 compiled manifest校验的配置；未编译 kid是 fallback/error状态，不需要在合法 `LaunchConfig` 中再保存 `compiled=False`。

### 4.3 WorkspacePlan

新增 `aiter/ops/opus/_workspace.py`：

```python
@dataclass(frozen=True)
class WorkspacePlan:
    shape: tuple[int, ...]
    dtype: str
    required_numel: int
    alignment: int
    layout: str
    allocation_split_k: int     # 用于推导容量的 GEMM split数；不等于所有layout的存储slice数
```

必做任务首先支持现有 two-stage layout：

```text
[batch, allocated_split_k, padded_M, padded_N]
```

其中 `allocated_split_k` 在任务 A 是 requested上界，在任务 B 后是精确 `effective_split_k`。gfx1250当前只允许 batch 1，必须先拒绝其他 batch，而不是通过“公式不乘 batch”默许调用。对 fused layout，`allocation_split_k=spec.split_k.static_value`，但实际存储slice数少 1。

若实施可选 fused split-K，则再支持：

```text
[batch, tiles_M, tiles_N, spec.split_k.static_value - 1, block_M, block_N]
```

fused kernel的最后一个 split保留自身 fp32 accumulator，只需要发布前 `split_k - 1` 个 partial。它仍需要 workspace，只是不再单独 launch reduce kernel。

workspace 必须按元素 dtype 分配，不按猜测的 byte size 分配：

```python
workspace = torch.empty(plan.shape, dtype=to_torch_dtype(plan.dtype), device=A.device)
```

不使用 `torch.uint8` byte buffer作为默认接口：byte buffer虽然能提供足够地址空间，却丢失 writer dtype语义，无法防止 bf16/fp32 workspace混用。typed Tensor让 Python和generated launcher同时验证 dtype和元素容量。

默认不使用全局 `lru_cache`：PyTorch caching allocator已负责 eager重用；graph capture使用 graph-private pool；每次调用独立 Tensor也能避免双 stream共享 scratch。可以保留内部高级参数 `workspace=`，让明确管理生命周期的调用方复用。

caller-provided workspace的契约必须写清：Tensor至少存活到当前 stream上的 main/reduce kernel完成；两个可能重叠的 launch不得写同一 storage，除非调用方显式分区并传入对应 view。默认 wrapper始终为每次调用创建独立 Tensor，因此不会把这个并发责任转嫁给普通用户。

## 5. 任务 A：现有 split-K workspace Torch 化

### 5.1 边界与调用流程

任务 A 参考 PR #4246 的“外部 Tensor workspace + direct pointer”实现方式，但不引入 fused kernel，也不删除 tuned lookup/heuristic。所有公开和仓内 a16w16调用最终必须经过显式 kid路径：

```text
normalize
  -> resolve_launch_config()             # Python知道最终kid
  -> make_workspace_plan(spec, shape)    # 任务A按安全上界
  -> torch.empty(..., dtype=spec.workspace.dtype)
  -> _opus_gemm_a16w16_launch(..., workspace, kid, split_k)
```

当前 `_opus_gemm_bf16_dispatch -> C++ opus_gemm -> C++ heuristic` 路径必须在 PR0 中停止作为正常 Python fallback，否则 Python无法在 launch前知道 workspace类型和容量。旧 C++ selector保留一段时间只用于 golden对照；不能保留一条仍使用 raw HIP workspace的隐藏执行路径。

因此任务 A落地后，generic `opus_gemm()` 的 bf16分支不再是可用 launch入口：仓内私有 binding先迁到显式 kid wrapper，C++中的旧 shape selector只能通过 test-only probe读取结果。a8w8分支可暂时留在 generic entry，待任务 B拆成 family API。这样不会出现“allocator已经删除，但某条旧 bf16路径仍选中 split-K launcher且拿不到 workspace”的半迁移状态。

这是任务 A中唯一可能提前暴露的兼容断点。当前仓内 bf16调用只经过 `_opus_gemm_bf16_dispatch` 私有 wrapper，可以同批迁移；合入前仍应扫描 vLLM/sglang等已知下游是否直接取 extension symbol `opus_gemm`。若存在支持承诺，应在 Python层提供兼容 wrapper并写 release note，而不是让 C++ generic分支继续隐藏选核和分配策略。

### 5.2 分配策略

任务 A 不复制三份 launcher中的 runtime split-K clamp逻辑。对于 `spec.split_k.mode="runtime"` 的 two-stage kernel：

- `allocation_split_k` 使用 Python请求值/selector返回值的安全上界。
- explicit/tuned值先按 `SplitKSpec.max_splits` 校验；不允许用任意大 `splitK` 诱发无界 workspace分配。
- gfx942的 `splitK <= 0` auto语义先根据 CU数、tile数和 `auto_target_wg_per_cu` 得到初始候选，再把该候选作为安全上界；显式 `splitK == 1` 仍表示不切 K。
- launcher只会向下 clamp，因此上界容量恒安全。
- generated launcher按 clamp后的实际 split-K重新计算最小需求并做最后一道容量检查。
- 任务 B 将 clamp策略移到 Python后，再切换成精确容量。

workspace必须使用 kid声明的自然 dtype：

```python
workspace = torch.empty(
    plan.shape,
    dtype=to_torch_dtype(spec.workspace.dtype),
    device=A.device,
)
```

不使用无类型 `uint8` buffer，也不按 output dtype推导 workspace dtype。gfx942 的 bf16 workspace变体必须实际得到 bf16 Tensor；gfx950/gfx1250现有 fp32 workspace得到 fp32 Tensor。

### 5.3 低层 ABI 与安全检查

raw Python/C++ binding显式接收 workspace：

```python
_opus_gemm_a16w16_launch(
    A, B, out, bias,
    workspace,       # 非workspace kid显式传None
    kid,
    split_k,
)
```

公开 `gemm_a16w16_opus()` 和 `opus_gemm_a16w16_tune()` 的旧调用方式由 Python compatibility wrapper保持；不能仅依靠 C++末位 `workspace=nullopt` 默认值，因为 split-K旧调用若未经过 Python分配仍会失败。

每个 generated launcher根据自己的 compile-time traits检查：

- workspace是否存在。
- workspace与 A位于同一 device。
- workspace dtype等于 `D_WS`。
- workspace contiguous且地址满足 alignment。
- `workspace.numel() >= required_numel`。
- padded extent和 `required_numel` 使用 checked `size_t`乘法，任何溢出在 launch前失败。

容量和 dtype检查应位于 generated launcher或同源生成的 `KernelEntry` metadata中，不在 `opus_gemm.cu` 重新维护 `kid_needs_ws()` 区间判断。

随后将所有 kargs/reduce ABI从：

```text
const opus_splitk_ws_handle *ws_handle
```

改为：

```text
void *ptr_ws
```

device kernel直接将 `ptr_ws` 转成 `D_WS *`，不再通过 host/device mirror二次解引用。

### 5.4 删除旧机制

任务 A 同批删除：

- `opus_gemm.cu` 中 `SplitkWsRegistry`、`Owner`、mutex map和全部 `opus_splitk_ws_*` 函数。
- host pinned handle、device mirror和手工 `sync_to_device`。
- `opus_gemm_workspace_init()` 的 C++声明、pybind注册和真实实现。
- 三个 arch codegen中的 grow、4 MiB对齐、`hipDeviceSynchronize`、`hipMalloc/hipFree`。
- `aiter/tuned_gemm.py` 中 `_opus_needs_ws_prewarm`、默认 capture stream、warmed signature和 prewarm调用。

为下游兼容，Python层保留一个 release周期的 `opus_gemm_workspace_init()` 空壳，调用时发出 `DeprecationWarning`；C++ API和内部资源管理立即删除。

### 5.5 任务 A 的文件级修改

在基线 `35696023` 上，`rg -l 'ws_handle|opus_splitk_ws' csrc aiter` 精确命中 22 个文件，其中真正读取 handle 的 kernel面是 3 个 traits、8 个 pipeline和3个 reduce文件。下面按符号/职责定位，不把易漂移的行号当成实现契约。完成后应再次用同一命令审计；若 #4320 已合入，命中集合会增加其 BMM文件，见本节末尾的条件清单。

#### Python、metadata 和调用点

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `csrc/opus_gemm/opus_gemm_common.py` | `OpusGemmInstance`投影和查询 API | 补齐三架构现有 split-K kid的 workspace dtype/layout、split-K、alignment、batch和addressing事实；生成 frozen `KernelSpec`，不通过 kid范围推导。 |
| `aiter/ops/opus/_catalog.py`（PR0新增） | runtime metadata facade | 显式经 `aiter.jit.core`建立 `AITER_META_DIR`后转发 `get_kernel_spec()`；不复制实例表。 |
| `aiter/ops/opus/_build_plan.py`（PR0新增） | compile-set planner/fingerprint | 用与 codegen相同的输入计算 planned compiled set；区分首次构建计划与 actual manifest，禁止缺失 manifest时假定全量 kid可用。 |
| `aiter/ops/opus/_workspace.py`（新增） | `make_workspace_plan()`、`allocate_workspace()`、`validate_workspace()` | typed Tensor、requested上界容量、caller workspace校验；禁止全局无界 cache。 |
| `aiter/ops/opus/_selector.py`（PR0新增） | `resolve_launch_config()` | 在所有正常 Python路径中先得到 kid；旧 C++ selector仅用于 shadow/golden对照。 |
| `aiter/ops/opus/gemm_op_a16w16.py` | raw binding、tune wrapper、CSV miss fallback、workspace init | raw binding增加显式 workspace；所有路径改为 resolve -> allocate -> launch；移除 `_opus_gemm_bf16_dispatch` 正常执行路径；workspace init改为 deprecated空壳。 |
| `aiter/ops/opus/__init__.py` | exports/stubs | 暂时导出 workspace init兼容空壳并标记 deprecated；下一 release删除。 |
| `aiter/tuned_gemm.py` | 顶部 workspace helper和 `opus_gemm()` backend | 删除全部 prewarm逻辑；调用新的 Python wrapper，由 wrapper管理 workspace。 |
| `csrc/gemm_a16w16/gemm_a16w16_tune.py`、`csrc/opus_gemm/opus_gemm_tune.py` | OPUS runner | 继续调用 Python compatibility wrapper，不能直调缺少 workspace的旧 binding。 |

#### C++、traits 和 pipeline

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `csrc/opus_gemm/include/opus_gemm.h` | a16w16声明、workspace init声明 | 增加 optional workspace到低层 launch；删除 C++ workspace init。最终命名在任务 B收敛为 `opus_gemm_a16w16_launch`。 |
| `csrc/include/rocm_ops.hpp`、`csrc/pybind/opus_gemm_pybind.cu` | a16w16参数和 workspace init宏 | 注册 workspace参数；删除真实 workspace init binding。 |
| `csrc/opus_gemm/opus_gemm.cu` | registry、includes、bf16 auto path | 删除 registry及 `<mutex>/<unordered_map>`；停用 generic `opus_gemm()` 的 bf16 launch分支，旧 selector代码只供 test-only golden probe使用。a8w8分支暂留到任务 B拆分。 |
| `csrc/opus_gemm/include/opus_gemm_common.cuh` | shared include/ABI说明 | 删除 handle类型依赖和过期注释；只保留 direct-pointer launcher需要的共享定义。 |
| `csrc/opus_gemm/include/gfx942/opus_gemm_traits_a16w16.cuh` | handle、`opus_splitk_ws_ptr()` helper、split-K kargs | 删除结构体、guard和二次解引用 helper；kargs保存 `ptr_ws`。 |
| `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh` | flatmm split-K handle/kargs | 同上；明确 `D_ACC/D_WS/D_OUT`。 |
| `csrc/opus_gemm/include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh` | cluster TDM handle/kargs | 同上；保持当前 two-stage数值算法。 |
| `csrc/opus_gemm/include/gfx942/opus_gemm_pipeline_a16w16_em3en4_lds1_pgr2_sk.cuh`、`opus_gemm_pipeline_a16w16_kbuf1.cuh`、`opus_gemm_pipeline_a16w16_kbuf2v.cuh`、`opus_gemm_pipeline_a16w16_kbuf2v_bk128.cuh`、`opus_gemm_pipeline_a16w16_quad_mfma32_kbuf1.cuh` | workspace读取 | 改为直接使用 `ptr_ws`；不改变计算流水线。 |
| `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh` | workspace读取 | 改为 direct pointer；main仍写 fp32 partial。 |
| `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh`、`opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh` | workspace读取 | 改为 direct pointer；当前 batch==1约束进入 spec和launcher guard。 |
| `csrc/opus_gemm/include/gfx942/splitk_reduce_gfx942.cuh`、`gfx950/splitk_reduce_gfx950.cuh`、`gfx1250/splitk_reduce_gfx1250.cuh` | reduce ABI | 输入改为直接 workspace pointer；显式区分 `D_WS` 与 `D_OUT`。gfx942有 baseline/OOB重载和转发层，必须全部迁移，不能只改最外层签名。 |

#### codegen、测试和文档

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `csrc/opus_gemm/gen_instances.py` | `SPLITK_REDUCE_ABI_MAP`、launcher prototype | 三架构统一 direct-pointer ABI；保留 arch特有 reduce symbol override。 |
| `csrc/opus_gemm/codegen/common.py` | shared validation/template | 生成统一 workspace参数和安全检查，避免三份模板复制。 |
| `csrc/opus_gemm/codegen/gen_instances_gfx942.py` | split-K launcher grow块 | 接收 workspace Tensor；删除 allocator/handle逻辑；保留 bf16/fp32 workspace实例。 |
| `csrc/opus_gemm/codegen/gen_instances_gfx950.py` | flatmm split-K launcher grow块 | 同上。 |
| `csrc/opus_gemm/codegen/gen_instances_gfx1250.py` | current two-stage launcher grow块 | 同上；本任务不新增 fused generator。 |
| `op_tests/test_opus_workspace.py`（新增） | workspace contract | 刚好容量、少1元素、错误 dtype/device/noncontiguous、caller reuse、双 stream/TBO。 |
| `op_tests/test_opus_graph.py`（新增） | graph/compile | eager、HIP graph capture/replay、`torch.compile`和相同 shape双 stream。 |
| `op_tests/test_opus_a16w16_gemm.py` | 三架构 split-K矩阵 | 覆盖 gfx942 bf16 workspace、gfx950 fp32 workspace、gfx1250 batch限制和bf16/fp32 output。 |
| `aiter/ops/opus/README.md`、`csrc/opus_gemm/README.md` | workspace/TBO章节 | 删除 registry/prewarm说明，记录 typed Tensor和兼容空壳。 |

#### #4320 先合入时的任务 A 增量

#4320 会新增 gfx950 mxscale BMM split-K/fused workspace使用者。若它先于任务 A 落地且 Definition of Done仍是“整个 OPUS无 raw allocator”，还需同步修改：

| 文件/区域 | 增量动作 |
|---|---|
| `aiter/ops/opus/bmm_op.py` | high-level wrapper按 `(arch,family,kid)` spec分配 typed workspace；raw binding显式接收 workspace；tuned kid失效时仍原子丢弃 kid和split-K。 |
| `csrc/opus_gemm/include/opus_bmm.h`、`csrc/opus_gemm/opus_bmm.cu`、`csrc/include/rocm_ops.hpp`、`csrc/pybind/opus_gemm_pybind.cu` | BMM low-level ABI增加 workspace Tensor；unknown explicit kid改为严格失败。 |
| `csrc/opus_gemm/codegen/gen_instances_gfx950.py` | `_BMM_MXSCALE_SPLITK_LAUNCHER_BODY` 及 fused launcher模板删除 `opus_splitk_ws_get/hipMalloc/hipFree`，改用 direct `ptr_ws`并生成容量/dtype检查。 |
| `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh`、`splitk_reduce_gfx950.cuh`和相关 BMM pipeline | BMM kargs/reduce切到 direct pointer；为 partial和fused counter布局建立独立 `WorkspaceSpec`，不能套用 a16w16的 `[batch,split,padded_M,padded_N]`公式。 |
| `op_tests/test_opus_a8w8_bmm.py`、graph/workspace测试 | 覆盖 `splitK>1`、fused kid、少1元素、capture/replay和并发；当前 tuned CSV全是 `splitK=1`，仅跑 tuned表不会触达 allocator路径。 |

这部分不应凭当前 open PR 的行号预写 patch；#4320 rebase仍在变化。合入后先重新运行符号清单，再按 family workspace layout补 spec和测试。

## 6. 任务 B：接口重构方案

### 6.1 Python API

保留高层兼容接口：

```python
gemm_a16w16_opus(
    A,
    B,
    bias=None,
    dtype=torch.bfloat16,
    *,
    kernelId=None,
    splitK=None,
    out=None,
    workspace=None,       # internal/advanced parameter
)
```

新增内部低层 binding：

```python
_opus_gemm_a16w16_launch(
    A,
    B,
    out,
    bias,
    workspace,
    kid,
    effective_split_k,
)
```

如果 a8w8 继续由 OPUS module 暴露，则使用独立入口：

```python
_opus_gemm_a8w8_launch(A, B, out, kid)
_opus_gemm_a8w8_blockscale_launch(A, B, x_scale, w_scale, out, kid)
```

不再让 a16w16 入口携带 `group_layout/x_scale/w_scale/bpreshuffle` 等无关参数。

`opus_gemm_a16w16_tune()` 保留一个 release cycle 作为 Python compatibility wrapper：它根据 kid 查询 spec、准备 workspace 后调用新 launch binding。旧 C++ tune binding 不应继续独立拥有一套 policy。

入口数量不是验收目标。plain a8w8与 blockscale a8w8如果 tensor/layout契约不同，就应保留不同 launcher；不能为了“3个入口变2个入口”重新制造 optional-scale mega entry。纯 kid dispatch统一使用 `launch` 命名，`tune` 仅保留为兼容别名。

接口迁移关系如下：

| 当前接口/调用点 | 目标接口 | 兼容策略 |
|---|---|---|
| `gemm_a16w16_opus()` | 同名高层 auto API | 保留；内部统一走 normalize -> selector -> workspace -> launch。 |
| `opus_gemm_a16w16_tune()` | Python wrapper -> `_opus_gemm_a16w16_launch()` | 旧名字保留一个 release并告警；不再直接绑定一套独立 C++ policy。 |
| C++/pybind `opus_gemm()` mega entry | family-specific a16w16/a8w8 launch binding | Python调用点迁完后删除，不在 C++保留 dtype/scale自动路由。 |
| `opus_gemm_workspace_init()` | 无需替代 | Python no-op兼容一个 release；真实 C++ binding在任务 A立即删除。 |
| `aiter/tuned_gemm.py` 的 OPUS backend | 高层 API或显式 `LaunchConfig` | 禁止绕过 workspace planner直调 raw binding。 |
| 当前 generic entry中的 a8w8 no-scale/scale分支 | `_opus_gemm_a8w8_launch()` / `_opus_gemm_a8w8_blockscale_launch()` | 新建 Python family module；是否公开 auto API由该 family能力决定。 |

### 6.2 Python selector

新增 `aiter/ops/opus/_selector.py`，固定执行顺序：

```text
1. normalize shape/dtype/layout
2. explicit kid（若给定）-> 严格校验 -> return/error
3. runtime tuned CSV lookup -> 校验 tuned kid与split-K
4. tuned配置无效时原子丢弃 (kid, split-K)，再进入 Python heuristic
5. validate heuristic spec capability + compiled-kid set
6. resolve effective split-K
7. return LaunchConfig
```

规则：

- explicit kid 不兼容时直接抛错，不静默换 kernel。
- tuned row的 kid只要未编译、arch/family不匹配或不满足 shape/dtype/layout/alignment，就必须把该行的 kid和split-K一起作废；默认 warning后走 heuristic，strict模式直接失败。不得把旧 tuned split-K传给新 kid。该规则直接采用 #4320 已验证的 shape gate。
- heuristic产生自己的 split-K默认值/策略；它的结果再次走同一 capability和compiled-set校验，失败后才交给 framework fallback。
- heuristic 只能从当前 build 的 compiled-kid set 中选择。
- 不支持的 shape/layout 应交给 framework fallback，而不是让 C++再次选择。
- selector迁移阶段同时调用旧 C++ selector做 shadow comparison，只比较 `(kid, split_k)`，不重复 launch kernel。
- raw C++ binding收到的 kid必须精确存在于对应 `(arch,family,output dtype)` dispatch表；unknown kid直接 `AITER_CHECK`失败，不能采用 #4320 当前的默认 launcher回退。

任务 B完成后，传给 raw binding的是已经解析好的 effective split-K。generated launcher可以用同源常量重算一次并 `AITER_CHECK` 相等，但不得再静默向下改值；否则 Python按一个 split数分配、kernel按另一个 split数执行，职责边界又会失效。

不要在 `resolve_launch_config()` 外再套无界 `functools.cache`。tuned table加载后已经是 O(1) dict lookup；如果性能数据证明还需缓存，key必须至少包含 device/gfx/CU数、batch、M/N/K、全部 dtype/layout、bias、配置版本和 compiled manifest版本，并提供失效机制。

### 6.3 tuned lookup

新增中立 loader `aiter/ops/gemm_tune_lookup.py`，由 `tuned_gemm.py` 和 OPUS selector 共用。runtime key 至少包含：

```text
gfx, cu_num, batch, M, N, K,
input A/B dtype, output dtype,
bias flag/dtype, A/B layout,
scale mode/dtype/block shape, B preshuffle layout, kernel family
```

build-time 和 runtime 的 CSV 用途必须区分：

- build-time `--tune_files` 可以取多个 CSV 的 kid 并集，用于 subset compile。
- runtime 只能查询 framework 当前明确选择的配置文件，不能默认 glob 所有 model CSV 后取全局最小 `us`。

codegen最终仍需维持：

```text
compiled_set = (
    csv_kids
    | sidecar_kids
    | heuristic_fallback_kids
    | mandatory_family_kids
) & valid_kids
```

`mandatory_family_kids` 包含同一 module内无 selector替代的固定入口，例如当前 gfx950 a8w8 no-scale/scale kid；若未来拆成独立 module，则由各 family自己的 build recipe维护。删除 C++ shape lookup时只删除“CSV -> runtime lookup header”的部分，不能删除上述任何 compile-set来源。CI应逐行验证 runtime opus tuned row的 `solidx` 存在于 compiled manifest。

首次 JIT build前可能还没有 actual manifest，不能把“manifest不存在”解释为“所有 catalog kid都已编译”。应由共享的 pure-Python build-plan helper先计算 `planned_compiled_set + fingerprint`，selector在首轮只从 planned set选择；extension生成完成后写出带相同 fingerprint的 actual manifest。已有 `.so` 但 manifest缺失或 fingerprint不匹配时应触发重建或明确失败，不能带着 stale set继续运行。

当前 `{bd_dir}/compiled_kids_opus.json` 更接近跨 rebuild保留的 planned sidecar，而 `{working_path}/compiled_kids.json` 是本次生成结果。重构后应在命名或 JSON字段中明确 `planned`/`actual`、arch/family、generator版本和 fingerprint，避免 selector把 sidecar愿望集合误当成已经进入当前 `.so` 的事实。

旧 CSV 没有 `gfx` 时，只能在当前 arch 唯一、无冲突时兼容；新输出必须写入 `gfx`。

### 6.4 heuristic 迁移

新增：

```text
aiter/ops/opus/heuristics/
├── __init__.py
├── gfx942.py
├── gfx950.py
└── gfx1250.py
```

第一阶段逐字等价移植现有 C++规则，并用 shape corpus 对比旧、新 `(kid, split_k)`。在完成 differential test 前不调整性能策略。

gfx942 当前部分 heuristic 直接关联 launcher symbol；迁移时从权威 `KernelSpec.launcher_symbol` 反查 symbol -> `(arch,family,kid)`，不能手写第二张表。对三架构的 CSV hit/miss corpus逐条对比旧、新 `(kid, split_k)`，全部差异解释完成后才能删除 C++ header。

shadow comparison不得新增长期 production ABI。可在测试构建开关下生成一个 legacy selector probe，把旧 function pointer通过同源的 symbol -> kid表转换成 kid，并将 corpus结果写成 golden JSON；Python selector与该 JSON/测试 probe比较。完成迁移后 probe与旧 heuristic header同批删除。

### 6.5 C++ ABI

`include/opus_gemm.h` 最终只声明 family-specific launcher：

```cpp
void opus_gemm_a16w16_launch(
    aiter_tensor_t& A,
    aiter_tensor_t& B,
    aiter_tensor_t& out,
    std::optional<aiter_tensor_t> bias,
    std::optional<aiter_tensor_t> workspace,
    int kid,
    int split_k);

void opus_gemm_a8w8_launch(...);
void opus_gemm_a8w8_blockscale_launch(...);
```

C++ 中删除：

- `(M,N,K) -> launcher` runtime lookup。
- heuristic。
- kid-range capability 判断。
- workspace registry/init/release/prewarm。
- output/workspace allocator。
- a16w16/a8w8 混合 mega entry。

C++ 中保留：

- runtime arch 与编译 arch检查。
- `(arch, family, kid, output dtype) -> launcher`。
- tensor dim/dtype/layout 和 workspace 最小容量的安全检查。
- kargs 构造、main/reduce launch 和 launch error reporting。

不要在 `opus_gemm.cu` 新建 `(input dtype, output dtype, needs_workspace) -> CType` policy表；`needs_workspace` 无法表达 gfx942 bf16 workspace等差异，也会继续混淆 `D_WS` 与 `D_OUT`。codegen应直接生成 `(kid, output dtype) -> launcher` entry，launcher的 compile-time traits决定 `D_ACC/D_WS`。

### 6.6 任务 B 的文件级修改

#### Python 文件

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `csrc/opus_gemm/opus_gemm_common.py` | authoring records、frozen spec projection和查询 API | 在任务 A workspace字段基础上补齐 dtype/layout/bias/alignment/addressing等 capability；维持唯一实例登记处。 |
| `aiter/ops/opus/_catalog.py`（PR0新增） | runtime facade | 只转发权威查询与 fingerprint，不复制 metadata；保证 source/editable/wheel导入一致。 |
| `aiter/ops/opus/_build_plan.py`（PR0新增） | planned/actual compiled set | 纳入完整 tune key、mandatory family和manifest fingerprint；与 codegen输出做一致性校验。 |
| `aiter/ops/opus/_selector.py`（PR0新增） | selection policy | 从 shadow模式切换为唯一 selector；完成 exact effective split-K、compiled set过滤和 fallback。 |
| `aiter/ops/opus/_workspace.py`（任务 A新增） | runtime split-K planner | 从 requested上界切换到 selector给出的精确 effective split-K；保留 caller workspace校验。 |
| `aiter/ops/opus/heuristics/*.py`（新增） | 各 arch selector | 等价迁移 C++ heuristic，只返回 kid/split-K，不返回 launcher symbol。 |
| `aiter/ops/gemm_tune_lookup.py`（新增） | 全文件 | 统一 tuned CSV loader和完整 key；提供按 `libtype/family` 查询。 |
| `aiter/ops/opus/gemm_op_a16w16.py` | module doc、`opus_gemm_a16w16_tune()`、`gemm_a16w16_opus()` | 任务 A已完成 allocate/launch；本任务收敛公开入口、删除残余 auto-dispatch兼容路径并将 raw binding正式改名为 launch。 |
| `aiter/ops/opus/gemm_op_a8w8.py`（新增） | 全文件 | 从 mega entry中接管 a8w8/a8w8-scale接口；不同 scale/layout family 使用不同 launcher。 |
| `aiter/ops/opus/common.py` | CSV loader、mono kid表 | 首先改成对统一 tune loader/`_catalog` facade 的 compatibility re-export；所有调用迁完后删除。mono tile capability 从 KernelSpec读取。 |
| `aiter/ops/opus/__init__.py` | import/stub/`__all__` | 导出新的 family public API；在兼容周期结束时移除 `opus_gemm_workspace_init` 空壳；旧 tune wrapper标记 deprecated。 |
| `aiter/ops/opus/_arch.py` | arch probe 文档/接口 | 保留探测功能；不要在包级别假设所有 OPUS family 支持同一 arch，支持集合由 family catalog决定。 |
| `aiter/tuned_gemm.py` | OPUS backend和 tuned config读取 | prewarm已由任务 A删除；本任务复用统一 tune loader并直接调用高层 OPUS API或 selector结果。 |
| `aiter/ops/deepgemm.py` | `opus_gemm_a16w16_tune` shim | 继续转发到 Python compatibility wrapper；更新 warning 和新参数；下一 release删除。 |

#### C++ binding 和入口

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `csrc/opus_gemm/include/opus_gemm.h` | 任务 A后的 launch声明和旧 mega entry | 用稳定的 family-specific launch声明替换 `opus_gemm()`/旧 tune名字；workspace ABI保持不变。 |
| `csrc/include/rocm_ops.hpp` | `OPUS_GEMM_*_PYBIND` 宏 | 注册最终 a16w16/a8w8 family launch名字；兼容名只放在 Python。 |
| `csrc/pybind/opus_gemm_pybind.cu` | `PYBIND11_MODULE` | 注册最终 family bindings；不再注册 generic mega entry。 |
| `csrc/opus_gemm/opus_gemm.cu` | `opus_dispatch_a16w16()` router、`opus_gemm()`、`opus_kid_*`范围函数 | registry已由任务 A删除；本任务删除 auto dispatch、mega entry和全部 kid-range policy；保留/改名 `opus_a16w16_tune_dispatch()` 这一类纯 kid router及小型 family launcher。 |
| `csrc/opus_gemm/include/opus_gemm_arch.cuh` | arch probe/shared types | 保留 arch探测；将真正跨 arch 的 launcher entry/metadata类型放在这里或单独的 `opus_gemm_dispatch.cuh`。 |
| `csrc/opus_gemm/include/opus_gemm_common.cuh` | traits umbrella | 删除 workspace handle相关依赖；只聚合 launcher构造需要的 traits/kargs。 |

#### 每个 arch 的 dispatch 和 kernel ABI

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `csrc/opus_gemm/include/gfx942/opus_gemm_arch_gfx942.cuh` | `opus_dispatch_a16w16_gfx942<>` 与 `opus_a16w16_tune_dispatch_gfx942<>` | 删除前者的 shape lookup/heuristic fallback；保留并改名后者的 generated kid dispatch。 |
| `csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh` | `opus_dispatch_a16w16_gfx950<>`、`kSplitkKid*`/`kid_is_splitk()`、4GiB guard | 删除 shape policy、kid区间和4GiB策略；保留 `opus_a16w16_tune_dispatch_gfx950<>`。4GiB能力由 KernelSpec/Python过滤，launcher同源复核。 |
| `csrc/opus_gemm/include/gfx1250/opus_gemm_arch_gfx1250.cuh` | `opus_dispatch_a16w16_gfx1250<>`、`check_shape_4g()` 与 tune dispatch | 删除 shape/heuristic和粗粒度4GiB策略；保留 `opus_a16w16_tune_dispatch_gfx1250<>`。若可选任务 C落地，fused kid进入同一表。 |
| `csrc/opus_gemm/include/gfx942/opus_gemm_heuristic_dispatch_gfx942.cuh` | 全文件 | Python heuristic golden对齐完成后删除。需要的 function-pointer typedef先移到共享 dispatch header。 |
| `csrc/opus_gemm/include/gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh` | 全文件 | 同上。 |
| `csrc/opus_gemm/include/gfx1250/opus_gemm_heuristic_dispatch_gfx1250.cuh` | 全文件 | 同上。 |

traits、split-K pipeline和reduce ABI已由任务 A完成 direct-pointer迁移；任务 B不再机械重写 kernel文件。

#### codegen、build 和 tuner

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `csrc/opus_gemm/gen_instances.py` | `gen_lookup_dict()`、`get_tune_dict()`、CSV烘焙调用、`gen_a16w16_tune_lookup()` | 删除前三者对应的 `(M,N,K)` runtime lookup生成；保留并改名 `gen_a16w16_tune_lookup()` 为 kid dispatch生成器，产物改为 `opus_gemm_a16w16_kid_dispatch.h`；保留 `csv_kids`读取、sidecar、heuristic和mandatory family compile set，并生成带 fingerprint的 manifest。 |
| `csrc/opus_gemm/opus_gemm_common.py` | `OpusGemmInstance`、kid集合、capability helper | 保持 authoring SSOT；新增/完善 immutable spec投影，删除其他文件中的 kid range/capability副本。 |
| `csrc/opus_gemm/codegen/common.py` | generated dispatch metadata | workspace模板已由任务 A统一；本任务增加/收敛 `(kid, output dtype) -> launcher` entry结构。 |
| `aiter/jit/optCompilerConfig.json` | `module_deepgemm_opus.blob_gen_cmd` | 保留 tune files用于 compile-set并集，但更新 generator参数名/说明，确认不再生成 runtime shape lookup。 |
| `csrc/opus_gemm/opus_gemm_tune.py` | launch和candidate逻辑 | 使用新 catalog/low-level wrapper；输出完整 arch/family/workspace metadata；不触发旧 C++ heuristic。 |
| `csrc/gemm_a16w16/gemm_a16w16_tune.py` | OPUS runner | 使用新 Python launch wrapper；不直接依赖旧 pybind签名。 |
| `op_tests/test_gemm_codegen.py` | OPUS codegen断言 | 增加“无 runtime lookup”“dispatch kid唯一”“generated files稳定”“无 allocator模板”等检查。 |

#### 文档和测试

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `aiter/ops/opus/README.md` | dispatch、API、workspace、build、file map | 按目标调用链重写；删除 C++ lookup和 prewarm说明；记录兼容周期。 |
| `csrc/opus_gemm/README.md` | layout、dispatch、split-K/TBO | C++侧只描述 kid launcher和kernel文件；删除旧 registry生命周期说明。 |
| `op_tests/test_opus_a16w16_gemm.py` | 现有 e2e | 区分 explicit/tuned/heuristic路径；检查 selector source；加入多 arch、bias和output dtype矩阵。 |
| `op_tests/test_opus_dispatch.py`（新增） | selector单元测试 | 不需要 GPU kernel即可验证完整 key、优先级、stale CSV、unsupported kid和heuristic filter；迁移期读取 test-only legacy probe生成的 golden corpus。 |
| `op_tests/test_opus_graph.py`（新增） | HIP graph/compile | eager、capture/replay、`torch.compile`、同 shape双 stream并发。 |

### 6.7 与 #4320 的冲突面和保留项

#4320 是 open PR，不应在本文中假定它已经先合或后合；但它与任务 A/B 修改同一组枢纽文件，排期前必须选择 rebase方向。

| #4320 文件 | 与本方案的交集 | 合并时要求 |
|---|---|---|
| `aiter/jit/core.py` | 新增 mxscale BMM tuned config并修改 merge/dedup规则；任务 B也要统一 tuned loader | 保留新 family配置；将 dedup key规则纳入统一 loader测试，不回退其 trailing `tuned`文件名修复。 |
| `aiter/jit/optCompilerConfig.json` | 同一个 `module_deepgemm_opus` build recipe | BMM tune file继续参与 compile set；“不生成 runtime shape lookup”不能误删 BMM kid dispatch。 |
| `aiter/ops/opus/__init__.py` | family导出列表 | 保留 BMM public API和 arch gate；只统一底层 catalog/selector约定。 |
| `csrc/include/rocm_ops.hpp`、`csrc/pybind/opus_gemm_pybind.cu` | 同一 extension的注册点 | 保留 `opus_bmm_a8w8_mxscale` family binding；不要为入口数量目标把 BMM塞回 GEMM mega entry。 |
| `csrc/opus_gemm/gen_instances.py` | manifest、kernel集合、kid lookup生成 | 删除的只有 a16w16 `(M,N,K)` runtime lookup；保留 #4320 的 BMM kid map、manifest声明和 family compile set，并逐步统一为 `*_kid_dispatch`生成器。 |
| `csrc/opus_gemm/codegen/gen_instances_gfx950.py` | task A workspace模板和 #4320 BMM emit都在此文件 | 若追求模块级 allocator清零，先按 §5.5迁移 BMM workspace，再做机械冲突解决；不得保留隐藏的 BMM `hipMalloc`块。 |
| `csrc/opus_gemm/opus_gemm_common.py` | 两边都增加 instance字段和 family列表 | 以 `(arch,family,kid)`为身份合并；#4320 已出现跨 family重复 kid，任何只以整数 kid为 key的全局 catalog都不合法。 |
| `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh`、`splitk_reduce_gfx950.cuh` | #4320也修改 shared traits/reduce | 先按最终文件重新审计模板实例和 ABI，不能靠旧行号套 patch。 |
| `csrc/opus_gemm/README.md` | 两边都更新 file map/dispatch说明 | 文档同时保留 BMM family，并明确其最终 workspace所有权。 |

若 #4320 即将合入，较少返工的顺序是先在其 head上 rebase PR0/任务 A/B，因为它以新增 family为主，而本方案会系统改写公共注册和生成层；若其落地时间不确定，任务不应被 open PR无限阻塞，可以先在当前基线实施，但必须给 #4320 提供迁移说明：改用 frozen spec查询、严格 unknown-kid行为和外部 workspace ABI。无论哪种顺序，#4320 新增的 `opus_bmm.cu`、`opus_bmm.h`、`bmm_op.py`及测试都应保留为独立 family，而不是被任务二删除。

## 7. 可选任务 C：gfx1250 fused split-K

只有在需求方明确要求新增 fused kernel时才实施本节，并单独提交 PR。任务 C复用任务 A的 typed workspace和任务 B的 catalog/dispatch，不增加新顶层入口。

### 7.1 算法与约束

- `SplitK`、cluster geometry和 `D_WS` 是每个 kid的编译时属性。
- cluster x维对应 K split，cluster y维用于可选的 N-tile peer multicast。
- 非最后 split把 partial写入 `ptr_ws`。
- cluster barrier保证最后 split读取 partial前所有发布完成。
- 最后 split保留自己的 fp32 accumulator，累加前 `SplitK - 1` 个 partial、融合 bias并一次性转换到最终输出。
- fused kernel仍然需要 workspace，但不需要独立 reduce kernel。

必须由 KernelSpec表达：

```text
spec.split_k.static_value <= ceil_div(K, block_k)
product(spec.cluster_dims) <= 16
workspace layout = tile_major_fused
workspace dtype = bf16 或 fp32
batch mode / bias / M-N-K tail capability
```

### 7.2 文件修改

| 文件 | 修改内容 |
|---|---|
| `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_fuse_gfx1250.cuh`（新增） | fused main/reduce pipeline、cluster barrier和partial发布/读取。 |
| `csrc/opus_gemm/include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh` | 新 fused kargs/traits，复用任务 A的 `ptr_ws`，明确 `D_ACC/D_WS/D_OUT`。 |
| `csrc/opus_gemm/codegen/gen_instances_gfx1250.py` | 新 `gen_splitk_fuse_instance()`，生成具体 cluster dims和static split-K launcher。 |
| `csrc/opus_gemm/opus_gemm_common.py` | 在权威实例表注册 fused kid，并投影 static split-K、cluster、workspace和capability；runtime经 `_catalog.py` 自动可见。 |
| `csrc/opus_gemm/gen_instances.py` | 注册 generator并将 kid加入 subset compile和现有 kid dispatch。不得新增 shape lookup。 |
| `csrc/opus_gemm/opus_gemm_tune.py` | fused kid不 sweep runtime split-K；按static属性过滤shape并记录结果。 |
| `op_tests/test_opus_splitk_gfx1250.py`（新增） | 覆盖static split-K、bf16/fp32 workspace/output、bias、K-tail和非法cluster配置。 |

## 8. dtype 与 layout 契约

### 8.1 dtype

所有层统一使用以下语义：

```text
input_dtype_a
input_dtype_b
acc_dtype
workspace_dtype
output_dtype
bias_dtype
```

split-K 常见合法组合是：

```text
A/B       = bf16
ACC       = fp32
WORKSPACE = bf16 或 fp32（由 kid 决定）
OUT       = bf16 或 fp32
BIAS      = spec声明的 bf16/fp32
```

不能因为 OUT 是 bf16 就分配 bf16 workspace。workspace writer和 reader的类型必须与 `workspace_dtype` 完全一致，否则可能产生越界和 GPU hang。

### 8.2 layout

a16w16 的逻辑布局固定为：

```text
A         [batch, M, K]       K-contiguous
B         [batch, N, K]       K-contiguous，计算 A @ B^T
Y         [batch, M, N]       contiguous
bias      [N] 或 [batch, N]
workspace 由 WorkspacePlan定义
```

物理 stride不能被统一收紧为“所有输入完全 contiguous”。当前 A/B已经支持 padded leading dimension，应保持：

```text
A/B stride[-1] == 1
A/B stride[-2] >= K
batch == 1，或 stride[0] == rows * stride[-2]
Y stride == (M*N, N, 1)
```

这会拒绝 batch broadcast view、transpose和overlapping slice，但保留合法的 padded row pitch。bias shape继续允许 `[N]` 或 `[batch,N]`，dtype组合则必须按 `KernelSpec.output_bias_dtype_pairs` 校验：当前 gfx950/gfx942共享 launcher要求 bias dtype匹配 output，gfx1250 two-stage reduce额外支持 bf16 output + fp32 bias。高层 API可以接收 bf16/fp32 bias，但 selector只能选择真正支持该组合的 kid；不能把 gfx1250的能力误推广到其他 arch，也不能在重构时丢掉已支持的组合。

每个 spec需要声明 tail、batch和 stride capability。对于 transpose、expand/broadcast、负 stride 和 sliced view，高层 wrapper只能选择以下一种行为：

1. kernel原生支持；
2. 明确 materialize contiguous；
3. 抛出清晰错误。

禁止依赖 launcher中硬编码 stride 后继续接受任意 view。

4GiB校验必须基于实际可访问 storage span，而不是只看逻辑 `numel`。对本方案允许的非负 stride tensor，可按 `1 + Σ((size[d]-1) * stride[d])` 得到覆盖元素数，再乘 element size；随后结合 spec的 `addressing_mode` 判断是整段 descriptor受限、每个 workgroup窗口受限，还是使用64位 flat addressing。Python做策略过滤，generated launcher保留同源的边界复核。

## 9. 实施 PR 顺序

### 上游集成门：先确认 #4246/#4320 的落地状态

#4246 对任务 A 有三种可行集成方式：

1. 等 #4246 整体合入，再补 gfx942/gfx950并清理剩余 registry。代码量最少，但任务被 fused pipeline的 review进度绑定。
2. 在当前基线独立完成三架构 workspace迁移，由 #4246 后续 rebase。边界最可控，也能尽快消除 allocator，但双方会在 gfx1250文件产生冲突。
3. 从 #4246 提取 workspace-only改动先合，再让 fused pipeline基于它。架构和 review边界最好，若能协调上游维护者优先推荐。

不能只 cherry-pick `4ac1fa8` 就宣告完成：该 commit仅覆盖 gfx1250，且最初按默认 bf16分配；`86d1eba` 才按 kid真实 workspace dtype/shape修正潜在越界并删除 prewarm，但它同时理解 #4246 新增的 fused kid。实际实施应移植两者的最终不变量，而不是机械搬一个 commit。

#4320 的 rebase选择见 §6.7。若它先合，PR0必须把 BMM family纳入 `(arch,family,kid)` catalog和compiled manifest；任务 A是否同时迁移其 raw allocator必须在 PR范围中明确。该选择是排期决定，不是任务 A/B 的架构依赖。

### PR0：权威 metadata 查询层与 Python selector bridge

- 保留 `csrc/opus_gemm/opus_gemm_common.py` 为 authoring SSOT，补齐字段并提供 frozen `KernelSpec`查询；新增无数据副本的 `_catalog.py` runtime facade。
- 加 source/editable/wheel三种无 GPU import/fingerprint测试，固定 `AITER_META_DIR -> csrc` 契约。
- 新增 `LaunchConfig`、`WorkspaceSpec`、planned compile-set helper和带 fingerprint的 compiled-kid manifest读取。
- 等价移植 gfx942/gfx950/gfx1250 heuristic，先以 shadow模式对比旧 C++ selector。
- 对 CSV hit和miss corpus完成 `(kid, split_k)` golden对齐后，高层 Python切到显式 kid launch。
- 不删除旧 C++ selector，不改变 kernel数值实现；metadata投影加入后既有 kernel codegen产物应无差异。

### PR-A：现有 split-K workspace Torch 化

- 按 gfx1250 -> gfx950 -> gfx942 的提交顺序迁移 direct-pointer ABI；gfx942最后处理其 bf16 workspace变体。
- Python使用 typed `torch.empty`，任务 A阶段按 runtime split-K安全上界分配。
- 删除三架构 launcher内部 allocator、全部 handle/registry和 `tuned_gemm.py` prewarm。
- 若 #4320 已合且采用模块级 allocator清零目标，同批完成 §5.5 的 BMM增量；若不纳入，PR标题、DoD和审查命令都限定为 a16w16。
- Python保留 `opus_gemm_workspace_init()` deprecation空壳一个 release。
- 完成 HIP graph、双 stream/TBO、错误 workspace和三架构数值测试。

### PR-B1：Python 成为唯一 dispatch policy

- 抽出统一 tuned loader和完整 runtime key。
- selector解析精确 effective split-K，workspace由上界切换为精确分配。
- 删除 `opus_gemm_lookup.h` 生成和三个 C++ heuristic。
- 保留并重命名 kid dispatch；保留 build-time `csv_kids`、sidecar、heuristic fallback和mandatory family compile set。
- CI验证所有 runtime tuned/heuristic kid均存在于 compiled manifest。

### PR-B2：拆分 family API 与 dtype/layout 清理

- 删除 generic `opus_gemm()` mega entry。
- a16w16、plain a8w8、blockscale a8w8按实际契约分别提供 launch binding。
- `D_C` 语义拆成 `D_ACC/D_WS/D_OUT`。
- 合并 Python normalize/layout检查，同时保留 padded leading dimension和各 kid已支持的 bias dtype组合；若要把 fp32 bias扩展到其他 arch，应单独增加 reduce实例和测试。
- 旧 `opus_gemm_a16w16_tune()` 作为 Python别名保留一个 release。

### PR-C：可选 gfx1250 fused split-K

- 仅在需求明确后实施。
- 复用已有 catalog、typed workspace和 kid dispatch，不增加入口和 shape lookup。
- 先只支持 explicit/tuner路径；数值和性能验证后再允许 tuned/heuristic选择。

### PR-cleanup：兼容层与文档

- 下一 release删除 workspace init空壳、旧 tune alias和 `common.py` shim。
- 更新 tuner、文档、示例和 release note。
- 创建并验证新 kernel接入 skill。

## 10. 验收矩阵

### 10.1 现有 split-K 功能

- gfx942、gfx950、gfx1250现有 two-stage split-K。
- `split_k=1`、runtime `>1`、requested上界和 clamp边界。
- gfx942 bf16/fp32 workspace；gfx950/gfx1250 fp32 workspace。
- bf16/fp32 output；按每个 spec覆盖合法的 bias dtype组合及 `[N]`、`[batch,N]`，并对不支持的跨 dtype组合做负例测试。
- M/N/K tail及每个 spec明确不支持的边界。
- CPU fp32或可靠 GEMM reference correctness。
- tuned shape性能不应出现不可解释的回退。

### 10.2 workspace 安全与生命周期

- typed workspace正好容量、少1 element、错误 dtype、错误 device、noncontiguous和错误alignment。
- eager、HIP graph capture/replay、`torch.compile`。
- 两个 stream并发运行相同 shape和不同 shape。
- 两个 CPU线程各驱动一条 stream的 TBO场景。
- caller-provided workspace生命周期和复用。
- 多次不同 shape调用后显存占用不因无界 Python cache持续增长。
- 若 #4320 已纳入任务 A，额外覆盖 mxscale BMM `splitK>1`和fused counter workspace；其当前 tuned CSV全为 `splitK=1`，必须构造显式用例。

### 10.3 dispatch、lookup 和 build

- explicit、tuned hit、heuristic miss、framework fallback。
- explicit unknown/incompatible kid严格失败；不得静默落到默认 launcher。
- tuned kid因 alignment、dtype、arch/family或compiled-set失效时，断言 kid和split-K被原子丢弃，heuristic不会继承 stale split-K。
- 三架构全 CSV行和代表性 miss shape，重构前后 `(kid, split_k)` golden对比。
- gfx942 launcher symbol -> kid映射由权威 spec生成并覆盖测试。
- source、editable install、wheel install均可在无 GPU/未加载 extension时导入 `_catalog.py`，代表性 spec和 catalog fingerprint一致。
- 单 arch、多 arch和 subset compile。
- 每个 runtime tuned row、每个 heuristic可能返回值和mandatory family kid均属于 compiled set。
- stale CSV、未编译 kid产生明确 warning/error。
- `opus_gemm_lookup.h` 不再生成，kid dispatch仍存在。
- 生成文件二次生成 diff为空。
- 小 M shape端到端 host overhead基准，确认 Python selector没有明显回退。

### 10.4 可选 fused split-K

- static split-K、cluster大小、K steps、N-tail/batch限制。
- bf16/fp32 workspace和bf16/fp32 output。
- 证明只有一个 fused kernel launch、没有独立 reduce launch，但仍正确使用 `SplitK - 1` partial workspace。

## 11. Definition of Done

任务 A 完成必须满足：

- gfx942/gfx950/gfx1250现有 split-K全部接收外部分配的 typed workspace Tensor。
- 正常 Python a16w16路径在 launch前已知最终 kid，不存在隐藏的 C++ auto路径继续 raw allocation。
- a16w16 launch路径不存在 workspace相关 `hipMalloc`、`hipFree`、`hipHostMalloc` 或 registry；若 PR范围声明模块级清理，则 #4320 BMM路径也必须满足，不能只检查 `opus_gemm.cu`。
- workspace dtype、device、contiguous和容量错误在 launch前被拒绝，不允许越界或 hang。
- graph capture、同 shape双 stream和TBO测试通过。
- fused kernel不是任务 A完成条件。

任务 B 完成必须满足：

- runtime tuned lookup和 heuristic只在 Python执行一次。
- tuned配置失效时 `(kid, split-K)` 原子回退；explicit/low-level unknown kid严格失败。
- C++ 不再包含 `(M,N,K)` lookup和 framework fallback policy。
- C++ 只按 arch/family/kid/output dtype dispatch。
- codegen只生成 kid dispatch，不生成 runtime shape lookup；build-time CSV compile set仍保留。
- a16w16、plain a8w8和blockscale按各自契约使用 family-specific API，不以入口数量作为目标。
- dtype/layout/workspace/addressing capability来自统一 KernelSpec。
- padded leading dimension和各 arch当前合法的 bias dtype组合没有被意外收紧或错误放宽。
- 旧公开 Python API在约定兼容周期内仍可用，并发出明确 deprecation warning。

可选任务 C 完成必须满足：

- fused split-K数值和性能经过独立验证。
- fused kernel仍使用合法 typed workspace，并明确没有独立 reduce kernel。
- 新 kid只需登记 catalog/codegen/test，不修改顶层 C++/Python入口。

## 12. Bonus：新 OPUS kernel 接入 skill

skill应在任务 B接口稳定后创建，避免固化即将删除的 pattern。若目标是 Codex，使用 Codex skill结构；仓内 `.claude/skills` 可共享参考材料和校验脚本，但不能把 Claude专用目录当成 Codex安装位置。

建议结构：

```text
integrate-opus-kernel/
├── SKILL.md
├── agents/openai.yaml
├── scripts/
│   ├── add_kernel_spec.py
│   ├── validate_kernel_catalog.py
│   └── check_generated_diff.py
└── references/
    ├── architecture.md
    ├── kernel-spec-schema.md
    └── test-matrix.md
```

skill强制流程：

1. 判断 arch、family、ABI和workspace/launch kind。
2. 只增加一条 KernelSpec，复用现有 launcher/codegen模板。
3. 仅当 kernel需要成为 fallback时修改对应 Python heuristic。
4. 禁止新增 C++ shape policy、raw allocator、kid-range capability判断或顶层 pybind入口。
5. 生成并验证最小 diff，运行 catalog/build/correctness/graph/multi-stream测试。
6. 注释只解释硬件约束和不明显的 invariant；禁止逐行复述和 changelog式注释。

kid band不能只写死在 skill prose中；脚本必须从 catalog读取分配规则，验证 `(arch, family, kid)` 唯一性、区间冲突和 compiled manifest一致性。这样未来新增 fused kid段时无需同步修改多份说明。

## 13. 实施时的审查重点

每个 PR review时额外检查：

```text
rg -n "hip(Malloc|Free|HostMalloc)" csrc/opus_gemm
rg -n "ws_handle|opus_splitk_ws_|SplitkWsRegistry" csrc/opus_gemm
rg -n "workspace_init|prewarm" csrc/opus_gemm aiter/ops/opus aiter/tuned_gemm.py
rg -n "opus_gemm_lookup|heuristic_dispatch" csrc/opus_gemm
rg -n "kid.*>=|kid.*<" aiter/ops/opus csrc/opus_gemm
rg -n "sys.path\.(insert|append)" aiter/ops/opus csrc/opus_gemm
rg -n "from csrc.opus_gemm.opus_gemm_common" aiter/ops/opus
rg -n "torch.empty\(.*uint8" aiter/ops/opus
```

这些搜索不要求最终仓库绝对零命中，例如 deprecation空壳、历史说明、未纳入本期的 #4320 BMM或非 GEMM代码可能保留文本；但每个命中都要按 PR声明的范围解释。runtime读取 `csrc` 本身是允许的，禁止的是 OPUS模块各自修改 `sys.path`或绕过 `_catalog.py` facade直接依赖可变 instance dict；任何新的 split-K raw allocator、kid-range capability判断或 byte workspace都必须被阻止。
