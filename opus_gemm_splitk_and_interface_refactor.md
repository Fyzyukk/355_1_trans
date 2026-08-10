# OPUS GEMM Split-K 与接口重构设计

> 状态：设计提案，尚未实施  
> 基线：当前工作区 `35696023`  
> 参考实现：[ROCm/aiter PR #4246 - gfx1250 opus gemm splitk fuse](https://github.com/ROCm/aiter/pull/4246)  
> 参考版本：`86d1eba4c2b9de9649fe36228b3b8bba578ab0bd`

## 1. 任务拆分

这项工作应明确拆成两个大任务。两者共享 kernel metadata 和低层 launcher ABI，但交付物、风险和验收标准不同，不应放在一个大 PR 中完成。

### 任务 A：实现并统一 split-K

任务 A 包含两个层次：

1. 参考 PR #4246，为 gfx1250 接入 fused split-K kernel。
2. 将 gfx1250、gfx950、gfx942 的 split-K workspace 全部改成由 Python/PyTorch 分配，并以 Tensor/直接指针传给 C++ launcher。

任务 A 负责的是 kernel 功能、workspace 契约和正确性，不负责 tuned CSV 的选择策略重构。

### 任务 B：重构 OPUS GEMM 接口

任务 B 将当前混合了 a16w16、a8w8、lookup、heuristic、workspace registry 和兼容逻辑的入口拆开：

- Python 负责输入规范化、tuned lookup、heuristic、split-K 解析、output/workspace 分配和 fallback。
- C++ 仅负责运行时安全检查、arch 校验、`kid -> launcher` dispatch、kargs 构造和 kernel launch。
- a16w16、a8w8、blockscale 等 kernel family 使用各自的低层 ABI，不再共用一个 mega entry。

### 两个任务的公共前置

两个任务都依赖一份统一的 `KernelSpec`。不能让任务 A 在 `gemm_op_a16w16.py` 中增加一套 kid-range/workspace 特判，同时任务 B 又重新设计一套 metadata，否则会制造第二轮重构。

建议依赖关系如下：

```text
公共前置：KernelSpec + family-specific launch ABI
                  │
          ┌───────┴────────┐
          ▼                ▼
任务 A：split-K       任务 B：Python selector
          │                │
          └───────┬────────┘
                  ▼
       删除旧 C++ policy/registry/mega entry
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
- gfx1250 fused split-K 的 `split_k`、cluster geometry 和 workspace dtype 是编译时属性，不能由运行时 `splitK` 参数或 kid 范围猜测。

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

## 4. 公共数据模型

### 4.1 新增 KernelSpec

建议新增 `aiter/ops/opus/_kernel_catalog.py`。该文件必须是纯 Python 数据模块，不导入 `torch`、`aiter` 顶层包或 JIT extension，使 runtime 和 codegen 都可以消费同一事实源。

```python
@dataclass(frozen=True)
class KernelSpec:
    arch: str
    family: str
    kid: int
    kernel_tag: str

    block_m: int
    block_n: int
    block_k: int

    input_dtypes: tuple[str, ...]
    output_dtypes: tuple[str, ...]
    acc_dtype: str
    workspace_dtype: str | None

    a_layout: str
    b_layout: str
    output_layout: str
    workspace_layout: str | None

    split_k_mode: str          # none / runtime / static
    static_split_k: int | None
    launch_kind: str           # direct / two_stage / fused

    supports_bias: bool
    bias_dtypes: tuple[str, ...]
    supports_batch: bool
    supports_m_tail: bool
    supports_n_tail: bool
    supports_k_tail: bool

    cluster_n: int = 1
    max_tensor_bytes: int | None = None
```

这里的 dtype 使用稳定字符串枚举，例如 `bf16`、`fp32`，runtime 再映射到 `torch.dtype`，codegen 映射到 C++ 类型。不要让 catalog 依赖 PyTorch 初始化。

`kid` 的逻辑身份是 `(arch, family, kid)`。数值区间只可以作为 ID 分配规则，不能再作为 runtime capability API。

### 4.2 动态选择结果

```python
@dataclass(frozen=True)
class LaunchConfig:
    spec: KernelSpec
    effective_split_k: int
    source: str                # explicit / tuned / heuristic
```

静态属性和动态选择结果必须分开：

- fused split-K 的 `static_split_k` 来自 `KernelSpec`。
- two-stage runtime split-K 的 `effective_split_k` 来自 tuner/heuristic，再经过统一 clamp。
- 非 split-K kernel 的 `effective_split_k` 固定为 1。

### 4.3 WorkspacePlan

新增 `aiter/ops/opus/_workspace.py`：

```python
@dataclass(frozen=True)
class WorkspacePlan:
    shape: tuple[int, ...]
    dtype: str
    required_numel: int
    alignment: int
```

必须支持至少两种 layout：

1. two-stage split-K：

   ```text
   [batch, effective_split_k, padded_M, padded_N]
   ```

2. gfx1250 fused split-K：

   ```text
   [batch, tiles_M, tiles_N, static_split_k - 1, block_M, block_N]
   ```

   fused kernel 的最后一个 split 保留自身 fp32 accumulator，只需要发布前 `split_k - 1` 个 partial。实现可以分配更大的兼容 buffer，但 planner 和校验应表达实际最小需求。

workspace 必须按元素 dtype 分配，不按猜测的 byte size 分配：

```python
workspace = torch.empty(plan.shape, dtype=to_torch_dtype(plan.dtype), device=A.device)
```

默认不使用全局 `lru_cache`：PyTorch caching allocator 已负责 eager 重用；graph capture 使用 graph-private pool；每次调用独立 Tensor 也能避免双 stream 共用 scratch。可以保留内部高级参数 `workspace=`，让明确管理生命周期的调用方复用。

## 5. 任务 A：split-K 修改方案

### 5.1 gfx1250 fused split-K 算法

参考 PR #4246，但只借鉴其 kernel/launcher 思路，不直接复制整个 PR head。

目标行为：

1. `SplitK` 是每个 fused kid 的编译时属性。
2. cluster x 维对应 K split，cluster y 维可用于 N-tile peer multicast。
3. 每个 split 处理一段按 `block_k` tile 均衡划分的 K 区间。
4. 非最后 split 将 partial 写入 `DataWs` workspace。
5. 所有 split 通过 cluster barrier 保证 partial 对最后 split 可见。
6. 最后 split 保留自己的 fp32 accumulator，读回前 `SplitK - 1` 个 partial，以 fp32 累加，融合 bias，最后只做一次到 bf16/fp32 output 的转换。
7. workspace 从 Python 传入 `ptr_ws`，kernel 不读取 `ws_handle->ptr`。

需要编码到 `KernelSpec` 的约束：

- `static_split_k <= ceil_div(K, block_k)`，避免空 split。
- `static_split_k * cluster_n <= 16`。
- round 1 若只支持 `batch == 1`，必须显式声明 `supports_batch=False`。
- 若 kernel 没有 N-tail mask，则要求 `N % block_n == 0`。
- bias 支持的 shape/dtype 必须显式声明。
- `workspace_dtype` 可以是 bf16 或 fp32，与最终 `output_dtype` 无关。

### 5.2 不照搬 PR #4246 的部分

- 不在 Python 通过 `20000 <= kid < 30000` 判断 workspace policy。
- 不从 runtime 修改 `sys.path` 反向 import `csrc/opus_gemm/opus_gemm_common.py`。
- 不使用 `lru_cache(maxsize=None)` 永久缓存 exact-shape workspace。
- 不保留 fallback 中的临时 `hipMalloc + hipDeviceSynchronize + hipFree`。
- 不把 fused 和 two-stage split-K 共用一个含糊的 `D_C` 类型参数。
- 不允许 launcher 只取 `workspace.data_ptr()` 而不验证 dtype、device、contiguous 和容量。

### 5.3 任务 A 的文件级修改

#### Python 和公共 metadata

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `aiter/ops/opus/_kernel_catalog.py`（新增） | gfx1250 a16w16 catalog | 注册 fused kid；填写 tile、static split-K、cluster_n、workspace dtype/layout、tail/bias/batch capability。不要用 kid range推导属性。 |
| `aiter/ops/opus/_workspace.py`（新增） | `make_workspace_plan()`、`validate_workspace()` | 实现 two-stage 和 fused 两种 workspace 公式；支持 caller-provided workspace；检查 device/dtype/layout/numel/alignment。 |
| `aiter/ops/opus/gemm_op_a16w16.py` | `opus_gemm_a16w16_tune()` 和最终 launch 前 | 暂时通过 `KernelSpec` 获取 workspace plan并分配；给 raw binding 增加 `workspace` 参数。任务 B 完成后，这些逻辑迁入 selector/launcher wrapper，此文件只编排调用。 |

#### gfx1250 kernel 和 traits

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_fuse_gfx1250.cuh`（新增） | 新 fused pipeline | 实现 compile-time `SplitK`、cluster launch、balanced K partition、partial publish、cluster barrier、last-split fused reduce 和最终 output store。只保留解释硬件约束/invariant 的注释。 |
| `csrc/opus_gemm/include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh` | workspace handle、kargs、traits | 新增/调整 fused kargs；将 `ws_handle` 改成 `void *ptr_ws`；traits 中明确 `D_ACC`、`D_WS`、`D_OUT`，不再用 `D_C` 表示 workspace。 |
| `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_cluster_tdm_splitk_ws_gfx1250.cuh` | workspace 地址构造 | two-stage 版本也改成直接使用 `kargs.ptr_ws`，删除 handle 解引用。保持数值算法不变。 |
| `csrc/opus_gemm/include/gfx1250/opus_gemm_pipeline_a16w16_clusterlaunch_tdm_splitk_ws_gfx1250.cuh` | workspace 地址构造 | 同上；clusterlaunch two-stage 与 fused 使用一致的 direct-pointer ABI。 |
| `csrc/opus_gemm/include/gfx1250/splitk_reduce_gfx1250.cuh` | reduce kernel signature/template | 输入改为 `const void *ws_ptr`；模板参数显式命名 `D_WS` 和 `D_OUT`；支持所需的 bf16/fp32 workspace 与 bf16/fp32 output 组合。 |
| `csrc/opus_gemm/include/gfx1250/splitk_reduce_launch_gfx1250.cuh`（新增，可仅供 two-stage） | host reduce dispatcher | 将 runtime split-K 映射到 compile-time unrolled reduce instance。fused kernel不经过该文件，不能把两种 launch kind 混淆。 |

#### codegen 和 catalog 适配

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `csrc/opus_gemm/codegen/common.py` | kernel tag/生成器注册 | 增加 fused tag 和必要的生成信息；避免在多个生成器重复写 tag/dtype 映射。 |
| `csrc/opus_gemm/codegen/gen_instances_gfx1250.py` | 新 `gen_splitk_fuse_instance()`；现有 two-stage generator | 生成统一 launcher 签名；接收 workspace Tensor；验证容量/dtype；构造 direct-pointer kargs；生成具体 `__cluster_dims__` launch；删除所有内部 allocator。 |
| `csrc/opus_gemm/opus_gemm_common.py` | `OpusGemmInstance` 和 gfx1250 list | 迁移期间从 `_kernel_catalog.py` re-export/适配；最终只保留 codegen/tuner helper，不再作为 runtime metadata。不要新增第二份 fused kid 表。 |
| `csrc/opus_gemm/gen_instances.py` | generator route、manifest、device TU | 注册 fused generator；将 fused kid加入 subset compile；生成所需显式实例。不要为 fused kernel新增 runtime shape lookup。 |
| `csrc/opus_gemm/opus_gemm_tune.py` | candidate filter、splitK sweep、launch 调用 | fused kid 的 split-K来自 spec，不再 sweep runtime `splitK`；过滤 cluster、K steps、tail、batch限制；记录 workspace dtype/static split-K。 |
| `csrc/gemm_a16w16/gemm_a16w16_tune.py` | OPUS backend 调用 | 继续调用兼容 Python wrapper，不直接调用缺少 workspace 的旧 C++ binding。 |

#### 测试和文档

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `op_tests/test_opus_a16w16_gemm.py` | shape matrix | 增加 explicit fused kid、tuned fused kid、bf16/fp32 output、bias 和 tail/constraint cases。 |
| `op_tests/test_opus_splitk_gfx1250.py`（新增） | 独立 split-K 回归 | 对比 fp32 reference；覆盖每种 static split-K、bf16/fp32 workspace、K-tail、非法 cluster/K 配置。 |
| `op_tests/test_opus_workspace.py`（新增） | workspace contract | 覆盖刚好大小、少 1 element、错误 dtype/device、noncontiguous、caller reuse、双 stream 并发。 |
| `aiter/ops/opus/README.md` | split-K/graph 部分 | 文档改为 Tensor workspace；删除 registry/prewarm 使用说明；解释 fused 与 two-stage 的差异。 |
| `csrc/opus_gemm/README.md` | C++ layout 和 split-K 部分 | 更新 direct-pointer ABI、生成器和 kernel 文件图。 |

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

### 6.2 Python selector

新增 `aiter/ops/opus/_selector.py`，固定执行顺序：

```text
1. normalize shape/dtype/layout
2. explicit kid（若给定）
3. runtime tuned CSV lookup
4. Python heuristic
5. validate spec capability + compiled-kid set
6. resolve effective split-K
7. return LaunchConfig
```

规则：

- explicit kid 不兼容时直接抛错，不静默换 kernel。
- tuned row 引用了未编译 kid时，默认 warning 后 heuristic；strict 模式直接失败。
- heuristic 只能从当前 build 的 compiled-kid set 中选择。
- 不支持的 shape/layout 应交给 framework fallback，而不是让 C++再次选择。

### 6.3 tuned lookup

新增中立 loader `aiter/ops/gemm_tune_lookup.py`，由 `tuned_gemm.py` 和 OPUS selector 共用。runtime key 至少包含：

```text
gfx, cu_num, batch, M, N, K,
input dtype, output dtype,
bias flag/dtype, A/B layout,
scale mode, bpreshuffle, kernel family
```

build-time 和 runtime 的 CSV 用途必须区分：

- build-time `--tune_files` 可以取多个 CSV 的 kid 并集，用于 subset compile。
- runtime 只能查询 framework 当前明确选择的配置文件，不能默认 glob 所有 model CSV 后取全局最小 `us`。

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

gfx942 当前部分 heuristic 直接关联 launcher symbol；迁移时先建立 symbol 到 kid 的固定映射，再删除 C++ header。

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

### 6.6 任务 B 的文件级修改

#### Python 文件

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `aiter/ops/opus/_kernel_catalog.py`（新增） | 全文件 | 单一 kernel metadata；提供按 arch/family/kid 查询；不导入 torch/JIT。 |
| `aiter/ops/opus/_selector.py`（新增） | 全文件 | explicit/tuned/heuristic 选择、capability 校验、effective split-K 解析、compiled set过滤。 |
| `aiter/ops/opus/_workspace.py`（新增） | 全文件 | workspace planner、torch dtype映射、分配和 user workspace校验；不做无界缓存。 |
| `aiter/ops/opus/heuristics/*.py`（新增） | 各 arch selector | 等价迁移 C++ heuristic，只返回 kid/split-K，不返回 launcher symbol。 |
| `aiter/ops/gemm_tune_lookup.py`（新增） | 全文件 | 统一 tuned CSV loader和完整 key；提供按 `libtype/family` 查询。 |
| `aiter/ops/opus/gemm_op_a16w16.py` | module doc、raw binding、`opus_gemm_a16w16_tune()`、`gemm_a16w16_opus()`、workspace init | 删除 `_opus_gemm_bf16_dispatch`；高层入口改为 normalize -> select -> allocate -> launch；raw binding改名并增加 workspace；删除 workspace init；保留 tune兼容 wrapper。 |
| `aiter/ops/opus/gemm_op_a8w8.py`（新增） | 全文件 | 从 mega entry中接管 a8w8/a8w8-scale接口；不同 scale/layout family 使用不同 launcher。 |
| `aiter/ops/opus/common.py` | CSV loader、mono kid表 | 首先改成对统一 tune loader/catalog 的 compatibility re-export；所有调用迁完后删除。mono tile capability 从 KernelSpec读取。 |
| `aiter/ops/opus/__init__.py` | import/stub/`__all__` | 导出新的 family public API；移除 `opus_gemm_workspace_init`；旧 tune wrapper 标记 deprecated。 |
| `aiter/ops/opus/_arch.py` | arch probe 文档/接口 | 保留探测功能；不要在包级别假设所有 OPUS family 支持同一 arch，支持集合由 family catalog决定。 |
| `aiter/tuned_gemm.py` | 顶部 workspace prewarm；`opus_gemm()` backend | 删除 `_opus_needs_ws_prewarm`、capture stream和 warmed signature；复用统一 tune loader；直接调用高层 OPUS API或 selector结果。 |
| `aiter/ops/deepgemm.py` | `opus_gemm_a16w16_tune` shim | 继续转发到 Python compatibility wrapper；更新 warning 和新参数；下一 release删除。 |

#### C++ binding 和入口

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `csrc/opus_gemm/include/opus_gemm.h` | 三个现有声明 | 用 family-specific launch声明替换 `opus_gemm()` 和 C++ tune接口；增加 optional workspace；删除 `opus_gemm_workspace_init()`。 |
| `csrc/include/rocm_ops.hpp` | `OPUS_GEMM_*_PYBIND` 宏 | 注册新的 a16w16/a8w8 launch binding和 workspace参数；删除 workspace init宏；兼容名放在 Python，不再扩大 C++ ABI。 |
| `csrc/pybind/opus_gemm_pybind.cu` | `PYBIND11_MODULE` | 注册新的 family bindings；删除 `OPUS_GEMM_WORKSPACE_INIT_PYBIND`。 |
| `csrc/opus_gemm/opus_gemm.cu` | arch routers、`opus_gemm()`、tune入口、kid范围函数、registry | 删除 auto dispatch、mega entry、所有 kid-range policy和 workspace registry；保留小型 family launcher和 arch/kid dispatch；不得调用 allocator。 |
| `csrc/opus_gemm/include/opus_gemm_arch.cuh` | arch probe/shared types | 保留 arch探测；将真正跨 arch 的 launcher entry/metadata类型放在这里或单独的 `opus_gemm_dispatch.cuh`。 |
| `csrc/opus_gemm/include/opus_gemm_common.cuh` | traits umbrella | 删除 workspace handle相关依赖；只聚合 launcher构造需要的 traits/kargs。 |

#### 每个 arch 的 dispatch 和 kernel ABI

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `csrc/opus_gemm/include/gfx942/opus_gemm_arch_gfx942.cuh` | runtime lookup和 tune lookup | 删除 shape lookup和 heuristic fallback；只保留 generated kid dispatch。 |
| `csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh` | runtime lookup、kid range、4GiB heuristic guard | 4GiB能力迁入 KernelSpec/Python selector；只保留 generated kid dispatch。 |
| `csrc/opus_gemm/include/gfx1250/opus_gemm_arch_gfx1250.cuh` | runtime lookup和 heuristic | 删除 shape/heuristic；只保留 fused/two-stage kid dispatch。 |
| `csrc/opus_gemm/include/gfx942/opus_gemm_heuristic_dispatch_gfx942.cuh` | 全文件 | Python heuristic golden对齐完成后删除。需要的 function-pointer typedef先移到共享 dispatch header。 |
| `csrc/opus_gemm/include/gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh` | 全文件 | 同上。 |
| `csrc/opus_gemm/include/gfx1250/opus_gemm_heuristic_dispatch_gfx1250.cuh` | 全文件 | 同上。 |
| `csrc/opus_gemm/include/gfx942/opus_gemm_traits_a16w16.cuh` | `opus_splitk_ws_handle`、`opus_gemm_splitk_kargs` | 删除 handle/helper；kargs直接保存 `ptr_ws`。 |
| `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a16w16_gfx950.cuh` | flatmm split-K handle/kargs/traits | 删除 handle；区分 `D_ACC/D_WS/D_OUT`；kargs直接保存 workspace pointer。 |
| `csrc/opus_gemm/include/gfx1250/opus_gemm_traits_a16w16_gfx1250.cuh` | cluster TDM kargs/traits | 与任务 A 相同，统一 direct-pointer和明确 dtype。 |
| `csrc/opus_gemm/include/gfx942/opus_gemm_pipeline_a16w16_em3en4_lds1_pgr2_sk.cuh`、`opus_gemm_pipeline_a16w16_kbuf1.cuh`、`opus_gemm_pipeline_a16w16_kbuf2v.cuh`、`opus_gemm_pipeline_a16w16_kbuf2v_bk128.cuh`、`opus_gemm_pipeline_a16w16_quad_mfma32_kbuf1.cuh` | workspace访问 | 将 `opus_splitk_ws_ptr(ws_handle)` 或直接 handle访问改成对 `ptr_ws` 的显式类型转换；不改变数值流水线。 |
| `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a16w16_flatmm_splitk_gfx950.cuh` | workspace访问 | direct pointer；保持 main写 fp32 partial的语义。 |
| `csrc/opus_gemm/include/gfx942/splitk_reduce_gfx942.cuh` | reduce输入 | direct pointer；显式 workspace/output dtype；统一 bias shape/stride语义。 |
| `csrc/opus_gemm/include/gfx950/splitk_reduce_gfx950.cuh` | reduce输入 | 同上。 |
| `csrc/opus_gemm/include/gfx1250/splitk_reduce_gfx1250.cuh` | reduce输入 | 同上；同时支持任务 A 所需的 bf16/fp32 workspace实例。 |

只有实际引用 `ws_handle` 的 pipeline 需要改；非 split-K kernel 不做机械重写，以保持 diff 可审查。

#### codegen、build 和 tuner

| 文件 | 修改位置 | 修改内容 |
|---|---|---|
| `csrc/opus_gemm/gen_instances.py` | `generate_lookup`、tune lookup、main | 停止生成 `opus_gemm_lookup.h`；把 `opus_gemm_a16w16_tune_lookup.h` 重命名为 `opus_gemm_a16w16_kid_dispatch.h`；生成 dispatch metadata和 compiled-kid manifest；CSV 仅用于 subset compile。 |
| `csrc/opus_gemm/opus_gemm_common.py` | `OpusGemmInstance`、kid集合、capability helper | metadata迁到统一 catalog；迁移期间只做 adapter/re-export；删除重复的 kid range/capability集合。 |
| `csrc/opus_gemm/codegen/gen_instances_gfx942.py` | split-K launcher | 接收 workspace Tensor并验证；直接传 ptr；删除内部 `hipMalloc`/grow逻辑。 |
| `csrc/opus_gemm/codegen/gen_instances_gfx950.py` | flatmm split-K launcher | 同上；删除 capture/prewarm错误路径和 registry调用。 |
| `csrc/opus_gemm/codegen/gen_instances_gfx1250.py` | two-stage/fused launcher | 与任务 A 一致；所有路径都禁止内部 allocator。 |
| `csrc/opus_gemm/codegen/common.py` | 通用 launcher template | 统一参数顺序、workspace校验和错误信息；不在每个 arch重复模板。 |
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
| `op_tests/test_opus_dispatch.py`（新增） | selector单元测试 | 不需要 GPU kernel即可验证完整 key、优先级、stale CSV、unsupported kid和heuristic filter。 |
| `op_tests/test_opus_graph.py`（新增） | HIP graph/compile | eager、capture/replay、`torch.compile`、同 shape双 stream并发。 |

## 7. dtype 与 layout 契约

### 7.1 dtype

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

### 7.2 layout

a16w16 的逻辑布局固定为：

```text
A         [batch, M, K]       K-contiguous
B         [batch, N, K]       K-contiguous，计算 A @ B^T
Y         [batch, M, N]       contiguous
bias      [N] 或 [batch, N]
workspace 由 WorkspacePlan定义
```

每个 spec需要声明 tail、batch和 stride capability。对于 transpose、expand/broadcast、负 stride 和 sliced view，高层 wrapper只能选择以下一种行为：

1. kernel原生支持；
2. 明确 materialize contiguous；
3. 抛出清晰错误。

禁止依赖 launcher中硬编码 stride 后继续接受任意 view。

## 8. 实施 PR 顺序

### PR0：公共契约，不改变 kernel 数值行为

- 新增 `KernelSpec`、`LaunchConfig`、`WorkspacePlan`。
- 生成/暴露 compiled-kid set。
- 增加带 optional workspace 的 a16w16 low-level launch ABI。
- 旧入口暂时转发到新入口。

### PR-A1：gfx1250 fused split-K

- 接入 fused pipeline、traits、codegen和静态配置。
- 只启用 explicit kid/tuner测试，不立即改变默认 heuristic。
- 完成数值、边界和性能验证后再允许 tuned CSV选择。

### PR-A2：所有 split-K workspace Torch 化

- gfx1250 two-stage/fused -> gfx950 -> gfx942 顺序迁移。
- 删除 launcher内部 allocator、handle和 registry。
- 完成 graph与双 stream测试。

### PR-B1：Python selector 和统一 tuned lookup

- 等价迁移三个 arch heuristic。
- 新旧 selector跑 golden differential test。
- 高层 API 全部走 Python `select -> allocate -> launch`。

### PR-B2：删除旧 C++ policy 和拆分 family API

- 删除 C++ shape lookup、heuristic、kid-range policy和 mega entry。
- 删除生成的 runtime lookup。
- a16w16/a8w8/blockscale使用独立 binding。
- 保留一轮 Python兼容 shim。

### PR-B3：清理兼容层和文档

- 下一 release删除旧 tune alias和 `common.py` shim。
- 更新 tuner、文档、示例和 release note。

## 9. 验收矩阵

### split-K 功能

- gfx1250 fused 与 two-stage；gfx950/gfx942 existing split-K。
- `split_k=1`、runtime `>1`、static fused、最大值和 clamp边界。
- bf16/fp32 workspace；bf16/fp32 output。
- bias none/bf16/fp32，以及 `[N]`、`[batch,N]`。
- M/N/K tail和明确不支持的边界。
- CPU fp32或可靠 GEMM reference correctness。
- tuned shape性能不应出现不可解释的明显回退。

### workspace 安全

- 正好容量、少 1 element、错误 dtype、错误 device、noncontiguous。
- eager、HIP graph capture/replay、`torch.compile`。
- 两个 stream并发运行相同 shape和不同 shape。
- caller-provided workspace生命周期测试。
- 多次不同 shape调用后显存占用不因无界 Python cache持续增长。

### dispatch 和 build

- explicit、tuned hit、heuristic miss、fallback。
- gfx942/gfx950/gfx1250单 arch和多 arch build。
- subset compile下 heuristic不能返回未编译 kid。
- stale CSV产生明确 warning/error。
- 生成文件二次生成 diff为空。

## 10. Definition of Done

任务 A 完成必须满足：

- gfx1250 fused split-K正确且性能经过基准验证。
- gfx942/gfx950/gfx1250 split-K全部接收外部分配的 workspace Tensor。
- OPUS GEMM launch路径不存在 `hipMalloc`、`hipFree`、`hipHostMalloc` 或 workspace registry。
- workspace dtype/容量错误在 launch前被拒绝，不允许越界或 hang。
- graph capture和同 shape双 stream并发通过。

任务 B 完成必须满足：

- runtime tuned lookup和 heuristic只在 Python执行一次。
- C++ 不再包含 `(M,N,K)` lookup和 framework fallback policy。
- C++ 只按 arch/family/kid/output dtype dispatch。
- a16w16和a8w8不再共享 mega entry。
- codegen只生成 kid dispatch，不生成 runtime shape lookup。
- dtype/layout/workspace capability来自统一 KernelSpec。
- 旧公开 Python API在约定的兼容周期内仍可用，并发出明确 deprecation warning。

## 11. 实施时的审查重点

每个 PR review时额外检查：

```text
rg -n "hip(Malloc|Free|HostMalloc)" csrc/opus_gemm
rg -n "ws_handle|workspace_init|prewarm" csrc/opus_gemm aiter/ops/opus aiter/tuned_gemm.py
rg -n "opus_gemm_lookup|heuristic_dispatch" csrc/opus_gemm
rg -n "kid.*>=|kid.*<" aiter/ops/opus csrc/opus_gemm
```

这些搜索不要求最终仓库绝对零命中，例如历史说明或非 GEMM 代码可能存在相关文本，但任何新的 split-K workspace policy、allocator或 kid-range capability判断都必须被阻止。
