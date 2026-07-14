# 集成 block_scale_gemm_bpreshuffle 到 aiter — 行动方案

> 生成日期：2026-07-13
> 目标 kernel：`/root/workspace/amdgcn_fyz/gcnasm/opus_gemm/block_scale_gemm_bpreshuffle/`
> 目标仓库：`/root/workspace/aiter`
>
> ⚠️ 本文档取代 `opus_mxfp8_gemm_integration_plan.md`。那份文档的目标 kernel
> 是 `mxfp8_gemm_scaled/`（microscale GROUP_K=32），与本 kernel 无关。

## 1. 背景与结论

待集成的 kernel 是 **fp8 e4m3 × fp8 e4m3 → fp32 block-scale GEMM，B 权重预重排（preshuffle）**：
- traits：`BLOCK 256×256×128`，`GROUP 1×128×128`，`BLOCK_SIZE=512`，`T_M=4, T_N=2`，
  MFMA `16×16×128`，`VEC (16,16,4)`。
- B 权重 host 侧预重排成 K-panel 布局：`B'[batch][kt][n][ki] = B[batch][n][kt*BLOCK_K+ki]`，
  形状 `[batch, K/BLOCK_K, N, BLOCK_K]`。
- kernel 以 `stride_b = BLOCK_K` 读 B，K 迭代间跳一整个 panel（`N*BLOCK_K`）。

**关键结论：这个 kernel 是 aiter 现有 `a8w8_scale`（kid 1）的 B-preshuffle 变体。**

对照 `csrc/opus_gemm/opus_gemm_common.py:204` 的 kid 1：

| 维度 | aiter a8w8_scale (kid 1) | 本 kernel bpreshuffle |
|---|---|---|
| traits（BLOCK/GROUP/MFMA/VEC/T） | 512,256,256,128, 4×2, 16×16×128, 1×128×128 | **完全相同** |
| kargs 结构 | `opus_gemm_scale_kargs_gfx950` | **完全相同**（字段一一对应） |
| launcher 签名 | `(XQ, WQ, Y, x_scale, w_scale)` 5-arg | **完全相同** |
| kernel `b_offset` K 步进 | `tile_k * B_K` | **`tile_k * kargs.n * B_K`**（跳整 panel） |
| `stride_b` | `K` | **`B_K`（=128）** |
| B 物理布局 | 原始 `[N,K]` | **host 预 shuffle 成 K-panel** |

**只有两处实质差异**：kernel 里一行 `b_offset` lambda + `stride_b` 常量。其余流水线
（LDS staging / ds_read / MFMA schedule / scale / waitcnt）、SFA/SFB 寻址、traits、kargs
**全部相同**。SFB 不需要重排（scale 按 `(n_group,k_group)` 坐标寻址，与 B 物理布局无关）。

### 决策（拟定）
- **不新建 JIT module**，复用 `module_deepgemm_opus`。
- 新增 `kernel_tag = a8w8_scale_bpreshuffle`，新 kid = **3**（1=a8w8_scale, 2=a8w8 已占用）。
- **Traits / kargs 直接复用 a8w8_scale 的**（`opus_gemm_a8w8_scale_traits_gfx950` +
  `opus_gemm_scale_kargs_gfx950`），不新建 traits/kargs 头文件。
- **新增独立 pipeline header + emit 函数**（不复用 `gen_scale_instance`，原因见 §2 的坑）。
- **新增独立 pybind 入口 + Python `gemm_a8w8_bpreshuffle_opus`**。

---

## 2. ⚠️ 关键坑：`gen_scale_instance` 硬编码 `stride_b = K`

`csrc/opus_gemm/codegen/gen_instances_gfx950.py:762` 里 `gen_scale_instance` 生成的
launcher **硬编码 `kargs.stride_b = K`**。bpreshuffle 需要 `stride_b = B_K`。

因此 **不能直接复用 `gen_scale_instance` 作为 emit 函数**。必须新增一个 emit 函数
（拷贝 `gen_scale_instance`，仅改 `kargs.stride_b = {k.B_K}` 一行），或给 `gen_scale_instance`
增加一个 bpreshuffle 分支。**推荐新增独立函数 `gen_scale_bpreshuffle_instance`**，保持
原函数不动、职责清晰。

> 注：`stride_b_batch` 仍是 `N*K`（per-batch 权重总元素数不变），无需改。

---

## 3. 现有 a8w8_scale 集成机制（参考地图）

### JIT 配置
`aiter/jit/optCompilerConfig.json` 的 `module_deepgemm_opus` 条目：
- `srcs`：`pybind/opus_gemm_pybind.cu` + `opus_gemm/opus_gemm.cu`
- `blob_gen_cmd`：跑 `opus_gemm/gen_instances.py`（触发 codegen）
- per-source flag：`*.device.cu → -D__HIPCC_RTC__`
- HIP flag：`-mllvm --amdgpu-mfma-vgpr-form`（兼容 `__launch_bounds__(512,2)`）

### Codegen 五个 tag map（`codegen/gen_instances_gfx950.py`）
- `PIPELINE_HEADER_MAP`（23 行）：tag → pipeline .cuh
- `TRAITS_HEADER_MAP`（42 行）：tag → traits .cuh
- `KERNEL_FUNC_MAP`（52 行）：tag → kernel 函数名
- `TRAITS_NAME_MAP`（68 行）：tag → traits 模板名
- `KARGS_NAME_MAP`（78 行）：tag → kargs 结构体名
- `register_emit("gfx950", tag, fn)`（1418 行附近）：tag → emit 函数

### 编译子集 S（`gen_instances.py`）
- `INPUT_DTYPE_MAP`（121 行）：tag → (da, db) 输入 dtype
- `SCALE_TAGS`（约 122 行区域，5-arg scale launcher）
- `TAG_TO_LIST`（957 行）：tag → kernels_list
- **kid 无条件加入 S**（1073-1074 行）：a8w8/a8w8_scale 特殊对待，dispatcher 无条件引用其
  符号，必须保证编译。**新 kid 3 也必须这样处理**，否则链接失败。

### C++ dispatcher（`opus_gemm.cu`）
- `opus_dispatch_scale<T>`（40 行）返回 kid 1 的 launcher 符号。
- `opus_gemm()`（102 行）：fp8 + has_scale + gfx950 → 调 `opus_dispatch_scale<fp32_t>(...)`。

### Python（`aiter/ops/opus/`）
- `__init__.py` — arch gate + 导出。
- `gemm_op_a16w16.py` — a16w16 wrapper；`lookup_tuned(..., bpreshuffle=False)`（479 行）
  **已预留 bpreshuffle 维度**。
- `common.py` — tuned-CSV lookup。

---

## 4. 行动步骤（按依赖顺序）

### Step 1 — Pipeline header（device kernel）
新建 `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_scale_bpreshuffle_gfx950.cuh`：
- 拷贝 `opus_gemm_pipeline_a8w8_scale_gfx950.cuh` 全文。
- `#include "opus_gemm_traits_a8w8_scale_gfx950.cuh"`（**复用现有 traits/kargs**，不新建）。
- kernel 函数重命名为 `gemm_a8w8_scale_bpreshuffle_kernel`。
- 唯一 body 改动 —— `b_offset` lambda（对照基线 259-261 行）：
  ```cpp
  // a8w8_scale 原版:
  //   return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K;
  // bpreshuffle 改为:
  auto b_offset = [&](int half_tile_n, int tile_k) {
      return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * kargs.n * T::B_K;
  };
  ```

### Step 2 — 注册 kernel 实例（`opus_gemm_common.py`）
- 新增（对照 203-205 行）：
  ```python
  a8w8_scale_bpreshuffle_kernels_list = {
      3: OpusGemmInstance(512, 256, 256, 128, 4, 2, 16, 16, 128, 16, 16, 4,
                          1, 128, 128, "a8w8_scale_bpreshuffle", ["fp32_t"]),
  }
  ```
  （参数与 kid 1 完全相同，仅 tag 不同。字段顺序见 `OpusGemmInstance` 定义，27 行起。）
- 并入 `kernels_list`（695 行，加 `**a8w8_scale_bpreshuffle_kernels_list`）。

### Step 3 — codegen 接线（`codegen/gen_instances_gfx950.py`）
- 五个 map 加 `"a8w8_scale_bpreshuffle"` 条目：
  - `PIPELINE_HEADER_MAP` → 新 pipeline 文件
  - `TRAITS_HEADER_MAP` → **复用** `gfx950/opus_gemm_traits_a8w8_scale_gfx950.cuh`
  - `KERNEL_FUNC_MAP` → `gemm_a8w8_scale_bpreshuffle_kernel`
  - `TRAITS_NAME_MAP` → **复用** `opus_gemm_a8w8_scale_traits_gfx950`
  - `KARGS_NAME_MAP` → **复用** `opus_gemm_scale_kargs_gfx950`
- 新增 emit 函数 `gen_scale_bpreshuffle_instance`（拷贝 `gen_scale_instance` 688-787 行，
  **仅改 `kargs.stride_b = {k.B_K}`**，762 行那处）。
- `register_emit("gfx950", "a8w8_scale_bpreshuffle", gen_scale_bpreshuffle_instance)`（1418 行附近）。

### Step 4 — 编译子集接线（`gen_instances.py`）
- import `a8w8_scale_bpreshuffle_kernels_list`（29 行附近）。
- `INPUT_DTYPE_MAP` 加 `"a8w8_scale_bpreshuffle": ("fp8_t", "fp8_t")`（121 行）。
- `SCALE_TAGS` 加 `"a8w8_scale_bpreshuffle"`（约 122 行区域，5-arg launcher）。
- `TAG_TO_LIST` 加 `"a8w8_scale_bpreshuffle": a8w8_scale_bpreshuffle_kernels_list`（957 行）。
- **无条件加入 S**：`S |= set(a8w8_scale_bpreshuffle_kernels_list.keys())`（仿 1073 行）。
  同时在 1083 行附近的 tag-filter 分支里也加一份。

### Step 5 — C++ dispatcher（`opus_gemm.cu` + `include/opus_gemm.h`）
- `opus_gemm.cu` 加：
  ```cpp
  template <typename CDataType>
  OpusScaleKernel opus_dispatch_scale_bpreshuffle(int M, int N, int K) {
    return <kid-3 launcher 符号>;  // build 一次后到 blob/impl 确认真实 k.name 回填
  }
  ```
  新入口 `opus_gemm_a8w8_bpreshuffle(XQ, WQ, Y, x_scale, w_scale)`：
  内部 `AITER_CHECK` gfx950 + fp8 输入 + fp32 输出，再调
  `opus_dispatch_scale_bpreshuffle<fp32_t>(M,N,K)(XQ,WQ,Y,x_scale,w_scale)`。
- `include/opus_gemm.h` 声明该入口。

> 备注：launcher 符号名由 `OpusGemmInstance.name`（`opus_gemm_common.py:70`）拼出。
> kid 3 与 kid 1 除 tag 外参数相同，name 会带 tag 区分。build 一次后到
> `aiter/jit/build/module_deepgemm_opus/blob/impl/` 确认真实符号名再回填。

### Step 6 — Pybind（`rocm_ops.hpp` + `opus_gemm_pybind.cu`）
- `rocm_ops.hpp` 加宏 `OPUS_GEMM_A8W8_BPRESHUFFLE_PYBIND`（对照 273 行 `OPUS_GEMM_PYBIND`）：
  ```cpp
  #define OPUS_GEMM_A8W8_BPRESHUFFLE_PYBIND     \
      m.def("opus_gemm_a8w8_bpreshuffle",       \
            &opus_gemm_a8w8_bpreshuffle,        \
            "opus_gemm_a8w8_bpreshuffle",       \
            py::arg("XQ"), py::arg("WQ"), py::arg("Y"), \
            py::arg("x_scale") = std::nullopt,  \
            py::arg("w_scale") = std::nullopt);
  ```
- `opus_gemm_pybind.cu` 的 `PYBIND11_MODULE` 内加 `OPUS_GEMM_A8W8_BPRESHUFFLE_PYBIND;`

### Step 7 — Python wrapper（新建 `aiter/ops/opus/gemm_op_a8w8_bpreshuffle.py`）
**开放决策：B 的 shuffle 由谁执行？** 推荐 Python 侧提供 shuffle helper（灵活、权重可离线
预重排复用；推理时权重固定只 shuffle 一次）。对照 host.cc:96 `shuffle_b()` 用 torch 实现：
```python
def shuffle_b_for_bpreshuffle(B, block_k=128):
    # B: [batch, N, K] -> [batch, K//block_k, N, block_k] -> flatten 回 [batch, N, K] 视图
    batch, N, K = B.shape
    return (B.view(batch, N, K // block_k, block_k)
             .permute(0, 2, 1, 3)          # [batch, kt, N, block_k]
             .contiguous()
             .view(batch, N, K))           # 物理已是 panel 序，逻辑形状复原给 kernel
```
- `@compile_ops("module_deepgemm_opus", fc_name="opus_gemm_a8w8_bpreshuffle", develop=True)` stub。
- 公开 `gemm_a8w8_bpreshuffle_opus(A, B, x_scale, w_scale, out=None, *, preshuffled=False)`：
  - `preshuffled=False` 时内部调 `shuffle_b_for_bpreshuffle`；`True` 时假定 B 已重排。
  - 校验 A 为 fp8/3D/contiguous `(M*K,K,1)`；WQ `stride_b_batch=N*K`（物理 panel，逻辑形状 `[batch,N,K]`）。
  - 校验 scale：SFA `[batch, num_groups_k, num_groups_m]`、SFB `[batch, num_groups_n, num_groups_k]`（fp32）。
  - 分配/复用 fp32 输出 Y `[batch, M, N]`。
- `aiter/ops/opus/__init__.py`：arch_ok 分支导出 `gemm_a8w8_bpreshuffle_opus` + unsupported-arch stub。

### Step 8 — 测试 + 构建验证（新建 `op_tests/test_opus_a8w8_bpreshuffle_gemm.py`）
- CPU 参考吃**原始 B**（`[N,K]`），Python 侧 shuffle 后送 kernel —— shuffle/寻址错误直接暴露。
  逻辑对照 host.cc `gemm_ref`（66 行）。
- arch-gate 到 gfx950（非 gfx950 skip）。
- argparse 驱动 config sweep，遵循 `aiter/test_common.py` 的 `checkAllclose` / `@perftest()`。
- 构建验证（需 gfx950 硬件）：
  ```bash
  AITER_REBUILD=2 GPU_ARCHS=gfx950 python3 op_tests/test_opus_a8w8_bpreshuffle_gemm.py
  ```
  首次触发 codegen + JIT build；build 后到 `blob/impl/` 确认 kid 3 真实符号名，回填 Step 5。

---

## 5. 需要改动的文件清单

| 步骤 | 文件 | 动作 |
|---|---|---|
| 1 | `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_scale_bpreshuffle_gfx950.cuh` | 新建（拷 a8w8_scale，改 `b_offset` 一行 + 改函数名，复用 traits/kargs） |
| 2 | `csrc/opus_gemm/opus_gemm_common.py` | 加 `a8w8_scale_bpreshuffle_kernels_list`（kid 3），并入 `kernels_list` |
| 3 | `csrc/opus_gemm/codegen/gen_instances_gfx950.py` | 五个 map 加条目 + 新 emit `gen_scale_bpreshuffle_instance` + `register_emit` |
| 4 | `csrc/opus_gemm/gen_instances.py` | import list、`INPUT_DTYPE_MAP`、`SCALE_TAGS`、`TAG_TO_LIST`、无条件加入 S |
| 5 | `csrc/opus_gemm/opus_gemm.cu` | 加 `opus_dispatch_scale_bpreshuffle` + `opus_gemm_a8w8_bpreshuffle` 入口 |
| 5 | `csrc/opus_gemm/include/opus_gemm.h` | 声明新入口 |
| 6 | `csrc/include/rocm_ops.hpp` | 加 `OPUS_GEMM_A8W8_BPRESHUFFLE_PYBIND` 宏 |
| 6 | `csrc/pybind/opus_gemm_pybind.cu` | 调用新宏 |
| 7 | `aiter/ops/opus/gemm_op_a8w8_bpreshuffle.py` | 新建 wrapper + shuffle helper |
| 7 | `aiter/ops/opus/__init__.py` | 导出 `gemm_a8w8_bpreshuffle_opus` |
| 8 | `op_tests/test_opus_a8w8_bpreshuffle_gemm.py` | 新建测试 |

---

## 6. 关键坑位提醒

1. **`gen_scale_instance` 硬编码 `stride_b=K`**（§2）——必须新 emit 或加分支改成 `B_K`，
   否则数值错误（B 按错误 stride 寻址）。这是本集成最容易踩的坑。
2. **kid 3 必须无条件加入编译集 S**（Step 4），否则 dispatcher 引用 launcher 符号链接失败。
3. **B 必须 preshuffle**：kernel 假定 B 已是 K-panel 布局。Python 侧 shuffle 或调用方自证。
4. **SFB 不重排**：scale 按 group 坐标寻址，与 B 物理布局无关（host.cc 已验证）。
5. **fp32-only 输出 + gfx950-only**：与 a8w8_scale 一致，入口处 `AITER_CHECK`。
6. **launcher 符号名 build 后回填**（Step 5）。
7. **CPU 参考吃原始 B**：验证时 shuffle 错误直接暴露；多 panel（K>BLOCK_K）、多 batch 必测。

---

## 7. 建议实施顺序

先做 **Step 1 → 2 → 3 → 4**（header + 注册 + codegen + 编译集），决定能否 build 出 kid 3；
再 **Step 5 → 6 → 7**（C++/pybind/Python 打通调用链）；最后 **Step 8**（测试 + 验证，需 gfx950）。

---

## 8. 待确认决策点

1. **B shuffle 归属**（Step 7）：Python 侧 shuffle（推荐）/ 调用方传已 shuffle 的 WQ / 两者都支持。
   文档现按"Python 侧提供 helper + `preshuffled` flag 两者兼顾"拟定，可按需收窄。
2. **是否接入 tuned-CSV 的 bpreshuffle 维度**：`gemm_op_a16w16.py:479` 已预留 `bpreshuffle` 参数；
   本次是否也走 CSV lookup，还是先做独立入口、后续再统一到 tuned 路径。
