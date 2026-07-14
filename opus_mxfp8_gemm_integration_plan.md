# 集成 MXFP8 Opus GEMM 到 aiter — 行动方案

> 生成日期：2026-07-12
> 目标 kernel：`/root/workspace/amdgcn_fyz/gcnasm/opus_gemm/mxfp8_gemm_scaled/`
> 目标仓库：`/root/workspace/aiter`

## 1. 背景与结论

待集成的 kernel 是 **fp8 e4m3 × fp8 e4m3 → fp32，MXFP8 microscale**：
- `GROUP_K = 32`，每 32 个 K 元素一个 E8M0（`unsigned char`）scale。
- 16×16×128 硬件 scaled MFMA，单条 MFMA 吃满 `B_K=128`（`E_K=1`），4 个 32-K microscale 块由硬件按 lane-group（`lane/W_M`，共 `WARP_SIZE/W_M=4` 组）内部路由。
- `__launch_bounds__(BLOCK_SIZE=512, 2)`，`T_M=4, T_N=2`。

aiter 里**最接近的现成路径是 `a8w8_scale`（kid 1）**：同样 gfx950-only、16×16×128 scaled MFMA、fp32 输出、5-arg launcher `(XQ, WQ, Y, x_scale, w_scale)`，kargs 布局完全一致。

**唯一实质差异**：`a8w8_scale` 是 block-scale `GROUP_K=128`，本 kernel 是 microscale `GROUP_K=32`。

### 决策（已选定）
- **不新建 JIT module**，复用现有 `module_deepgemm_opus`。
- 新增 `kernel_tag = a8w8_mxfp8`，新 kid = **3**（1=a8w8_scale，2=a8w8 已占用）。
- codegen emit **复用 `gen_scale_instance`**（launcher 签名一致；`num_groups_k = K/GROUP_K` 由 traits 自动流入，无需改 launcher 代码）。
- **新增独立 pybind 入口 `opus_gemm_a8w8_mxfp8` + Python `gemm_a8w8_mxfp8_opus`**，不复用 block-scale 入口，避免 GROUP_K=32/128 在同一入口做歧义分派。

### 为什么不新建 module
- `module_deepgemm_opus` 已含所需的一切：`opus.hpp` include 路径、`*.device.cu → -D__HIPCC_RTC__` per-source flag、`-mllvm --amdgpu-mfma-vgpr-form` HIP flag。
- 新建 module 需要重写整套 `optCompilerConfig.json` 条目 + 独立 `gen_instances.py`，纯属浪费。

---

## 2. 现有 Opus GEMM 集成机制（参考地图）

**单一 module：`module_deepgemm_opus`**

### JIT 配置
`aiter/jit/optCompilerConfig.json:298-319`
- `srcs`：`pybind/opus_gemm_pybind.cu` + `opus_gemm/opus_gemm.cu`
- `blob_gen_cmd`（318 行）：运行 `opus_gemm/gen_instances.py --working_path {0} --tune_files .../bf16_tuned_gemm.csv:... --compiled_kids_sidecar={0}/../../compiled_kids_opus.json`
- per-source flag：`*.device.cu → -D__HIPCC_RTC__`（308 行）
- `extra_include`：`opus_gemm/include`（313 行）

### Pybind
`csrc/pybind/opus_gemm_pybind.cu`（host-only，`#ifndef __HIP_DEVICE_COMPILE__` 包住整个 TU）：
```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    AITER_SET_STREAM_PYBIND
    OPUS_GEMM_PYBIND;
    OPUS_GEMM_A16W16_TUNE_PYBIND;
    OPUS_GEMM_WORKSPACE_INIT_PYBIND;
}
```
三个宏定义在 `csrc/include/rocm_ops.hpp:273-302`。

### Python wrapper
`aiter/ops/opus/`
- `__init__.py` — arch gate（gfx950/gfx942），导出公开 API，不支持的 arch 装 raising stub。
- `gemm_op_a16w16.py` — `@compile_ops("module_deepgemm_opus", ...)` 装饰的 stub。
- `common.py` — tuned-CSV lookup。
- `_arch.py` — `_detect_arch`。

### Codegen 流程（blob_gen_cmd 触发）
`csrc/opus_gemm/gen_instances.py`：
1. 导入 `codegen.gen_instances_gfx950` / `_gfx942`（side-effect：各 arch 模块在 import 时 `register_emit(arch, tag, fn)` + `register_arch_map(...)`，registry 在 `codegen/common.py:41-83`）。
2. `__main__`（881 行起）：从 CSV `libtype=='opus'` 行 + sidecar JSON + `HEURISTIC_DEFAULT_KIDS` 计算**子集编译集 S**，arch-filter，写出 `opus_build_archs.h`（`OPUS_BUILD_HAS_GFX942/GFX950` 宏）。a8w8/a8w8_scale kid 被**无条件加入 S**（1073-1074 行）。
3. `gen_instances(kdict)`：对每个 kid `dispatch_emit((kid_arch, kernel_tag))` → 注册的 emit fn → 写 `impl/<kidname>.cuh`；再产出 per-arch host TU、per-(kid,dtype) `.device.cu`、splitk reduce TU、以及 lookup/manifest headers。

**关键**：编译是**按 kid 子集**进行的。新 kid 3 必须像 a8w8 一样被无条件加入 S，否则 dispatcher 引用其 launcher 符号会链接失败。

### C++ dispatcher
`csrc/opus_gemm/opus_gemm.cu`
- `opus_dispatch_scale<T>`（40 行）返回 `opus_gemm_512x256x256x128_4x2_16x16x128_1x128x128<T>`。
- `opus_gemm()`（102 行）：`XQ.dtype()==fp8` 且 gfx950，`has_scale` 时调 `opus_dispatch_scale<fp32_t>(...)(XQ,WQ,Y,x_scale,w_scale)`（140 行）。

### kargs 对齐确认
本 kernel 的 `opus_gemm_kargs`（`gemm_a8w8_mxfp8_scaled_common.h`）字段与 `gen_scale_instance`（`codegen/gen_instances_gfx950.py:688-786`）生成的 kargs 填充**完全一致**：
```
ptr_a/b/c, m/n/k/batch, stride_a/b/c(=K/K/N), stride_*_batch(=M*K/N*K/M*N),
ptr_sfa/sfb, stride_sfa/sfb(=num_groups_k), stride_sfa/sfb_batch(=num_groups_m*ngk / num_groups_n*ngk)
```
`num_groups_k = K / GROUP_K`，GROUP_K=32 时自动得出正确 microscale stride，**launcher 无需改代码**。

---

## 3. 行动步骤（按依赖顺序）

### Step 1 — device 侧头文件
目录：`csrc/opus_gemm/include/gfx950/`
- 新建 `opus_gemm_pipeline_a8w8_mxfp8_gfx950.cuh`
  - 源：`gemm_a8w8_mxfp8_scaled_kernel_template.hpp`（device kernel body + 8 个 layout helper）。
  - kernel 函数名保持/重命名为 `gemm_a8w8_mxfp8_kernel`。
- 新建 `opus_gemm_traits_a8w8_mxfp8_gfx950.cuh`
  - 源：`gemm_a8w8_mxfp8_traits`，改写成 codegen 期望的模板形式（对照 `opus_gemm_traits_a8w8_scale_gfx950.cuh`）：
    ```
    template<int BLOCK_SIZE_, typename BLOCK_, typename DTYPE_, typename VEC_, typename GROUP_>
    ```
    其中 DTYPE 为 5-tuple `<D_A, D_B, D_C, D_ACC, D_SF>`，`D_SF = unsigned char`。
  - kargs 结构体放这里（布局同 a8w8_scale 的 `opus_gemm_scale_kargs_gfx950`，可直接复用同名类型或新建同布局类型）。

### Step 2 — 注册 kernel 实例
文件：`csrc/opus_gemm/opus_gemm_common.py`
- 新增（对照 203-205 行的 `a8w8_scale_kernels_list`）：
  ```python
  a8w8_mxfp8_kernels_list = {
      3: OpusGemmInstance(512, 256, 256, 128, 4, 2, 16, 16, 128, 16, 16, 4,
                          1, 1, 32, "a8w8_mxfp8", ["fp32_t"]),
  }
  ```
  **注意 GROUP_K=32**（第 15 个位置参数；确认 `OpusGemmInstance` 字段顺序：`opus_gemm_common.py:27`）。
- 合并进 `kernels_list`（~695 行，加 `**a8w8_mxfp8_kernels_list`）。

### Step 3 — codegen 接线
文件：`csrc/opus_gemm/codegen/gen_instances_gfx950.py`
- 在各 map dict 加 `"a8w8_mxfp8"` 条目：
  - pipeline_header → `gfx950/opus_gemm_pipeline_a8w8_mxfp8_gfx950.cuh`（23 行附近）
  - traits_header → `gfx950/opus_gemm_traits_a8w8_mxfp8_gfx950.cuh`（42 行附近）
  - kernel_func → `gemm_a8w8_mxfp8_kernel`（52 行附近）
  - traits_name → `opus_gemm_a8w8_mxfp8_traits_gfx950`（68 行附近）
  - kargs_name → 复用 `opus_gemm_scale_kargs_gfx950`（78 行附近）
- 注册 emit：`register_emit("gfx950", "a8w8_mxfp8", gen_scale_instance)`（复用现有函数，1418 行附近）。

文件：`csrc/opus_gemm/gen_instances.py`
- import `a8w8_mxfp8_kernels_list`（29 行附近）。
- `SCALE_TAGS` 加 `"a8w8_mxfp8"`（122-123 行附近，5-arg launcher 签名）。
- `TAG_TO_LIST` 加 `"a8w8_mxfp8": a8w8_mxfp8_kernels_list`（957 行附近）。
- 无条件加入编译集 S：`S |= set(a8w8_mxfp8_kernels_list.keys())`（仿 1073 行）。

### Step 4 — C++ dispatcher
文件：`csrc/opus_gemm/opus_gemm.cu`
- 加：
  ```cpp
  template <typename CDataType>
  OpusScaleKernel opus_dispatch_mxfp8(int M, int N, int K) {
    return opus_gemm_512x256x256x128_4x2_16x16x128_1x1x32_mxfp8<CDataType>;
    // 实际符号名以 gen_scale_instance 产出的 k.name 为准
  }
  ```
- 新增独立入口 `opus_gemm_a8w8_mxfp8(XQ, WQ, Y, x_scale, w_scale)`：内部 `AITER_CHECK` gfx950 + fp32 输出 + fp8 输入，再调 `opus_dispatch_mxfp8<fp32_t>(M,N,K)(...)`。

文件：`csrc/opus_gemm/include/opus_gemm.h`
- 声明 `void opus_gemm_a8w8_mxfp8(aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&, std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);`

> 备注：新 kid 的 launcher 符号名由 codegen 依据 `OpusGemmInstance` 参数拼出。build 一次后到 `aiter/jit/build/module_deepgemm_opus/blob/` 里确认真实 `k.name`，再回填 dispatcher。

### Step 5 — Pybind
文件：`csrc/include/rocm_ops.hpp`
- 加宏 `OPUS_GEMM_A8W8_MXFP8_PYBIND`（对照 273 行 `OPUS_GEMM_PYBIND`）：
  ```cpp
  #define OPUS_GEMM_A8W8_MXFP8_PYBIND        \
      m.def("opus_gemm_a8w8_mxfp8",          \
            &opus_gemm_a8w8_mxfp8,           \
            "opus_gemm_a8w8_mxfp8",          \
            py::arg("XQ"), py::arg("WQ"), py::arg("Y"), \
            py::arg("x_scale") = std::nullopt, \
            py::arg("w_scale") = std::nullopt);
  ```
文件：`csrc/pybind/opus_gemm_pybind.cu`
- 在 `PYBIND11_MODULE` 内加 `OPUS_GEMM_A8W8_MXFP8_PYBIND;`

### Step 6 — Python wrapper
文件：`aiter/ops/opus/gemm_op_a8w8_mxfp8.py`（新建）
- `@compile_ops("module_deepgemm_opus", fc_name="opus_gemm_a8w8_mxfp8", develop=True)` 的 stub。
- 公开 `gemm_a8w8_mxfp8_opus(A, B, x_scale, w_scale, out=None)`：
  - 校验 A/B 为 fp8、3D、contiguous strides（(N*K,K,1) 等，参照 `gemm_op_a16w16.py` 的 `_check_a16w16_tune_layout`）。
  - 校验 scale shape：SFA `[M, K/32]`、SFB `[N, K/32]`（E8M0 / uint8）。
  - 分配/复用 fp32 输出 Y `[batch, M, N]`。
文件：`aiter/ops/opus/__init__.py`
- 导出 `gemm_a8w8_mxfp8_opus`（arch_ok 分支里）+ unsupported-arch stub。

### Step 7 — 测试
文件：`op_tests/test_opus_a8w8_mxfp8_gemm.py`（新建）
- 复用 host driver（`gemm_a8w8_mxfp8_scaled_host.cc`）里的 CPU 参考逻辑：
  - E8M0 解码 `value = 2^(byte-127)`。
  - condition-aware 容差：`tol = abs_floor(1e-4) + rel_mag(5e-5) * Σ|term|`，double 累加做 ground truth。
- arch-gate 到 gfx950（非 gfx950 skip）。
- argparse 驱动 config sweep，遵循 `aiter/test_common.py` 的 `checkAllclose` / `@perftest()` 模式。

### Step 8 — 构建验证
```bash
AITER_REBUILD=2 GPU_ARCHS=gfx950 python3 op_tests/test_opus_a8w8_mxfp8_gemm.py
```
- 首次会触发 codegen + JIT build。
- build 后到 `aiter/jit/build/module_deepgemm_opus/blob/impl/` 确认 kid 3 的 `.cuh` 与真实符号名，回填 Step 4 的 dispatcher。

---

## 4. 需要改动的文件清单

| 步骤 | 文件 | 动作 |
|---|---|---|
| 1 | `csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxfp8_gfx950.cuh` | 新建（device kernel） |
| 1 | `csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxfp8_gfx950.cuh` | 新建（traits + kargs） |
| 2 | `csrc/opus_gemm/opus_gemm_common.py` | 加 `a8w8_mxfp8_kernels_list`（kid 3, GROUP_K=32），并入 `kernels_list` |
| 3 | `csrc/opus_gemm/codegen/gen_instances_gfx950.py` | 加 tag map 条目 + `register_emit` |
| 3 | `csrc/opus_gemm/gen_instances.py` | import list、`SCALE_TAGS`、`TAG_TO_LIST`、无条件加入 S |
| 4 | `csrc/opus_gemm/opus_gemm.cu` | 加 `opus_dispatch_mxfp8` + `opus_gemm_a8w8_mxfp8` 入口 |
| 4 | `csrc/opus_gemm/include/opus_gemm.h` | 声明新入口 |
| 5 | `csrc/include/rocm_ops.hpp` | 加 `OPUS_GEMM_A8W8_MXFP8_PYBIND` 宏 |
| 5 | `csrc/pybind/opus_gemm_pybind.cu` | 调用新宏 |
| 6 | `aiter/ops/opus/gemm_op_a8w8_mxfp8.py` | 新建 Python wrapper |
| 6 | `aiter/ops/opus/__init__.py` | 导出 `gemm_a8w8_mxfp8_opus` |
| 7 | `op_tests/test_opus_a8w8_mxfp8_gemm.py` | 新建测试 |

---

## 5. 关键坑位提醒

1. **kid 3 必须无条件加入编译集 S**（Step 3），否则 dispatcher 引用 launcher 符号链接失败——这是 a8w8/a8w8_scale 特殊对待的原因（`gen_instances.py:1073-1074`）。
2. **GROUP_K=32 vs 128**：这是与 a8w8_scale 的唯一实质区别，务必在 `OpusGemmInstance` 与 traits 里正确设置；`num_groups_k = K/32` 自动流入 launcher，无需改 `gen_scale_instance`。
3. **launcher 符号名**由 codegen 依据实例参数拼出，Step 4 的 dispatcher 里的符号名需 build 一次后回填确认。
4. **fp32-only 输出**：与 a8w8_scale 一致，入口处 `AITER_CHECK Y.dtype()==fp32`。
5. **gfx950-only**：a8w8 系全部 gfx950-only，入口处 arch check。
6. HIP flag `-mllvm --amdgpu-mfma-vgpr-form` 与 `__launch_bounds__(512, 2)` 已在 module 配置中，兼容。
7. **stride 校验**：launcher 硬编码 `stride_b_batch=N*K` 等，Python 侧必须拒绝 broadcast/transpose/slice 视图（参照 `gemm_op_a16w16.py` 的注释与守卫）。

---

## 6. 建议实施顺序

先做 **Step 1 → 2 → 3**（headers + 注册 + codegen），这三步决定能否 build 出 kid 实例；然后 **Step 4 → 5 → 6**（C++/pybind/Python 打通调用链）；最后 **Step 7 → 8**（测试 + 验证）。
