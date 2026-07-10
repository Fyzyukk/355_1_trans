# TODO: 用 opus scaled-MFMA 内建替换手写缩放路径

## 背景

当前 MXFP8 kernel 的 per-group E8M0 缩放是**软件手写**的：

- `D_SF = unsigned char`，scale byte 以裸字节载入（`gemm_a8w8_mxfp8_kernel_template.hpp` 顶部）。
- `e8m0_to_f32(b)`：`bit_cast(float, b << 23)` 手动把 E8M0 解码成 fp32。
- MMA 用**普通** `mfma_f32_16x16x32_fp8_fp8`（未缩放），出 partial sum 后由
  `scale_c_group(...)` 逐元素乘 `scale_a[m,g] * scale_b[n,g]` 再累加。
- 循环里 `E_K`(=4) 次：每个 32-K group 一条普通 MMA + 一次软件缩放。

> 注：`D_SF` 不能直接换成 `opus::e8m0_t`——它是 `OPUS_DEFINE_FPACKS` 定义的
> 打包子字节 fp 类型（opus.hpp:1212），不能作为 `vector_t` 元素，也不能直接 bit 操作。
> 所以"更 native"的方向不是改 typedef，而是换整条缩放路径（见下）。

## 目标：改用硬件 scaled-MFMA

gfx950 有硬件缩放 MMA 内建，opus 已封装：

- 内建：`__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`（opus.hpp:2293）
- opus 封装：`mfma_scale_f32_16x16x128_fp8_fp8`（opus.hpp:2334）
- 调用重载：`mfma::operator()(a, b, c, int scale_a, int scale_b, scale_op_sel_a, scale_op_sel_b)`
  （opus.hpp:2287-2295）
  - `scale_a/scale_b`：运行时 E8M0 指数值（int）；`127` = 不缩放（2^0）。
  - `scale_op_sel_a/b`：编译期 2-bit 字节选择器，从打包的 int32 scale word 里选哪一字节喂本次 MMA；`0` = 低字节 = 传统标量行为。
  - 输入寄存器恒为 256bit（i32x8），内部 bitcast；格式码 fp8=0。

这样硬件在 MMA 内部直接吃 E8M0 scale byte，**省掉** `e8m0_to_f32` 和 `scale_c_group` 两段软件逻辑。

## 已确认的硬件行为（2026-07-09 在 gfx950 实测，权威）

> 实测 probe 与结论见 `../mxfp8_wmma_scale_probe/`（`SCALED_MFMA_16x16x128_LAYOUT.md`
> 及 `probe_count.cu` / `probe_scale_map.cu` / `probe_routing.cu`）。

**结论：一条 `16x16x128` scaled MFMA 恰好能表达 GROUP_K=32，一条指令按 kg 分别缩放 4 个 block。**

1. **scale 是 per-lane VGPR，不是单标量广播**。lane 映射：
   `lane L → (row/col = L%16, K-block = L/16)`，每 block = 连续 32 个 K。
   - 每 lane 传自己那个 block 的 1 个 E8M0（放 int 的 byte0，`scale_op_sel=0` 选它）。
   - 一条指令共吃 **A_scale[16 行][4 block] = 64 个** + **B_scale[16 列][4 block] = 64 个**。
   - 因此 `16x16x128` scaled MFMA **原生支持每 32-K 一个独立 scale**（4 个 block 各自缩放）。

2. **数据布局（BLOCK 模型）**：一个 lane 的 32 个 fp8 全在同一连续 32-K block（实测 delta=32），
   即 `A[row][blk*32 : blk*32+32]`。这 32 个数据配的 scale 正好是同 lane 的那 1 个。

3. **E8M0 编码 `2^(byte-127)`，unit=127**（不是 0！）。`scale_op_sel` 只从 packed-int32 选字节，
   与 block 路由正交；本用法每 lane 只需 1 个 scale，byte0 + op_sel=0 是标准。

## 换硬件 scaled-MFMA 的改动要点

1. **W_K/GROUP_K 断言重写**：从 `W_K == GROUP_K == 32` 改为 `W_K == 128`，一条 128-K scaled MMA
   内部按 lane/16 自动对应 4 个 32-K block；GROUP_K=32 由 scale 的 per-lane 供给表达，不再靠多条 MMA。

2. **sfa/sfb layout**：需按 `lane → (row/col=L%16, K-block=L/16)` 排布，让每 lane 拿到它那个
   (row, block) 的 E8M0，放进 int 的 byte0。删掉 `scale_c_group` / `e8m0_to_f32`（硬件内部吃 E8M0）。

3. **调用**：`mma(v_a, v_b, v_c, scale_a, scale_b)`（scale_a/b 为 int，byte0=E8M0，op_sel 默认 0）。
   注意 opus 只有 scaled `16x16x128` / `32x32x64`，**没有 scaled 16x16x32**——所以硬件路径必须用 W_K=128。

## 影响文件

- `gemm_a8w8_mxfp8_kernel_template.hpp`：`scale_c_group`、`e8m0_to_f32`、MMA 循环（step_k 段）、
  `vtype_sfa/vtype_sfb`、sfa/sfb layout。
- `gemm_a8w8_mxfp8_common.h`：`W_K`/`GROUP_K` 相关断言与 `NUM_KGROUPS`。

## 状态

未开始。当前手写路径功能正确，先保留；此为后续性能/简洁性优化项。
