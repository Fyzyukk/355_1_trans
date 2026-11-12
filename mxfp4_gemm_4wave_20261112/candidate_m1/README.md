# candidate_m1 —— 任务 3 + 任务 4/5 合并版（待 GPU2 验证）

生成：2026-11-11，GPU6 上开发。主文件 `../gemm_a4w4_mxfp4_4wave_kernel_template.hpp` **没有改动**（md5 3565421…）。

## 包含的改动（相对主文件）

| 任务 | 改动 |
|---|---|
| 3 | scale panel 改为 global → VGPR → 4x4 字节 transpose → 写 LDS；LDS 里存的就是 MFMA 的 scale 布局，main loop 里不再做 transpose（`permlane` 和 `s_nop` 减半，VALU 少约 40%）。panel 的 transpose 穿插在 C11 的 MFMA 之间。 |
| 4 | LDS 按 stage 排布 `[A s0][B s0][A s1][B s1]`；C 用原来的 528 B 行距，只用 stage 1 分两批（上半、下半 128 行）暂存。 |
| 5 | persistent kernel：grid = CU 数，每个 block 处理多个 tile；在上一个 tile 的最后一个 k-tile 计算时，提前发出下一个 tile 的 k-tile 0（DMA）和 scale panel 0（读进 VGPR）；写回不再按 quadrant 等 `vmcnt(0)`。 |

任务 1（prologue 读顺序）在这个方案里不适用（prologue 不再做 transpose）；任务 2（barrier 挪位置）没有效果，未合入。各任务的单独版本在 `variants/`（任务 4/5 的单独版本需要本目录的 persistent host）。

**host 也改了**：`gemm_a4w4_mxfp4_4wave_host.cc` 按 CU 数 launch（可用环境变量 `PERSIST_GRID` 覆盖），batch 展开到 tile 编号里。

## 正确性（GPU6）

8192×8192×8192：普通随机 scale 和 `MXFP4_SFA_K_PATTERN=1` 都是 errors=0；另外验证过 persistent 下每个 block 处理多个 tile、多 batch、K=512/1024/2560/7168。无寄存器溢出（ScratchSize 0）。

## GPU6 上的结果（8192³，`-w 1000 -i 200`，与主文件交替 7 轮）

| 版本 | 7 轮（µs） | 中位数 |
|---|---|---|
| 主文件 | 243.2 243.2 243.3 243.0 243.0 243.0 243.3 | 243.2 |
| 本版本 | 245.6 252.9 242.6 253.6 246.2 246.0 245.9 | 245.9（波动大） |

## trace（GPU6，8192³）

`trace_8192_m1/att/ui_output_agent_37211_dispatch_1/` 用 ROCprof Compute Viewer 打开。注意：persistent 版本每个 SIMD 只有 1 个 wave（处理 4 个 tile）。

逐项对照（`trace_8192_m1/items_vs_main.md`，每个 tile 每个 wave 的 cycle）：

| 项目 | 主文件 | 本版本 |
|---|---|---|
| 合计 | 100893 | 103515 |
| main loop VALU | 9772 | 5646 |
| main loop `s_nop` | 1440 | 600 |
| 写回 `vmcnt(0)` 等待 | 1496 | 8 |
| `s_endpgm` 等 store | 1398 | 248 |
| block 之间空档 | 472 | 0 |
| prologue `buffer_load` | 6004 | 4226 |
| prologue `lgkmcnt` 等待 | 465 | 2380 |

目标指令/等待消掉了，但时间转移到了 ds_read、store 和 prologue 的等待上，总数没有下降。

## 在 GPU2 上测试

```bash
cd /root/workspace/mxfp4_gemm_4wave/candidate_m1
./compare.sh 2 7      # GPU 编号、轮数：先各自 8192^3 校验，再与主文件交替计时
```

先确认 GPU2 空闲（脚本会打印 GPU2 的占用和功耗）。
