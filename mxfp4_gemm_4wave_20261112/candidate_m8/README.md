# candidate_m8 —— 在 candidate_m7 基础上消除剩余的全部 LDS bank 冲突（scale panel + C 暂存）

生成：2026-11-12，GPU2（另一台 MI355X）。只改了 kernel 模板；traits/host/Makefile 与 m7 相同。
kernel md5：7f9a3aa3…

## 残余冲突的来源（m7，每 wave 每 tile，微基准实测，见 `ldscal/`）

| 指令 | 冲突周期/条 | 条数 | 合计 |
|---|---|---|---|
| put_sf `ds_write_b128`（写 scale panel） | 24 | 32 | 768 |
| load_sfa/sfb `ds_read2st64_b64`（读 scale） | 8 | 64 | 512 |
| C 暂存 `ds_write2_b64` | 8 | 32 | 256 |
| C 读回 `ds_read_b128` | 0 | 32 | 0 |

合计 1536 × 4 wave × 1024 tile ≈ 6.29e6，与全 kernel 实测 6.39e6 一致。
成因：写入端按连续 lane 组每拍 128 B 处理（b128 每组 8 lane，b64 每组 16 lane）；put_sf 的槽位 (lane%4)·16+lane/4 在每 8 lane 内 4 路冲突；sf 读每个 lane 只用 16 B 槽里的 8 B，bank 只用到一半；C 行距 528 B 使第 r 行和 r+8 行同 bank。

## 改动

| 部分 | 改动 |
|---|---|
| scale panel（主要收益） | LDS 布局从 [行][16 B] 改为 [tile_in_panel][行][8 B]（每 buffer 仍 4096 B），wave 内 slot XOR：s → s ^ (((s>>4)&3)<<2)；put_sf 改为两条 8 B 写，读端基址按同一 swizzle 计算 |
| C 暂存 | 第 3 位为 1 的行在每个 16 B 块内交换两半 8 B；读回时第 i 次读的行号第 3 位 = i&1（编译期确定），奇数次在寄存器里换回（多 33 条 v_mov） |

## 结果（GPU2，8192³，交替 7 轮中位数）

| | m7 | m8 |
|---|---|---|
| `SQ_LDS_BANK_CONFLICT` | 6.39e6 | **0** |
| `SQ_LDS_IDX_ACTIVE` | 2.79e7 | 2.15e7（−23%） |
| A/B `ds_read_b128` cycle/条（ATT） | 8.0（GPU4 测） | 6.1 |
| **时间** | 276.1 µs | **273.9 µs（−0.80%，7/7 轮）** |

单项：只改 scale panel（v_sf）274.1 µs（−0.72%）；只改 C 暂存（v_c）在噪声内（写回段受全局写带宽限制），并入后不变差，保留它让 LDS 冲突为 0。
相对主文件（287.1 µs）：**−4.6%**。

正确性（GPU2 独立重跑）：8192³ 随机 scale；1024×1024×512、1024×2048×1024×b2、2048×1024×7168×b3、768×1280×1536、256×256×512，`MXFP4_SFA_K_PATTERN=1`，`PERSIST_GRID` 默认 / 37 / 1，全部 VALID。VGPR 512 无 spill，SGPR spill 7。

## 同轮并行探索中没有采用的（详见各自目录，均以 m7 为基线、各自 GPU 上同卡 A/B）

| 方向 | 结果 |
|---|---|
| main loop 第一趟剥出、去掉 C 清零 | −0.2%（低于 0.5% 噪声带），代码 +3 KB、SGPR spill 6→12 |
| store 地址计算移到 SALU | 0：省下的 VALU 等量变成 store stall（写回段受带宽限制） |
| A/B LDS-DMA 加 `nt` | +6%：破坏 L2 跨 CU 复用 |
| C01 的 buffer_load 挪到 C11 | +0.7%：stall 跟着走，TC 持续吞吐受限 |
| 每两趟成对取 scale panel（v_pair） | 单独 −0.57%；叠加到 m8 上 +0.4%（变慢），不采用 |

## 下一个方向（诊断已完成，需改结构）

main loop 里 buffer_load 卡顿的主因是 **scale panel 的全局读**：每条 64 lane 各读不同行的 16 B（行距 256 B），访问 64 个 64 B sector（A/B 拷贝只有 16 个）；scale 只占数据量约 1/16，却占 L1→L2 请求的约 35%，L2 命中率约 97%（开销在 L1/TA 访问次数，不在 HBM）。
上界（结果错误，仅诊断）：去掉 scale 访问 −4.7%，减半 −2.7%。要拿到它，需要让一条指令的多个 lane 合读同一 sector 并直接按 MFMA 布局落地（例如 scale 也走 LDS-DMA、transpose 放到读 LDS 一侧），而 LDS 已在 160 KB 上限，需要重新规划 LDS 预算。

## 测试

```bash
cd candidate_m8
OPUS_INCLUDE_DIR=<opus include> ./compare.sh 2 7
```
