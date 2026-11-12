# candidate_m9 —— 在 candidate_m8 基础上把 scale A/B 改为 aiter 的 e8m0_shuffle 布局

生成：2026-11-12，GPU2（另一台 MI355X）。改了 kernel 模板、host（生成 shuffle 后的 scale）、traits 的 kargs 注释。
kernel md5：b3f29b0a…

## 接口变化（重要）

`ptr_sfa / ptr_sfb` 不再是行主序 `[rows, K/32]`，而是 aiter gfx950 a4w4 的标准 scale 布局，即
`aiter.utility.fp4_utils.e8m0_shuffle` / `aiter.ops.shuffle.shuffle_scale`（`quant(..., shuffle=True)` 产出的就是它）：
行补到 32 的倍数、K/32 补到 8 的倍数（填 0x7F），排成 `[rows/32][K/256][k_lane 4][row16 16][kk 2][row-pair 2]`，每块 256 B 连续。
`stride_sfa / stride_sfb` = 一个 32 行块的字节数（`32 * K/32 补到 8`）。host 里的 `shuffle_sf` 是逐字节等价的 CPU 实现；参考计算仍用原始 scale。
A、B 数据本身仍是原始行主序（B 的 preshuffle 是下一步）。

## 动机（W1 诊断，m7/m8）

原布局下，每条 scale 全局读的 64 个 lane 各读不同行的 16 B（行距 256 B），访问 64 个 64 B sector；scale 只占数据量约 1/16，却占 L1→L2 请求约 35%，并反压 A/B 的 LDS-DMA 预取（`TA_ADDR_STALLED_BY_TC`）。

## 改动

| 环节 | m8 | m9 |
|---|---|---|
| 全局读 | 每 lane 读一行 16 B，64 sector/条 | 一个 panel（2 K tile × 256 行）= 8 段连续 512 B，wave w 读第 2w、2w+1 段，每 lane 连续 16 B，16 sector/条 |
| 写 LDS（put_sf） | 4 次 DPP 4×4 字节 transpose + 两条 8 B 写（W2 swizzle） | 不做 transpose，原样一条 16 B 写（连续，无冲突） |
| 读 LDS（load_sfa/sfb） | 每半 tile 一条 8 B 读 | 每个重复 e 一个 dword（连续 256 B，无冲突），只发读、不等数据 |
| 拼 `v_sfa[h][kk]` | 读出即用 | 在 `read_sf`（C11/C00，数据早已到达）里用 4 条 `v_perm` 按 wave_m/wave_n 取字节；`v_sfa` 格式和 MFMA 调用完全不变 |
| 下一 tile 的 scale 预取偏移 | — | 在 Last tile 里用不透明 lane id 现算，避免逐 lane 偏移整 tile 存活导致的 VGPR spill |

## 过程中的变体（GPU2，与 m8 交替）

| 变体 | 结果 | 说明 |
|---|---|---|
| sc1：v_perm 紧跟在 LDS 读后 | +0.6%（变慢） | 机制生效（`TA_ADDR_STALLED_BY_TC` 5.47e6→1.31e6，−76%；buffer_load stall/tile 10866→2916），但 main loop 里 s_waitcnt 252→458 条/tile，等待吃掉收益；满载时钟 1475 MHz（MFMA 空闲多、功耗低）但 cycle 多约 7.5% |
| sc2：v_perm 推迟到 read_sf | −1.35% | 但有 2 个 VGPR spill，每 tile 一次 scratch 读后 `vmcnt(0)` 等 721 cycle |
| sc3：第 1 半的读推迟到 read_sf(0) | 与 sc2 持平 | spill 不来自这里 |
| **sc4（= m9）：下一 tile 的 scale 偏移就地重算** | **−1.6%** | spill 消失 |

## 结果（GPU2，8192³，`-w 1000 -i 200`，交替 7 轮中位数）

| | m8 | m9 |
|---|---|---|
| 时间 | 273.9 µs | **269.6 µs（−1.6%，7/7 轮）** |
| 满载时钟 | 1380 MHz | 1414 MHz |

相对主文件（287.1 µs）：**−6.1%**。
正确性：8192³ 随机 scale；1024×1024×512、1024×2048×1024×b2、2048×1024×7168×b3、768×1280×1536、256×256×512，`MXFP4_SFA_K_PATTERN=1`，`PERSIST_GRID` 默认 / 37 / 1，全部 VALID。VGPR 512 无 spill，SGPR spill 8。

## 下一步

B 改为 aiter `shuffle_weight(w, layout=(16,16))` 的 preshuffle 布局（对齐 `gemm_a4w4(bpreshuffle=True)`）。
研究结论（PR6696 路径）：照搬 PR6696 的 chunk 定义——每条 DMA 搬 shuffled B 中连续 1 KB 原样落入一个 LDS chunk，读取按 `lane*16` 连续读；每条仍 16 sector，B 侧天然无 bank 冲突、无需 swizzle。只改 `make_layout_gb`、`gb_offset`、`make_layout_rb` 与 `load_b` 的 kk 偏移。

## 测试

```bash
cd candidate_m9
OPUS_INCLUDE_DIR=<opus include> ./compare.sh 2 7
```
