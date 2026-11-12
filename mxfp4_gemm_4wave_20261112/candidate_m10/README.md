# candidate_m10 —— 在 candidate_m9 基础上 B 改为 aiter 的 preshuffle 布局（对齐 gemm_a4w4 接口）

生成：2026-11-12，GPU2（另一台 MI355X）。改了 kernel 模板的 B 侧 layout、host（生成 shuffle 后的 B）、traits 注释。
kernel md5：aa3f2fd6…

## 接口：现在与 aiter gfx950 `gemm_a4w4(..., bpreshuffle=True)` 的输入约定一致

| 输入 | 布局 |
|---|---|
| A | 行主序 `[M, K/2]`（不变） |
| B | `aiter.ops.shuffle.shuffle_weight(w, layout=(16, 16))`：字节 (n, kb) → `(n/16)*16*Kb + (kb/32)*512 + ((kb/16)%2)*256 + (n%16)*16 + kb%16`，Kb = K/2；`stride_b` 仍填 K/2 |
| scale A / B | `e8m0_shuffle` / `shuffle_scale`（m9 起） |

要求 N 是 16 的倍数、K/2 是 32 的倍数（kernel 本来就要求 K 是 512 的倍数）。host 里的重排是逐字节等价的 CPU 实现；参考计算仍用原始 B。

## 改动（参考 PR6696 的 bpreshuffle 路径）

shuffled B 中，一个 16 行块 × 一个 128 B 的 K tile 是连续 2 KB，其中每个 K 半段（kk = 0/1）连续 1 KB，内部排成 `[g 4][n 16][16 B]`——正好是一条 `ds_read_b128` 按 `lane*16` 读出的 MFMA 操作数（行 lane%16，K 字节 64kk + 16g）。

| 位置 | 改动 |
|---|---|
| `make_layout_gb` | wave (wm, wn) 的第 i = ih·2+il 个 issue 搬 16 行块 `(wm·2+wn)·2+ih` 的 K 半段 il：连续 1 KB，每 lane 偏移 `nblk·16·Kb + il·1024 + lane·16`；去掉 m7 的 XOR |
| `gb_offset` | K tile 步长 128 B → 2048 B |
| `make_layout_rb` | 第 e 个重复读 chunk `(2e+wn)·2 + kk`，lane 读 `chunk·1088 + lane·16`（连续 1 KB，无冲突）；去掉 r%2 / r/2 拆分和 XOR |
| `load_b` | kk 偏移 64 B → 一个 chunk（1088 B） |
| 不变 | LDS 写入（DMA）、`make_layout_sb`、A 侧、scale、C 暂存、LDS 总量 |

## 结果（GPU2，8192³，交替 7 轮中位数）

| | m9 | m10 |
|---|---|---|
| 时间 | 269.6 µs | **269.3 µs**（−0.1%，噪声内；7 轮中 6 轮更快、1 轮持平） |
| `SQ_LDS_BANK_CONFLICT` | 0 | 0 |
| `TCP_TCC_READ_REQ` / `TCP_TOTAL_CACHE_ACCESSES` | 1.736e7 / 3.774e7 | 完全相同 |
| `TA_ADDR_STALLED_BY_TC` | 1.70e6 | 1.12e6 |

结论：B preshuffle 不损失性能（每条 DMA 仍 16 个 64 B sector，读取天然无冲突），B 侧代码更简单，换来与 aiter 接口的对齐。
相对主文件（287.1 µs）：**−6.2%**。

正确性：8192³ 随机 scale；1024×1024×512、1024×2048×1024×b2、2048×1024×7168×b3、768×1280×1536、256×256×512，`MXFP4_SFA_K_PATTERN=1`，`PERSIST_GRID` 默认 / 37 / 1，全部 VALID。VGPR 512 无 spill，SGPR spill 8。

## 测试

```bash
cd candidate_m10
OPUS_INCLUDE_DIR=<opus include> ./compare.sh 2 7
```
