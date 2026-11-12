# candidate_m7 —— 在 candidate_m6 基础上消除 A/B 操作数读取的 LDS bank 冲突

生成：2026-11-12，GPU2（另一台 MI355X）。相对 m6 只改了 traits 的 `smem_padding` / LDS 断言，以及模板里 4 处 layout 坐标（共 7 行）。
kernel md5：5df8a63a…

## 发现过程（PMC，ATT 看不出来）

`SQ_LDS_BANK_CONFLICT` 约占 `SQ_LDS_IDX_ACTIVE`（LDS 忙碌周期）的 50%。先用微基准标定计数器语义（`ldscal/`）：
连续读 1 KB 的 `ds_read_b128` 计 0、全同 bank 计 60/条，说明计数器测的是真实冲突，`ds_read_b128` 天然的 4 拍不计入。

A/B 读模式：lane (r = lane%16, g = lane/16) 读 `(r%2)·chunk_stride + (r/2)·128 + g·16`（+kk·64）。
chunk_stride = 1024 + padding = 1056 时，一条指令所有地址 mod 256 只覆盖 192 B，**每条 ds_read_b128 多 4 拍（8 拍而不是 4 拍）**；
只在 64 B 半行内重排 16 B 块的 256 种线性 XOR 全部实测过，最好仍是 4。
chunk_stride = 1088（padding 64）时，256 种里有 96 种实测为 0 冲突。

## 改动

| 位置 | 改动 |
|---|---|
| traits | `smem_padding` 32 → 64，LDS 总量 159744 → 163840（正好 160 KB） |
| `make_layout_ga/gb`（LDS-DMA 写入端） | lane 的全局 K 块号 `lane%8` → `(lane%8) ^ (((lane/8 >> 1) & 1) << 1)`：只在一行 128 B 内调换 16 B 块，全局访存仍合并 |
| `make_layout_ra/rb`（读取端） | `g` → `g ^ (((lane_m >> 2) & 1) << 1)`；只翻 bit1，kk 的 +64 与 E_M 的立即数偏移不变 |
| C 暂存的 static_assert | `==` → `<=`（stage 大了 2 KB） |

## 结果（GPU2，8192³）

| | m6 | m7 |
|---|---|---|
| `SQ_LDS_BANK_CONFLICT` | 2.32e7 | 6.39e6（−72%） |
| `SQ_LDS_IDX_ACTIVE` | 4.47e7 | 2.79e7（−38%） |
| `ds_read_b128` cycle/条（ATT） | 7.6 | 5.7 |
| main loop cycle/趟 | 5039 | 4930 |
| **时间（`-w 1000 -i 200`，交替 7 轮中位数）** | 278.7 µs | **276.0 µs（−1.0%，7/7 轮更快）** |
| 满载时钟 | 1400 MHz | 1392 MHz |

相对主文件（286.9 µs）：−3.8%。

正确性：8192³ 随机 scale；1024×1024×512、1024×2048×1024×b2、2048×1024×7168×b3、768×1280×1536、256×256×512，`MXFP4_SFA_K_PATTERN=1`，`PERSIST_GRID` 默认 / 37 / 1，全部 VALID。无 VGPR spill。

## 下一个瓶颈

LDS 压力下降后，main loop 里 buffer_load 发射造成的空隙从约 360 升到约 500 cycle/趟。
`TA_BUSY` 各版本相同（约 44%），但 `TA_ADDR_STALLED_BY_TC_CYCLES` 从主文件 3.8e6 升到 m5 起的 5.2–6.6e6：TA 被下游 TC 反压。

## 测试

```bash
cd candidate_m7
OPUS_INCLUDE_DIR=<opus include> ./compare.sh 2 7
```
