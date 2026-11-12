# 8-wave（每 SIMD 2 个 wave）可行性分析 —— 2026-11-12

结论：**不建议做**。估计相对 m10 在 −8% 到 0% 之间，最可能慢 3–6%；工作量 2–3 人周。
意外发现：同一 C 上的 MFMA 累加链从 2 条加长到 4 条，实测 +1.7~2.3%（有 LDS 读时），是减少每 FLOP 能耗的方向，正在单独做原型。

测量卡：HIP 设备 3（PCI 0000:15:00.0），对应 **rocm-smi 的 GPU[1]**（HIP 编号和 rocm-smi 编号不一定一致，采时钟前先核对）。

## 1. PR6696 wave8 怎么切（wave8n4，kid30，256×256×128 fp8）

- 512 线程，1 block/CU → 2 wave/SIMD；wave 网格 T_M=2 × T_N=4（`opus_gemm_common.py:1183-1217`，traits `opus_gemm_traits_a8w8_scale_gfx950.cuh:907-917`）。
- 每 wave C 128×64；寄存器 = 累加 128 + A 64（LDS）+ B 32（全局直接进 VGPR）= 224，traits:1162-1183 用 static_assert 卡在 224（2 wave/SIMD 时每 wave 上限 256）。B 单缓冲，靠另一个 wave 掩盖延迟；无 pin。
- LDS 只放 A（3 stage 约 99 KB）+ scale；所有 wave 都发 A 的 LDS-DMA，没有 producer/consumer 分工；每 K tile 1 个 barrier；MFMA 段 `s_setprio(1)`。
- persist 版 C 直接从寄存器写出，不经 LDS。

## 2. 移植到我们的 kernel 的两种设计

| 每 CU 每 K tile | m10 | 设计 A（2×4，A/B 都走 LDS） | 设计 B（照搬 PR6696，B 直接进 VGPR） |
|---|---|---|---|
| MFMA | 512 | 512 | 512 |
| LDS→VGPR | 128 KB | **192 KB（+50%）** | 128 KB |
| LDS-DMA 写入 | 64 KB | 64 KB | 32 KB |
| 全局访存指令（TA） | 64 | 64 | **96（+50%）** |
| h1（写回推迟，约值 1%） | 有 | 寄存器不够 | 放不下 |

8 个 wave 平分 256×256 时，每 wave（A 行 + B 行）最少 192，+50% 的读取量是下限。设计 A 寄存器约 254/256，只能单缓冲。

## 3. 实测（`w8bench.cc`，`results2.txt`）

同一份二进制分别以 256 线程（1 wave/SIMD）和 512 线程（2 wave/SIMD）启动，每 SIMD 工作量相同；每 32 条 MFMA 配 NDS 条 `ds_read_b128` 和 NBL 条 LDS-DMA，DMA 按真实 8192³ tile 顺序。

| 配置 | 1 wave/SIMD | 2 wave/SIMD |
|---|---|---|
| 只有 MFMA | 5.42 PF @1354 MHz | 5.44 PF（+0.4%） |
| NDS=8（m10 的读比例） | 3.956 PF，1319 W，利用率 59.8% | 4.191 PF @1390 MHz，1364 W（+5.9%） |
| NDS=12（设计 A 的读量） | 3.796 | **4.064** |
| NDS=16 | 3.691 | 3.961 |
| m10 真实 GEMM（同卡） | 271.0 µs，**4.056 PF**，1456 MHz，1382 W | — |

- 只有 MFMA 时，2 wave 填气泡几乎没收益；2 wave 时每 32 条 MFMA 每多 1 条 ds_read 约 −0.7%，设计 A 的 +50% 读量约 −3%。
- 微基准里 +5.9% 是相对一个 59.8% 利用率的差基线；m10 已经和 2 wave 微基准在同一时钟/功耗点（功耗墙上）。

## 4. 累加链长度（`chain/`）

| 同一 C 上连续 MFMA 条数 | 1 | 2（m10） | 4 | 8 |
|---|---|---|---|---|
| 只有 MFMA（PF） | 5.43 | 5.43 | **5.60** | 5.61 |

- 链长 4 时时钟 1356 → 1400 MHz（每 FLOP 能耗更低）；加上 1:4 的 LDS 读后，链长 4 比 2：1 wave +2.3%，2 wave +1.7%。
- 链长 1 和 2 实测相同：kk0/kk1 配对本身不省能耗，要到 4 以上才省。
- 32×32×64 MFMA 更差（5.07–5.14 PF，−6%），排除。

## 5. 能耗判断

设计 A 每 FLOP 搬运更多（LDS +64 KB/K tile，scale gather 和地址计算在 8 个 wave 上各做一遍），只能填气泡、不能省能耗；设计 B 把 LDS 读换成 +50% 的 TA 请求，按 m8/m9 的经验更贵（估计）。m10 的 MFMA 忙碌已有 84.7%，2 wave 还能拿到的估计不超过约 1%，抵不过约 3% 的额外读取和约 1% 的 h1。

## 文件

- `w8bench.cc` / `run.sh` / `results1.txt` / `results2.txt`：1 vs 2 wave/SIMD 仿 GEMM 微基准（宏 NDS/NBL/DEPPAIR）及原始数据
- `chain/chain.cc`、`chainlds.cc`、`chain32.cc`：累加链长度测试（只 MFMA / 带 LDS 读 / 32×32×64）
- `mfma_ceiling.cc`：原工具加 kernel 内时钟测量（这张卡 5.57 PF @1382 MHz，利用率 96%）
