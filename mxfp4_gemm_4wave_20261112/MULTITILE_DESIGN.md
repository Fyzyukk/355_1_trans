# MXFP4 GEMM 多 tile / 任意 shape 支持方案（2026-11-13）

输入约定不变（与 aiter `gemm_a4w4(bpreshuffle=True)` 一致）：A `[M, K/2]` 行主序 fp4x2；B `shuffle_weight(16,16)`；
scaleA / scaleB 为 `e8m0_shuffle`（行补到 32 的倍数——aiter 实际补到 256——、K/32 补到 8 的倍数）；C `[M, N]` bf16。

## 0. shape 分布与目标（依据 aiter gfx950 a4w4 调优表，1470 个 shape）

| M 范围 | shape 数 | aiter 选的 tile（M×N） |
|---|---|---|
| ≤ 64 | 419 | 32×128 为主（~65%），其次 64×128 |
| 65–512 | 370 | 32×128、64×128、96×128、128×128 |
| 513–1024 | 135 | 256×256、128×128、128×256、160×256 |
| 1025–4096 | 308 | 256×256 为主，其次 160/192×256 |
| > 4096 | 238 | 256×256 为主，其次 128×512 |

aiter 全部 splitK = 0。选小 tile 的原因有两个：M 小时大 tile 大量空算；中等 shape 时 tile 数填不满 256 个 CU（例如 1024×4096 用 256×256 只有 64 个 tile，CU 利用 25%；用 128×128 有 256 个 tile）。

## 1. 大 / 中 shape 方案：参数化的 2×2 流水线（M ≳ 128）

### 1.1 支持的 tile（B_M × B_N × B_K）

| tile | 每 wave 负责 | E_M × E_N | 每 quadrant MFMA | 用途 |
|---|---|---|---|---|
| 256×256×256 | 128×128 | 4×4 | 32 | 大 shape（现有，m10 zc） |
| 256×128×256 | 128×64 | 4×2 | 16 | M 大、N 小或 N 方向 tile 数不足 |
| 128×256×256 | 64×128 | 2×4 | 16 | N 大、M 中等 |
| 128×128×256 | 64×64 | 2×2 | 8 | 中等 shape（M 128–1024），填满 CU |

B_K 固定 256（一个 K tile = 两条 16x16x128 MFMA 的 K）。暂不做 160/192 这类非 2 的幂（需要不整除的 wave 划分），也不做 128×512（2×2 下 B 寄存器翻倍，放不下）。

### 1.2 怎么做：把写死的 256 改成由 B_M / B_N 推导

不是删断言，而是让下面这些量全部成为 traits 里的 constexpr，并让断言检查推导后的一致性：

| 现在写死的 | 推导方式 |
|---|---|
| E_M、E_N（每 wave 每半区的 16 行/列块数，现 4） | `HALF_B_M / (W_M * T_M)`、`HALF_B_N / (W_N * T_N)` |
| 每 wave 每半区 A/B 的 LDS-DMA 条数（`a_buffer_load_insts` 现 4，断言 "4 issues per half"） | `HALF_B_M * B_K_BYTES / (BLOCK_SIZE * VEC_A)`（128 行 → 2）；make_layout_ga/gb/sa/sb 的 issue 维度随之变 |
| LDS stage 大小、chunk 编号、C 暂存（半块 C 正好占 stage 1） | 按 B_M、B_N 计算；C 暂存检查 `HALF_B_M * (B_N + 8) * 2 <= stage 字节数` |
| scale panel（断言 `B_M == NUM_WAVES * WARP_SIZE`：256 线程一线程一行） | e8m0_shuffle 下 panel 是连续的 32 行块：256 行 = 8 块 × 512 B（2 个 K tile）。B_M=128 时只有 4 块：让 2 个 wave 各读 2 块，另 2 个 wave 读 B 的（或每 wave 读一块），DMA 条数按 `rows / 32 * 512 / 1024` 推导 |
| 调度模板（每 4 条 MFMA 插 1 条 load，按 32 条 MFMA 写死） | 按每 quadrant 的 MFMA 数和 DS / VMEM 条数生成，例如 `sched_spread<MFMAS, DS, VMEM>()` |
| 手写 vmcnt（B_b 的 `vmcnt(8)`、prologue 的 `vmcnt(16/32)`） | 写成公式：等待点之后发出的 VMEM 条数（只能计入比"必须完成的那条"更新的指令） |
| SF_SHIFT 的移位、opsel 选字节 | 与 E_M / E_N 相关的循环上界改成推导值 |

实施顺序：先把 256×256 改成参数化写法并确认生成的代码与现版本逐字节相同；再实例化 128×128（最小，最容易暴露漏改的地方）；再 256×128、128×256。

### 1.3 tile 选择
先用启发式：在候选 tile 中选"轮数 × 单 tile 时间"最小的。轮数 = ceil(tile 数 / CU 数)，单 tile 时间按 tile 面积和效率系数估计。之后像 aiter 一样对目标 shape 实测调优，生成 (M, N, K) → tile 的 csv。

## 2. 小 shape 方案：单独的流水线（M ≲ 64，必要时到 128）

小 M 时瓶颈是读权重 B 的带宽（B 有 N×K/2 字节，A 只有几 KB），2×2 结构不合适：每 wave 只剩 ≤16 行，A、B 都过 LDS、每 K tile 两个 barrier，开销占比太大。

### 2.1 支持的 tile
| tile（M×N×K） | 用途 |
|---|---|
| 16×128×256、32×128×256 | M ≤ 32（decode），对应 aiter 最常用的 32x128 |
| 64×128×256、64×256×256 | M 33–64（以及 128 以内的备选） |

### 2.2 怎么做
- **wave 布局 1×4**：每个 wave 负责 B_N/4 列、全部 B_M 行（沿用 1×4 v9s 的思路）。
- **B 直接读进寄存器**：shuffle_weight(16,16) 下每个 MFMA 操作数是连续 1 KB，`buffer_load` 直接进 AGPR，双缓冲、提前 1–2 个 K tile，不过 LDS；目标是把 HBM 带宽用满。
- **A 很小**：每个 K tile 只有 B_M × 128 B（≤ 8 KB），一次 LDS-DMA 进 LDS 由 4 个 wave 共享，或每个 wave 直接读。
- **scale**：scaleB 每 wave 只要自己的列，直接读进 VGPR；scaleA 很小，跟 A 一起进 LDS。沿用"原始 dword + 编译期 opsel"。
- **CU 利用**：N=4096、B_N=128 只有 32 个 tile，填不满 256 个 CU。先做 split-K 作为可选项（K 切几段，fp32 部分和用原子加或二次归约），由选择表决定是否开启；aiter 表里没用 split-K，所以先不做、等实测证明有收益再加。
- C 很小，直接从寄存器写（或经一小块 LDS 合并成整行写）。

## 3. 边界处理：在 kernel 里处理，不在 host 上补 pad

不补 pad：补 pad 要在每次调用时多拷一遍 A/B/C、多占显存，aiter asm 也不这么做。做法如下：

| 边界 | 处理 |
|---|---|
| **M 不是 B_M 的倍数** | A 的 buffer range 设成 `(M - row) * stride_a`，超出的行读回 0；C 的 range 设成 `(M - row) * stride_c`，超出的行写被丢弃。scaleA 由 e8m0_shuffle 补到 256 行，补出来的值是 `torch.empty` 的随机值（可能是 0xFF = NaN），但它只影响 C 的第 ≥ M 行，这些行不会写出，安全。 |
| **N 不是 B_N 的倍数** | B 的 range 设成 `(N - col)` 对应的字节数，超出的列（B 的行）读回 0；scaleB 同上。**C 的列需要显式屏蔽**：C 行主序，一行越过 N 的部分是下一行的开头，不会越界，所以写回时按列判断（每条 store 写 8 个 bf16 = 16 B，N 是 16 的倍数时一条 store 不会跨过 N，按 16 B 块用 exec mask 屏蔽即可）。 |
| **K** | 要求 K 是 256 的倍数（一个 K tile）。现在要求 512 是因为 scale panel 一次读 2 个 K tile；放宽到 256 时最后一个 panel 只用前一半，后一半读到的是补齐值或越界的 0，不会参与计算。A 一行越过 K 的部分是下一行数据，不能靠越界返回 0，所以 K 不对齐的情况不支持（MXFP4 模型的 K 基本都是 256 的倍数）。 |
| **输入布局要求** | B 的 shuffle_weight(16,16) 要求 N 是 16 的倍数、K/2 是 32 的倍数；与 aiter 接口一致，在 host 端检查。 |
| **persistent 分配** | tile 数 = ceil(M / B_M) × ceil(N / B_N) × batch，grid = min(tile 数, CU 数)；尾块和完整块走同一份代码（只靠 range 和列 mask）。 |

## 4. 验证
- 正确性矩阵在现有 6 个形状之外加入不对齐形状：M ∈ {1, 16, 100, 300, 1000}，N ∈ {128, 400, 1280, 4112}，K ∈ {256, 768, 7168}，batch 1/2；每个都配 默认 / MXFP4_SFA_K_PATTERN=1 / PERSIST_GRID=37 / PERSIST_GRID=1。
- 性能：每个 tile 在它的目标 shape 上与 aiter asm 同卡对比；256×256 在 8192³ 上必须与现版本持平。

## 5. 分工
- 线 1：在 2×2（work_m10 zc 为基础）上做 §1.2 的参数化 + §3 的边界处理，先 128×128 跑通正确性，再 256×128 / 128×256，最后性能对比。
- 线 2：查 2×2 移植（地址简化、REMAP）实测变慢的原因（trace + PMC）。
- 小 shape 流水线（§2）在线 1 稳定后再开。
