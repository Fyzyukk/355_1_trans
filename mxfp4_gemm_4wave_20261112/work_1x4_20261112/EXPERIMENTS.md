# work_1x4 —— 1×4 wave 布局 + B 直接 global→AGPR（2026-11-12/13，GPU6）

基线：work_m10 `-DSF_DMA -DSF_DMA2`（/tmp/exp_m10/sfdma2），GPU6 8192³ 约 223.7 µs。
构建：`./build_var.sh name "-D..."` → /tmp/exp_1x4/name；正确性 `/tmp/exp_1x4/check.sh exe 6`（6 形状 × 4 环境 × 2 遍 = 48）；
计时 `/tmp/exp_1x4/bench6.sh N dir...`（每轮前等 GPU6 空闲：另一会话会间歇在 GPU6 上跑 launch_6696_fr）；trace `prof/att.sh name .. "-D..."`。
`template_m10_ref.hpp` 是原 m10 模板（参考），新模板从头写，复用 m10 的 A LDS-DMA/读 layout、mma_scale_one、e8m0_shuffle panel DMA。

## 设计
- wave w 负责 256 行 × 列 [64w, 64w+64)。A 照旧 LDS-DMA 进 LDS（4 个 wave 共享），每个 wave 按 64 行一个 quadrant（q = (行半区 h, wave_m 奇偶)）
  读 8 条 ds_read_b128 进 AGPR（双缓冲 a0..63）；B（shuffle_weight(16,16)）每 wave 4 个 16 行块，每 (块, kk) 连续 1 KB，lane*16，
  直接 buffer_load 进 AGPR，按 K tile 奇偶双缓冲（a64..127），提前一个 K tile 发出。C：q0,q1 在 AGPR 128..255，q2,q3 在 VGPR 128..255。
- scale：A 每 K tile 8 个 dword（全部 8 个 32 行块）+ B 2 个 dword，v_perm 拼 v_sfa[4][2] / v_sfb[2]（op_sel 按 m / n 选字节）。
- 每 K tile 一个 barrier（q2→q3 之间）：q3 里发 A(t+2) 前半的 DMA（+ 偶数 tile 发 panel）、读下一 K tile 的 A q0 和 scale。
  A(t+1) 后半的 DMA 在 q0 开头，B(t+1) 分散在 q0..q2；边界等待 vmcnt(8)（比 A(t+1) 后半新的只有 8 条 B）。
- C staging 用单独的 66 KB LDS（B 不再占 LDS）：A 2×34 KB + C 半块 66 KB + scale 2×2×4 KB = 150 KB。

## 版本与结果（GPU6，8192³，交替计时；sfdma2 同轮 223.4–224.4）
| 版本 | 内容 | 时间 µs | 备注 |
|---|---|---|---|
| v1 | 首版（无 tile 衔接重叠） | 238 | 修过：prologue 的 VMEM 之间要 sched_barrier，否则计数等待失效（随机错） |
| v2 | prologue 先发 B(0)（编译器 waitcnt 合并状态）；A(t+1) DMA 拆半到 q0；B 分散到 q0..q2 | 234.7 | |
| v3 | + scale gather 挪到 q3（V3_GATHER）+ 下一 tile head 在 C 写回前发（V3_PREF）；b_soff 改 VGPR（SGPR 被提出循环后 spill，reload 带 vmcnt(0)） | 231.4 | |
| v5 | tensor 级 gmem + 每 tile SGPR 基址；K tile ≥ loops 的拷贝即下一 tile 的 head（最后 q3 读到的 A q0/scale 也是下一 tile 的），下一 tile 无 prologue；第一个 K tile 用 C=0 的 MFMA（不清零累加器）；下半 C 延后到下一 tile 第一趟写 | 226.1 | 修过：sfa_raw / v_sfa / v_sfbn 声明在 tile 循环内 → 下一 tile 拿到未定义值 |
| v6 | v5 + 最后一趟单独展开，下一 tile 选择变编译期（中间趟 SALU 62→24/趟） | 226.3 | K=512 不做跨 tile 预取（loops==2 单独实例会让 VGPR spill 50） |
| v7 | B(t+1) 全部在 q0 前 8 条 MFMA 发 | 231.3 | q0 12 条 VMEM，TA 发射卡 170/象限 |
| v8 | v6 + q0 的 MFMA 按 n 优先（B 块到达顺序） | 226.5 | 无收益 |

## trace 结论（v6，单 CU）
- 每 tile 周期：v6 91.1k vs sfdma2 89.3k（+2%）；但 v6 时钟更高：最慢 XCD 1619 vs 1586 MHz（+2.1%，B 不走 LDS、LDS 写减半省能耗），净慢约 1%。
- main loop 每 K tile 2491 vs 2394：q1/q2 约 530（接近理想 512），损耗在 q0（~690：B(t) 后几块到得晚，编译器逐条 vmcnt 等，约 110）
  和 q3（~740：每 K tile 唯一的 barrier ~80 + A/panel DMA 发射卡）。
- 衔接段：下一 tile 的 head 已经融进流水；剩 C 上半写回（stage + 16 store，~4k）和下半 stage（~1k）。

## 当前最好：v6 = `-DV4 -DV3_GATHER`（模板默认宏组合），226.3 µs vs sfdma2 223.7 µs（+1.2%，5 轮交替，48/48 正确）
SGPR spill 7（进 VGPR lane）、VGPR spill 16（只在 K=512 的 `if (loops > 2)` 汇合处，每 tile 执行一次）。

## 遗留 / 下一步
1. q0 等 B(t)：B 只能提前一个 K tile（寄存器只够双缓冲），L2/MALL 延迟 >1000 cycle；全部提前到 q0 开头又被 TA 发射卡（v7）。
   可试：B 的 4 块按 q0 / q3 末尾（上一 K tile 的 q3 里 v_b[1-P] 中 n 已用完的块）错开发，或把 A 的 DMA 挪出 q0。
2. q3 的每 K tile barrier（~80）+ DMA 发射：可把 panel DMA 挪到 q1/q2（计数要相应调整）。
3. 能耗上 1×4 已占优（同功耗时钟 +2.1%）；周期追平 sfdma2 时约快 2%。
4. 上半 C 写回（每 tile ~4k cycle）仍在关键路径上：LDS 只够放半块 C，无法两半都延后。
