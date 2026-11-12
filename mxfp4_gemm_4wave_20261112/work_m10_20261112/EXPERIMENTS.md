# work_m10 —— 在 candidate_m10 上继续优化（2026-11-12，GPU6）

模板里的实验宏（`./build_var.sh name "-DXXX"` 编到 /tmp/exp_m10/name；`prof/att.sh name .. "-DXXX"` 抓 trace，`prof/seg.py`/`quad.py`/`coexec.py` 分析）：

| 宏 | 内容 | 8192³ GPU6 | 结论 |
|---|---|---|---|
| （无） | = m10 | 226.2 | |
| NO_SF / NO_SF_G / NO_SF_R | 去掉全部 / 全局+写 panel / LDS 读+v_perm 的 scale 代码（仅对照，结果错误或需 MXFP4_UNIT_SCALE） | 208.5 / 220.0 / 214.6（base scale=1.0 时 221.9） | scale 代码 13.5 µs + scale 数据功耗 5 µs |
| NO_CSTORE | 不写 C（对照） | 212.2 | C 写回 13 µs：最慢 XCD 时钟 +2%（能耗）+ 周期 -4% |
| **SF_DMA** | 主循环 scale panel 改 LDS-DMA（不经 VGPR、无 ds_write） | 225.2（-1.0） | 保留 |
| **SF_DMA2**（配 SF_DMA） | 下一 tile 的 panel 0 / panel 1 sfb 也 DMA；prologue 计数等待 vmcnt(32)/(16)，去掉编译器插的 vmcnt(0)（它连 tile1 预取和 C store 一起等） | **223.7（-2.5，-1.1%）** | 保留，当前最好 |
| EPI_OVL | 最后一个 k-tile 先算 C01，上半 C 的转换/写 LDS 放进 C10、写回放进 C11 | +0.8 | store 被全卡同步突发写带宽卡住，挪位置无用 |
| CZ_OVL | 累加器清零挪进 prologue C00 | 无变化 | 清零位置由寄存器分配决定（pin），C++ 位置不起作用 |

trace（SF_DMA2，单 CU）：MFMA 利用率约 72%；main loop 每 k-tile 损耗集中在 C10/C01（4 个 wave 同时发 8 条 A/B LDS-DMA，TA 饱和），C11/C00 接近理想；
tile 衔接段每 tile 约 13.4k 周期（占 15%）：C 写回（全卡同步突发，64 CU 对照时 store 停顿 1262→434）、下半 C 暂存、tile1 预取发射卡、累加器清零 800。
时钟：GPU6 的 XCD0/1 比其他慢约 8%，persistent 静态分配下时间由最慢 XCD 决定。

## GPU2 复测（2026-11-12 晚）
m10 214.3 µs，SF_DMA+SF_DMA2 211.1–211.7 µs（-1.5%，约 5.20P）。

## B_DIRECT（2×2 布局下 B 直接 global→AGPR，配 SF_DMA2）
- 正确（24 用例 ×2）。踩过的坑：v_a/v_b 原来声明在 persistent 循环内，上一 tile 预取的 B0 被当死代码删掉 → 挪到循环外；
  B_b 计数要以 C11 的 scale DMA 为界（12，不是 13）。LLVM 对 gfx9 的建模：无 vscnt 时 store 与 load 同一事件类型，按序完成（SIInsertWaitcnts getVmemWaitEventType）。
- GPU2：215.3 vs 211.7（+3.5 µs，更慢）。trace：main loop 5004 vs 4788 cycle/趟；C01 12 条 VMEM（B 流量翻倍，TA 卡）、
  C00 发出的 B1 只有 C10 一个 quadrant 的时间，C01 开头编译器的 vmcnt 等待暴露延迟。2×2 下每个 B 半在 4 个 quadrant 里用 2 个，寄存器不够双缓冲，结构上藏不住。
- 结论：B 直接读要配 1×4 wave 布局（每 wave 256×64，B 不重复、可双缓冲一整个 k-tile；A 留 LDS 由 4 个 wave 共享；LDS 写减半并腾出 ~66 KB 给 C 延后写回）。

## 小方向（GPU2，2026-11-12 晚）
- **SF_SHIFT**（配 SF_DMA SF_DMA2）：scale 不再用 v_perm 拼字节；每个 LDS dword [kk0 w0, kk0 w1, kk1 w0, kk1 w1] 按 8·wave 右移（v_lshrrev_b64 一次两个 dword），MFMA opsel = 2·kk 取字节。v_perm 80 → lshr64 40，指令 2997 → 2945，无 spill。
  GPU2 7 轮：211.7 → **210.2 µs（约 5.23P）**，24 用例全对。**当前最好：-DSF_DMA -DSF_DMA2 -DSF_SHIFT**。
- SF_UNALIGN（LDS 读地址 + wave 号，省掉移位）：大 K 有错（8192³ 21.5 万个），非对齐 ds_read2st64 不可靠，放弃。
- XCD 均衡：GPU2 各 XCD 1700–1756 MHz，完全均衡上限 1.8%；每 CU 正好 4 个 tile，快 XCD 只早 ~0.13 tile，tile 粒度动态分配无效；只能 stream-K（切 ~8 份 + fp32 跨 CU 归约），净收益估计 <1%，不做。
- trace（GPU2，单 CU）：m10 main loop 5120 cyc/趟（MFMA 忙 80.0%），衔接段 20587/tile；best 4910（83.4%），18520/tile。
  打包在 traces/trace_8192_GPU2_{m10_base,best_sfsh}.tar.gz（ui_output 目录 + stats csv + kernel.co）。

## 2×2 shuffle 主线继续（GPU2，flock /tmp/gpu2.lock；三条线并行：1×4 用 GPU0，无 shuffle 路线用 GPU1）
- **PF_SPREAD**：A1/B1 预取从 C01 挪到 C11（A1）/ C00（B1）。main loop 4910 → 4814 cyc/趟（-2%），但最慢 XCD 时钟 1687 → 1669（-1.1%）；GPU2 210.0 → 209.7。保留。
- C01S（C01 的 16 DS + 8 VMEM 均匀排）：排布生效但 TA 吞吐不变，无收益。
- C 写回缓存策略 C_AUX：nt（2，现状）最好；0 / sc0 慢约 3 µs，sc1 / nt sc1 慢约 2 µs。
- A/B LDS-DMA 缓存策略 AB_AUX：0 / sc0 / sc1 / sc0 sc1 无差别；nt 明显变慢（~237 µs）。
- main loop 指令（每趟 2 K tile）：256 MFMA、64 ds_read_b128、34 buffer_load、16 ds_read2st64、22 VALU、39 SALU，已无可删。
- **ZC_PEEL**：main loop 第一趟单独实例化（main_trip lambda），C10/C01/C11 的第一条 MFMA 用 C=0，每 tile 省 192 条清零指令。48/48 正确，GPU2 209.6 → **208.8 µs（约 5.27P）**。SGPR spill 16（进 VGPR lane）。
  **当前最好：-DSF_DMA -DSF_DMA2 -DSF_SHIFT -DPF_SPREAD -DZC_PEEL**
- PEND_SPREAD（下半 C 的 16 条 store 从 prologue C00 摊到第一趟 4 个 quadrant）：24/24 正确，208.9 vs 208.9，无收益。
  衔接段的根因是全卡同步的 C 突发写（每轮约 32 MB），在一个 tile 内部挪 store 位置降不了全卡峰值；2×2 没有空余 LDS 把 C 留到下一个 tile 慢慢写。
