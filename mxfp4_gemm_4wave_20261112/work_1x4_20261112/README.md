# work_1x4_20261112 —— 1×4 wave 布局 + B 直接 global→AGPR（进行中）

每个 wave 负责 256 行 × 64 列；A 走 LDS（4 个 wave 共享），B 按 shuffle_weight(16,16) 直接 buffer_load 进 AGPR（双缓冲，提前一个 K tile），不经过 LDS。
当前最好 v9s：`./build_var.sh v9s "-DV4 -DV3_GATHER -DV9 -DV9_SFR"`。正确（48/48）。GPU2（HIP 2）8192³ 205.3 µs（约 5.36P），同时段 aiter asm 205.8 µs、2×2 最好 208.5 µs。
最慢 XCD 时钟 +2.1%（LDS 写减半省能耗），但每 tile 周期 +2%（K tile 第 1 段等 B、第 4 段 barrier + DMA）。详见 EXPERIMENTS.md。
template_m10_ref.hpp 是改写前的 m10 模板（参考）。
