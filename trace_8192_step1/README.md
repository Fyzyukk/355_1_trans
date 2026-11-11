# 8192x8192x8192 MXFP4 4-wave ATT trace —— 当前主文件版本（pin A/B/C + unroll 2 + ①三处小修）

生成：2026-11-11 UTC，**GPU6**（MI355X，gfx950），rocprofv3 --att，追踪 SE0 的 CU 1：16 个 wave（4 个 block × 4 个 wave），第 1 次 dispatch。
编译器：pin 工具链 /root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build（clang 24）。
性能（GPU6，随机数据，-w 1000 -i 200）：**243.3 µs / 4.52 PF**。

| 文件 | 内容 |
|---|---|
| `gemm_a4w4_mxfp4_4wave_kernel_template.hpp`、`gemm_a4w4_mxfp4_4wave_traits.hpp` | 本次 trace 对应的源码（与主文件相同，md5 3565421…） |
| `kernel_gfx950.s` | 汇编；主循环（一次 2 个 tile）的循环头在第 737 行（`Inner Loop Header`） |
| `kernel_gfx950.co` | trace 用的代码对象 |
| `att/ui_output_agent_20361_dispatch_1/` | **用 ROCprof Compute Viewer 打开这个目录** |
| `att/stats_ui_output_agent_*_dispatch_1.csv` | 每条指令的 Hitcount / Latency / Stall / Idle |
| `summary.md` | 汇总 |

与 `../trace_8192/` 的区别：那份是 GPU2 上的旧版本（pin A/B/C + unroll 2，没有 ① 三处小修）。
同一旧版本在 GPU6 上重抓的 trace 在 /root/workspace/scratch_mxfp4/opt/prof/t_old_g6/，用于同卡对比。
