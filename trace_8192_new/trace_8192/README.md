# 8192x8192x8192 MXFP4 4-wave ATT trace —— 当前主文件版本（pin A/B/C + unroll 2，约 262 µs / 4.19 PF）

生成：2026-11-11 07:06 UTC，GPU2（MI355X，gfx950），rocprofv3 --att，追踪 SE0 的 CU 1：16 个 wave（4 个 block × 4 个 wave），第 1 次 dispatch。
编译器：pin 工具链 /root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build（clang 24）。

| 文件 | 内容 |
|---|---|
| `gemm_a4w4_mxfp4_4wave_kernel_template.hpp`、`gemm_a4w4_mxfp4_4wave_traits.hpp` | 本次 trace 对应的源码（与主文件相同） |
| `kernel_gfx950.s` | 汇编；主循环（一次 2 个 tile）的循环头在第 740 行附近（`Inner Loop Header`） |
| `kernel_gfx950.co` | trace 用的代码对象 |
| `att/ui_output_agent_*_dispatch_1/` | **用 ROCprof Compute Viewer 打开这个目录** |
| `att/stats_ui_output_agent_*_dispatch_1.csv` | 每条指令的 Hitcount / Latency / Stall / Idle |
| `summary.md` | 汇总（下附） |

旧版本（约 285 µs 的 bf16 版）的 trace 在 `../trace_8192_old/`。

每个 wave 总 cycle：110364（16 个 wave，4 个 block × 4 个 wave）

| 类别 | 占比 | 执行次数 | 平均 cycle/条 |
|---|---|---|---|
| MFMA | 50.8% | 65536 | 13.7 |
| buffer_load | 15.5% | 8736 | 31.3 |
| VALU | 11.6% | 33392 | 6.1 |
| waitcnt | 5.8% | 5424 | 18.9 |
| SALU | 5.3% | 16800 | 5.5 |
| ds_read | 4.4% | 17936 | 4.4 |
| barrier | 3.3% | 1072 | 55.1 |
| nop | 1.3% | 3520 | 6.6 |
| buffer_store | 1.2% | 512 | 41.1 |
| accvgpr | 0.8% | 3072 | 4.5 |

主循环里最耗时的指令（hits >= 200）：

| 地址 | 占比 | cycle/次 | 指令 |
|---|---|---|---|
| 10004 | 1.6% | 114 | `s_barrier` |
| 14320 | 1.0% | 77 | `buffer_load_dwordx4 v26, s[28:31], s67 offen lds` |
| 12976 | 0.6% | 46 | `s_waitcnt lgkmcnt(0)` |
| 13744 | 0.5% | 40 | `buffer_load_dwordx4 v18, s[20:23], s69 offen lds` |
| 12980 | 0.4% | 31 | `s_barrier` |
| 13648 | 0.4% | 31 | `buffer_load_dwordx4 v10, s[20:23], s69 offen lds` |
| 13004 | 0.4% | 29 | `v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], a[80:83], a[48` |
| 11176 | 0.4% | 28 | `buffer_load_dwordx4 v21, s[24:27], s71 offen lds` |
