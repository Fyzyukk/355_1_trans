# 8192x8192x8192 MXFP4 4-wave（bf16 输出，约 285 µs / 3.85 PF）的 ATT trace

生成：2026-11-11 02:20 UTC，GPU2（MI355X，gfx950），rocprofv3 --att，
追踪 SE0 的 CU 1：16 个 wave（4 个 block × 4 个 wave），第 1 次 dispatch。

## 文件
| 文件 | 内容 |
|---|---|
| `gemm_a4w4_mxfp4_4wave_kernel_template.hpp` | 本次 trace 对应的源码（与主文件相同） |
| `kernel_gfx950.s` | 编译器输出的汇编，主循环头在第 807 行附近（`Inner Loop Header`），循环体是 `.LBB0_6` + `.LBB0_7` |
| `kernel_gfx950.co` | 代码对象（ELF），trace 用的就是它 |
| `att/ui_output_agent_*_dispatch_1/` | **用 UI 打开这个目录** |
| `att/stats_ui_output_agent_*_dispatch_1.csv` | 每条指令的 Hitcount / Latency / Stall / Idle 汇总 |
| `att/*.att`, `att/*.out`, `att/run_results.db` | 原始 trace 和代码对象（重新解码用） |
| `summary.md` | 按指令类别的时间占比、主循环里最耗时的指令 |

## 用 UI 打开
服务器上没有图形界面，需要在本地电脑上用 **ROCprof Compute Viewer**（https://github.com/ROCm/rocprof-compute-viewer ，有 Windows / Linux 安装包）：
1. 下载 `trace_8192.tar.gz` 并解压；
2. 打开 ROCprof Compute Viewer → File → Open，选中 `att/ui_output_agent_*_dispatch_1` 目录；
3. 左侧选 wave（se0_sm*_wv*），可以看到每条指令的发射时间线、stall 原因，以及源码/汇编对照。

## 复现命令
```
HIP_VISIBLE_DEVICES=2 rocprofv3 --att 1 --att-target-cu 1 --att-shader-engine-mask 0x1 --att-buffer-size 268435456 \
  -d att -o run -- /root/workspace/scratch_mxfp4/prof/launch kernel_gfx950.co 8192 8192 8192 2 \
  _Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
```
| 类别 | 占比 | 执行次数 | 平均 cycle/条 |
|---|---|---|---|
| MFMA | 43.5% | 65536 | 13.8 |
| buffer_load | 18.1% | 8704 | 43.3 |
| VALU | 12.5% | 44000 | 5.9 |
| waitcnt | 9.6% | 6128 | 32.6 |
| SALU | 6.3% | 26400 | 5.0 |
| ds_read | 5.1% | 17936 | 5.9 |
| nop | 2.0% | 5424 | 7.6 |
| buffer_store | 1.6% | 512 | 66.3 |
| barrier | 1.5% | 592 | 52.5 |

主循环里最耗时的指令（每 tile 执行一次，hits >= 400）：

| 地址 | 占比 | cycle/次 | 指令 |
|---|---|---|---|
| 11772 | 6.2% | 268 | `s_waitcnt vmcnt(0) lgkmcnt(0)` |
| 10660 | 4.0% | 176 | `buffer_load_dwordx4 v137, s[28:31], s5 offen lds` |
| 10580 | 1.6% | 67 | `buffer_load_dwordx4 v149, s[28:31], s5 offen lds` |
| 11776 | 1.1% | 50 | `s_barrier` |
| 12732 | 0.7% | 33 | `buffer_load_dwordx4 v146, s[24:27], s23 offen lds` |
| 11888 | 0.6% | 25 | `v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[56:59], a[16` |
| 11904 | 0.6% | 24 | `v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[36:39], a[8:` |
| 10500 | 0.5% | 22 | `buffer_load_dwordx4 v147, s[28:31], s5 offen lds` |

追踪到的 wave 数：16，每个 wave 的 [开始, 结束] cycle 见 att/ui_output_*/filenames.json
