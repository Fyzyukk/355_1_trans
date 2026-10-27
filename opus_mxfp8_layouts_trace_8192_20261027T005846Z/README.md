# MXFP8 NT / TT / NN / TN — 8192 × 8192 × 8192

四种 layout 均已完成 rocprofv3 ATT 采集和自动解码。NT 使用原有
`kid9501 + B preshuffle`；TT / NN / TN 使用已验证的独立四-wave kernel。
输入准备与 NT 的 B preshuffle 位于 `prepare_inputs_8192x8192x8192` 区间。
每种先运行四次 warmup，ATT 采集该符号的第五次 dispatch。

四份采集均使用物理 GPU 4（MI355X）、CU 1、SE mask `0x1`、SIMD mask
`0xF`、activity interval 8。GPU 为共享设备；这里记录的是 trace，未进行
设备隔离或独立性能基准。

## ROCprof Compute Viewer

使用 [官方桌面 Viewer](https://github.com/ROCm/rocprof-compute-viewer/releases)。
在 `Menu → Import → Rocprofv3 UI Output` 中选择下表目录，或执行
`rocprof-compute-viewer <ui_output目录>`。如果下载 ZIP，先解压整个包。

| Layout | 已解码 ATT 目录 | Perfetto 文件 | 完整 wave 数 |
|---|---|---|---:|
| NT | [nt/ui_output_agent_26487_dispatch_13](nt/ui_output_agent_26487_dispatch_13/) | [nt_results.pftrace](nt/nt_results.pftrace) | 16 |
| TT | [tt/ui_output_agent_59969_dispatch_13](tt/ui_output_agent_59969_dispatch_13/) | [tt_results.pftrace](tt/tt_results.pftrace) | 16 |
| NN | [nn/ui_output_agent_53036_dispatch_11](nn/ui_output_agent_53036_dispatch_11/) | [nn_results.pftrace](nn/nn_results.pftrace) | 12 |
| TN | [tn/ui_output_agent_61871_dispatch_12](tn/ui_output_agent_61871_dispatch_12/) | [tn_results.pftrace](tn/tn_results.pftrace) | 12 |

Viewer 中可以查看 wave 时间线、对应 ISA、等待状态和 SQ 活动曲线。
对照四个 wave 时，可选 SE0 / SIMD0–3 的第一组 wave；每个完整 wave
执行 4096 条 scaled MFMA，即 64 个 K128 tile。

## Perfetto

打开 [ui.perfetto.dev](https://ui.perfetto.dev/)，选择 `Open trace file`，载入
对应的 `*_results.pftrace`。这些文件提供 kernel dispatch、HIP runtime 和
ROCTx 标注；指令级视图使用上面的 ATT 目录。

## 文件与验证

每个 layout 目录包含原始 `.att`、捕获的 `.out` code object、已解码
`ui_output_*`、逐指令统计 `stats_ui_*.csv`、kernel trace CSV 和 Perfetto 文件。
`manifest.json` 记录采集参数、目标 kernel、CO SHA256、文件路径和验证结果；
`capture_commands.json` 记录四个 rocprofv3 命令，`capture.py` 是本次固定
8192³ shape 的 runner。日志未发现丢失、溢出或解码失败；全部 wave 的
`num_insts` 与 `num_stitched` 一致，MFMA 总数与完整 wave 数相符。
