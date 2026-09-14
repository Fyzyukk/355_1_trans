# 最新4-wave通用kernel ATT

采集成功，退出码0。物理GPU2=HIP2，MI355X，PCI `0000:65:00.0`；8192×8192×8192、batch1、默认BF16、4wave/tile1。采集前后设备空闲。

有效GEMM数据在 [ui_output_agent_36221_dispatch_2](ui_output_agent_36221_dispatch_2/filenames.json)：16个wave，每个wave均包含4096条16×16×128 scaled MFMA。dispatch1没有解码出的wave，查看GEMM时选择dispatch2。

- [指令统计CSV](stats_ui_output_agent_36221_dispatch_2.csv)
- [ISA与指令索引](ui_output_agent_36221_dispatch_2/code.json)
- [原始ATT](08b495a2814a/744511_36221_shader_engine_0_2.att)
- [完整命令、环境及源码/产物hash](command.json)
- [设备、code object及wave完整性检查](trace_validation.json)
- [rocprofv3日志](rocprofv3.log)

trace内code object的SHA256为 `bb91697ef424f93138c2aeba916bcb7fbb054bf41dad577786753f9d570bb107`，与最新构建逐字节相同。对应实测候选 `unified_gsfa_mk_lane_bits`；工作区排版后的HPP为 `7bfc4e877d1747d53e4d460872d897450b44868c76eba3fbe70e3d4da9f817f2`。采集使用的源码在 `source_snapshot/`，二进制在 `build_snapshot/`，不受后续工作区编辑影响。

按用户命令保留 `--att=true --att-library-path /opt/rocm/lib` 和 `-m 8192 -n 8192 -k 8192 -b 1 -v 0`，只替换可执行文件与输出目录。CLI默认warmup200、iterations100，未增加kernel过滤或iteration过滤。

首次原命令在OpenMP工具的 `omp_get_num_devices / ompt_post_init` 初始化处SIGSEGV，尚未执行GPU kernel，失败记录保留在相邻 `trace_block_scale_latest_20260914_035904/`。本次设置 `OMP_TOOL=disabled` 关闭冲突的OpenMP工具钩子，并用 `HIP_VISIBLE_DEVICES=2` 绑定GPU2。未修改编译器、profiling库或kernel。

trace用于分析指令执行；日志中的插桩计时不作为性能提升证据。独立五轮同卡测量为BF16 3.310618P、FP32 3.094809P，43组通用shape和完整8192检查通过，详见项目results目录的 `gsfa_mk_selection_20260914.json`。
