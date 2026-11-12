# work_m10_20261112 —— m10 之后的实验版本（宏开关）

当前最好：`make EXTRA="-DSF_DMA -DSF_DMA2 -DSF_SHIFT -DPF_SPREAD -DZC_PEEL"`（GPU2 208.8 µs，约 5.27P）。
GPU2 8192³：m10 214.3 µs → 209.7 µs（约 5.24P）。各宏的含义、数据和失败尝试见 EXPERIMENTS.md。

trace（GPU2，单 CU，../traces/）：
- trace_8192_GPU2_m10_base.tar.gz：m10 基线
- trace_8192_GPU2_best_pfspread.tar.gz：当前最好（上面四个宏）
解开后用 ROCprof Compute Viewer 打开 ui_output_agent_* 目录；kernel.co 是对应的 code object。
prof/：att.sh（抓 trace）、launch.cpp（persistent launcher）、seg.py / quad.py / coexec.py（分析）。
