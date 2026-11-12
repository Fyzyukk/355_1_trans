#!/bin/bash
# Wait for an idle GPU (0% use for 15 s), then ATT-trace PR6696 (best 8192 variant) and our main kernel on it.
cd /root/workspace/scratch_mxfp4/pr6696_trace
P=/root/workspace/aiter-pr6696-small-runtime-6300-20261106/csrc/opus_gemm/gen_co/gfx950
CO=$P/opus_gemm_gfx950_4wave_pin_agpr_co_256x256x256x128_2x2_16x16x128_1x128x128_pm0_sp64_u2_gm0_wide1_vf1.co
SYM=$(/opt/rocm/llvm/bin/llvm-readelf --notes $CO | grep -oE "\.name: +opus_gemm\S+" | awk '{print $2}')
pick() { for g in 6 7 1 3 5 0 2 4; do ok=1; for i in 1 2 3 4 5 6; do u=$(rocm-smi -d $g --showuse 2>/dev/null | grep -oE "GPU use \(%\): [0-9]+" | grep -oE "[0-9]+$"); [ "${u:-100}" -eq 0 ] || { ok=0; break; }; sleep 2.5; done; [ $ok = 1 ] && { echo $g; return; }; done; }
G=""; while [ -z "$G" ]; do G=$(pick); [ -z "$G" ] && sleep 30; done
echo "IDLE GPU $G at $(date -u +%T)"
echo "timing pr6696: $(TIME=1 HIP_VISIBLE_DEVICES=$G timeout 300 ./launch_6696 $CO $SYM 8192 8192 8192 2 | grep time | awk '{print $2}' | paste -sd' ') us"
echo "timing main  : $(cd /tmp/var_s1 && for r in 1 2 3; do HIP_VISIBLE_DEVICES=$G OMP_NUM_THREADS=16 ./build/gemm_a4w4_mxfp4_4wave.exe -m 8192 -n 8192 -k 8192 -b 1 -v 0 -w 1000 -i 200 | grep -oE '[0-9.]+ us' | awk '{printf "%s ",$1}'; done) us"
rm -rf t_pr6696 && mkdir t_pr6696 && cd t_pr6696 && HIP_VISIBLE_DEVICES=$G timeout 600 rocprofv3 --att 1 --att-target-cu 1 --att-shader-engine-mask 0x1 --att-buffer-size 268435456 -d att -o run -- ../launch_6696 $CO $SYM 8192 8192 8192 2 > att.log 2>&1; cd ..
echo "pr6696 trace: $(ls -d t_pr6696/att/ui_output* 2>/dev/null)"
cd /root/workspace/scratch_mxfp4/opt/prof && GPU=$G ./att3.sh s1_idle /root/workspace/scratch_mxfp4/best_step1.hpp > /dev/null 2>&1
echo "main trace: $(ls -d /root/workspace/scratch_mxfp4/opt/prof/t_s1_idle/att/ui_output* 2>/dev/null)"
echo "after: $(rocm-smi -d $G --showuse 2>/dev/null | grep -oE 'GPU use \(%\): [0-9]+')"
echo DONE
