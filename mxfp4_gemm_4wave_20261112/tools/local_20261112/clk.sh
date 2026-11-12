#!/bin/bash
R=/root/workspace/355_1_trans/mxfp4_gemm_4wave_20261112
for v in "$@"; do
  sleep 20; n=$(basename $v)
  HIP_VISIBLE_DEVICES=${GPU:-2} OMP_NUM_THREADS=16 $v/build/gemm_a4w4_mxfp4_4wave.exe -m 8192 -n 8192 -k 8192 -b 1 -v 0 -w 30000 -i 200 > /tmp/att_local/clk_$n.txt 2>&1 &
  pid=$!; sleep 5; s=""; p=""
  for i in 1 2 3 4 5 6; do o=$(rocm-smi -d ${GPU:-2} --showclocks --showpower 2>/dev/null)
    s="$s $(echo "$o" | grep -oP 'sclk clock level: \d+: \(\K[0-9]+')"; p="$p $(echo "$o" | grep -oP 'Package Power \(W\): \K[0-9.]+')"; done
  wait $pid; echo "$n: sclk[$s ] W[$p ] -> $(grep -oE '[0-9.]+ us\)' /tmp/att_local/clk_$n.txt)"
done
