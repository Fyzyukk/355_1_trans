#!/bin/bash
# bench.sh ROUNDS dir... : GPU2 交替计时 8192^3 (-w 1000 -i 200)
R=$1; shift; A="-m 8192 -n 8192 -k 8192 -b 1 -v 0 -w 1000 -i 200"
rocm-smi -d ${GPU:-2} --showuse 2>/dev/null | grep 'use (%)'
for r in $(seq 1 $R); do line="r$r"; for d in "$@"; do
  t=$(HIP_VISIBLE_DEVICES=${GPU:-2} OMP_NUM_THREADS=16 $d/build/gemm_a4w4_mxfp4_4wave.exe $A 2>&1 | grep -oE '[0-9.]+ us' | head -1); line="$line  $(basename $d)=$t"; sleep 2; done; echo "$line"; done
