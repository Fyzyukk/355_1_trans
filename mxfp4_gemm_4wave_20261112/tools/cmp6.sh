#!/bin/bash
# cmp6.sh name1 name2 ... : alternate 8192^3 timing on GPU ${GPU:-6}, ${ROUNDS:-3} rounds
for r in $(seq 1 ${ROUNDS:-3}); do for n in "$@"; do
  t=$(cd /tmp/var_$n && HIP_VISIBLE_DEVICES=${GPU:-6} OMP_NUM_THREADS=16 timeout 300 ./build/gemm_a4w4_mxfp4_4wave.exe -m 8192 -n 8192 -k 8192 -b 1 -v 0 -w 1000 -i 200 2>&1 | grep -o '[0-9.]* us), [0-9.]* TFlops')
  echo "r$r $n $t"; sleep 2; done; done
