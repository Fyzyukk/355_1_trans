#!/bin/bash
# compare.sh [GPU] [ROUNDS] : 8192^3 correctness check + alternating timing of the main kernel vs this candidate.
# Main kernel is rebuilt from ../ (non-persistent host); candidate from this directory (persistent host).
GPU=${1:-2}; ROUNDS=${2:-7}; D=$(cd $(dirname $0) && pwd); M=$(dirname $D)
rm -rf /tmp/cmp_main && mkdir -p /tmp/cmp_main && cp $M/*.cc $M/*.hpp $M/Makefile /tmp/cmp_main/ && (cd /tmp/cmp_main && make >/dev/null 2>&1) || { echo "main build failed"; exit 1; }
(cd $D && make >/dev/null 2>&1) || { echo "candidate build failed"; exit 1; }
rocm-smi --showuse --showpower 2>/dev/null | grep -E "GPU\[$GPU\]"
A="-m 8192 -n 8192 -k 8192 -b 1"
for e in /tmp/cmp_main $D; do echo "verify $(basename $e): $(HIP_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=16 $e/build/gemm_a4w4_mxfp4_4wave.exe $A -v 1 -w 1 -i 1 2>&1 | grep -oE 'VALID|FAIL')"; done
for r in $(seq 1 $ROUNDS); do for e in /tmp/cmp_main $D; do
  echo "r$r $(basename $e): $(HIP_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=16 $e/build/gemm_a4w4_mxfp4_4wave.exe $A -v 0 -w 1000 -i 200 2>&1 | grep -oE '[0-9.]+ us\), [0-9.]+ TFlops')"; sleep 2; done; done
