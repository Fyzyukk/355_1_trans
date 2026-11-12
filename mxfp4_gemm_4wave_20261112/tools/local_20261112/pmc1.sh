#!/bin/bash
# pmc1.sh name grid counter... : 每个计数器单独一次 pass，输出 "name counter value"（各 dispatch 求和后取第 2 次 dispatch）
n=$1; g=$2; shift 2; d=/tmp/att_local/t_$n; SYM=_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
for c in "$@"; do
  rm -rf $d/p1_$c
  ( cd $d; [ -n "$g" ] && export GRID=$g; HIP_VISIBLE_DEVICES=${GPU:-2} timeout 300 rocprofv3 --output-format csv --pmc $c -d p1_$c -o r -- ./launch kernel.co 8192 8192 8192 2 $SYM > p1_$c.log 2>&1 )
  f=$(find $d/p1_$c -name '*counter_collection.csv' | head -1)
  if [ -z "$f" ]; then echo "$n $c ERR $(grep -m1 -oE 'error code [0-9]+: [^*]*' $d/p1_$c.log)"; continue; fi
  python3 - "$f" "$n" "$c" <<'PY'
import csv,sys,collections
s=collections.defaultdict(float)
for r in csv.DictReader(open(sys.argv[1])): s[int(r['Dispatch_Id'])]+=float(r['Counter_Value'])
ks=sorted(s); print(sys.argv[2],sys.argv[3],f"{s[ks[-1]]:.4g}",f"(dispatches {len(ks)})")
PY
done
