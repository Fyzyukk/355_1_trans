#!/bin/bash
# bench6.sh <name> <template.hpp> : build in /tmp/var_<name>, check correctness (2 shapes), no timing.
# timing: cmp6.sh name1 name2 ... (alternating, GPU6, -w 1000 -i 200)
name=$1; tmpl=$2
d=/tmp/var_$name; rm -rf $d; mkdir -p $d; src=/root/workspace/mxfp4_gemm_4wave
cp $src/*.cc $src/Makefile $src/gemm_a4w4_mxfp4_4wave_traits.hpp $d/; cp $tmpl $d/gemm_a4w4_mxfp4_4wave_kernel_template.hpp
[ -n "$HOST" ] && cp $HOST $d/gemm_a4w4_mxfp4_4wave_host.cc
[ -n "$TRAITS" ] && cp $TRAITS $d/gemm_a4w4_mxfp4_4wave_traits.hpp
cd $d; make >build.log 2>&1 || { echo "$name BUILD FAIL"; grep error build.log | head -5; exit 1; }
ok=""; for a in "-m 256 -n 256 -k 512 -b 1" "-m 512 -n 768 -k 2560 -b 2"; do
  r=$(HIP_VISIBLE_DEVICES=${GPU:-6} timeout 120 ./build/gemm_a4w4_mxfp4_4wave.exe $a -v 1 -w 1 -i 1 2>&1 | grep -o 'VALID\|FAIL'); ok="$ok$r "; done
echo "$name built check=[$ok]"
