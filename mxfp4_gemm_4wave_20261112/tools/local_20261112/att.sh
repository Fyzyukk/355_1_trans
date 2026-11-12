#!/bin/bash
# att.sh name srcdir p|g : 本机版 att3/att4。用 pin 工具链编 device code object，GPU${GPU:-2} 上抓 8192^3 ATT
name=$1; src=$2; L=$3
P=/root/workspace/355_1_trans/mxfp4_gemm_4wave_20261112/tools/prof
SYM=_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
I=/root/workspace/aiter-opus-upstream-compare-20261016T171653Z/csrc/include
CC=/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/clang++
d=/tmp/att_local/t_$name; rm -rf $d; mkdir -p $d/src
cp $src/gemm_a4w4_mxfp4_4wave_kernel.cc $src/gemm_a4w4_mxfp4_4wave_traits.hpp $src/gemm_a4w4_mxfp4_4wave_kernel_template.hpp $d/src/
cd $d/src
$CC -x hip -I$I -I. -std=c++17 -O3 -ffast-math --offload-arch=gfx950 --cuda-device-only gemm_a4w4_mxfp4_4wave_kernel.cc -o ../kernel.co || exit 1
$CC -x hip -I$I -I. -std=c++17 -O3 -ffast-math --offload-arch=gfx950 --cuda-device-only -S gemm_a4w4_mxfp4_4wave_kernel.cc -o ../kernel_gfx950.s
/opt/rocm/bin/hipcc -O2 -I. $P/launch_$L.cpp -o ../launch 2>/dev/null || exit 1
cd $d; HIP_VISIBLE_DEVICES=${GPU:-2} timeout 600 rocprofv3 --att 1 --att-target-cu ${ATT_CU:-1} --att-shader-engine-mask 0x1 --att-buffer-size 268435456 -d att -o r -- ./launch kernel.co 8192 8192 8192 2 $SYM > att.log 2>&1
echo "$name: $(file -b kernel.co | cut -c1-40) | $(find $d/att -name code.json | head -1)"
