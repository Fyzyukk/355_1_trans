#!/bin/bash
# att2.sh name template.hpp : build device code object with the pin toolchain, ATT trace 8192^3 on GPU ${GPU:-6}
name=$1; tmpl=$2
SYM=_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs; L=/root/workspace/scratch_mxfp4/opt/prof/launch_g
I=/root/workspace/aiter-blockscale-mxfp8-nn-nt-tt/csrc/include; CC=/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin/clang++
d=/root/workspace/scratch_mxfp4/opt/prof/t_$name; rm -rf $d; mkdir -p $d/src
src=/root/workspace/mxfp4_gemm_4wave; cp $src/gemm_a4w4_mxfp4_4wave_kernel.cc $d/src/; cp ${TRAITS:-$src/gemm_a4w4_mxfp4_4wave_traits.hpp} $d/src/gemm_a4w4_mxfp4_4wave_traits.hpp; cp $tmpl $d/src/gemm_a4w4_mxfp4_4wave_kernel_template.hpp
cd $d/src; $CC -x hip -I$I -I. -std=c++17 -O3 -ffast-math --offload-arch=gfx950 --cuda-device-only gemm_a4w4_mxfp4_4wave_kernel.cc -o ../kernel.co || exit 1
$CC -x hip -I$I -I. -std=c++17 -O3 -ffast-math --offload-arch=gfx950 --cuda-device-only -S gemm_a4w4_mxfp4_4wave_kernel.cc -o ../kernel_gfx950.s
cd $d; HIP_VISIBLE_DEVICES=${GPU:-6} timeout 600 rocprofv3 --att 1 --att-target-cu 1 --att-shader-engine-mask 0x1 --att-buffer-size 268435456 -d att -o r -- $L kernel.co 8192 8192 8192 2 $SYM > att.log 2>&1
find $d/att -name code.json | head -1
