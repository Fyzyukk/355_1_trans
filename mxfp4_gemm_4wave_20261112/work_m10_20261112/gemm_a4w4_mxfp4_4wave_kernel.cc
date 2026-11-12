#include <opus/hip_minimal.hpp>

#include "gemm_a4w4_mxfp4_4wave_kernel_template.hpp"

template __global__ void gemm_a4w4_mxfp4_4wave_kernel<gemm_a4w4_mxfp4_4wave_traits>(opus_gemm_mxfp4_scale_kargs);
