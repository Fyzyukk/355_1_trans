#pragma once

#include <hip/hip_runtime.h>
#include "traits.hpp"

namespace blockscale_generic {
template<class Traits>
__global__ void gemm_a8w8_mxfp8_scale_kernel(opus_gemm_scale_kargs);
}

// CLI and C ABI use the same runtime-K kernel, including the 8192 benchmark.
template<bool OutputBF16>
inline void launch_blockscale_kernel(const opus_gemm_scale_kargs& args,
                                    dim3 grid, dim3 block,
                                    hipStream_t stream = nullptr) {
    blockscale_generic::gemm_a8w8_mxfp8_scale_kernel<FourWaveTraits<OutputBF16>>
        <<<grid, block, 0, stream>>>(args);
}
