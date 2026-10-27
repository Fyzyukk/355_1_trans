#!/usr/bin/env python3
"""One warmed-up dense MXFP8 layout at M=N=K=8192 for rocprofv3."""

import argparse
import ctypes
import json
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))

import torch
from aiter.ops.opus import opus_gemm
from aiter.ops.opus.gemm_mxfp8_layouts import (
    opus_gemm_mxfp8_nn,
    opus_gemm_mxfp8_tn,
    opus_gemm_mxfp8_tt,
)
from aiter.ops.shuffle import shuffle_weight

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--layout", choices=("nt", "tt", "nn", "tn"), required=True)
args = parser.parse_args()

roctx = ctypes.CDLL("/opt/rocm/lib/librocprofiler-sdk-roctx.so")
roctx.roctxRangePushA.argtypes = [ctypes.c_char_p]
roctx.roctxRangePushA.restype = ctypes.c_int
roctx.roctxRangePop.argtypes = []
roctx.roctxRangePop.restype = ctypes.c_int

torch.manual_seed(8192)
m = n = k = 8192
roctx.roctxRangePushA(b"prepare_inputs_8192x8192x8192")
a = torch.randn((m, k), device="cuda").to(torch.float8_e4m3fn)
b = torch.randn((k, n), device="cuda").to(torch.float8_e4m3fn)
sfa = torch.full((k // 128, m), 127, device="cuda", dtype=torch.uint8).t()
sfb = torch.full((n // 128, k // 128), 127, device="cuda", dtype=torch.uint8)
out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
if args.layout == "nt":
    matrix_a = a
    matrix_b = shuffle_weight(b.t().contiguous(), layout=(16, 16))
    def launch():
        return opus_gemm(matrix_a, matrix_b, out, kid=9501,
                         layout="bpreshuffle", x_scale=sfa, w_scale=sfb)
elif args.layout == "tt":
    matrix_a = a.t().contiguous()
    matrix_b = b.t().contiguous()
    def launch():
        return opus_gemm_mxfp8_tt(matrix_a, matrix_b, sfa, sfb, out)
elif args.layout == "nn":
    matrix_a, matrix_b = a, b
    def launch():
        return opus_gemm_mxfp8_nn(matrix_a, matrix_b, sfa, sfb, out)
else:
    matrix_a, matrix_b = a.t().contiguous(), b
    def launch():
        return opus_gemm_mxfp8_tn(matrix_a, matrix_b, sfa, sfb, out)
torch.cuda.synchronize()
roctx.roctxRangePop()

# Four warmups. --kernel-iteration-range 5-5 selects the final dispatch
# of this symbol for ATT while Perfetto keeps the named surrounding ranges.
roctx.roctxRangePushA((args.layout.upper() + " warmup x4").encode())
for _ in range(4):
    launch()
torch.cuda.synchronize()
roctx.roctxRangePop()

roctx.roctxRangePushA((args.layout.upper() + " ATT 8192x8192x8192").encode())
launch()
torch.cuda.synchronize()
roctx.roctxRangePop()
print(json.dumps({"layout": args.layout, "shape": [m, n, k],
                  "gpu": torch.cuda.get_device_name(0),
                  "warmup_dispatches": 4, "att_iteration": 5,
                  "output_shape": list(out.shape)}), flush=True)
