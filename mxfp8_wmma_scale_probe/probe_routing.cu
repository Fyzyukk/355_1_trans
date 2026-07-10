// Probe for __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4 (gfx950).
// Answers two questions with raw builtin calls (no opus wrapper) so nothing is hidden:
//   Q1. E8M0 scale byte encoding: is a single instr's C = 128 * scale? what byte = unit?
//   Q2. Scale routing: does the packed-int32 route its 4 bytes to the 4 K-segments
//       [0,32)/[32,64)/[64,96)/[96,128) (R1a auto per-32-group), or does op_sel pick
//       ONE byte applied to all 128 K (R2)?
//
// Builtin signature (verified via clang error probing): 9 args, all trailing = int,
// args 4/5 (cbsz/blgp = fmt codes) and 6/8 (op_sel_a/b) are compile-time immediates,
// args 7/9 (scale_a/scale_b) are runtime int32 packing four E8M0 bytes.
//   f(a256, b256, c, cbsz, blgp, op_sel_a, scale_a, op_sel_b, scale_b)
//
// Data: A[m][k]=1, B[k][n]=1 (fp8 e4m3 1.0 = 0x38). With unit scale, C[m][n]=128.
// For routing test we set A=1, and give scale_b four distinct bytes; A scale = unit.

#include "opus/opus.hpp"
#ifndef __HIP_DEVICE_COMPILE__
#include "opus/hip_minimal.hpp"
#include <cstdio>
#include <cmath>
#include <vector>
#endif

using namespace opus;

// One instruction, A=B=1. scale_a and scale_b are full packed int32 passed from host.
// op_sel_a/op_sel_b are template immediates.
template<int OPSEL_A, int OPSEL_B>
__global__ void probe(int scale_a, int scale_b, fp32_t* __restrict__ ptr_c) {
#if defined(__gfx950__)
    int lane = (int)__builtin_amdgcn_workitem_id_x();
    const fp8_t one = __builtin_bit_cast(fp8_t, (unsigned char)0x38);  // e4m3 1.0
    fp8x32_t a_reg, b_reg;
    #pragma unroll
    for (int i = 0; i < 32; i++) { a_reg[i] = one; b_reg[i] = one; }

    // Defeat constant-folding of the scale args (CK XXX note: constant scale is
    // reinterpreted as an F32 constant, changing the encoding).
    volatile int va = scale_a, vb = scale_b;
    int sa = va, sb = vb;

    fp32x4_t c_reg{0};
    c_reg = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                __builtin_bit_cast(i32x8_t, a_reg),
                __builtin_bit_cast(i32x8_t, b_reg),
                c_reg, 0, 0, OPSEL_A, sa, OPSEL_B, sb);

    int row = lane % 16, gk = lane / 16;
    #pragma unroll
    for (int i = 0; i < 4; i++) ptr_c[(gk * 4 + i) * 16 + row] = c_reg[i];
#endif
}

template __global__ void probe<0,0>(int,int,fp32_t*);
template __global__ void probe<1,1>(int,int,fp32_t*);
template __global__ void probe<2,2>(int,int,fp32_t*);
template __global__ void probe<3,3>(int,int,fp32_t*);

#ifndef __HIP_DEVICE_COMPILE__

static unsigned pack4(unsigned b0, unsigned b1, unsigned b2, unsigned b3) {
    return b0 | (b1<<8) | (b2<<16) | (b3<<24);
}

int main() {
    fp32_t* d_c = nullptr;
    if (hipMalloc(&d_c, 256*sizeof(fp32_t)) != hipSuccess) { printf("malloc fail\n"); return 1; }
    std::vector<float> hC(256, -1.f);

    auto run = [&](int opsel, int sa, int sb)->float{
        switch(opsel){
            case 0: hipLaunchKernelGGL((probe<0,0>), dim3(1),64,0,0, sa,sb,d_c); break;
            case 1: hipLaunchKernelGGL((probe<1,1>), dim3(1),64,0,0, sa,sb,d_c); break;
            case 2: hipLaunchKernelGGL((probe<2,2>), dim3(1),64,0,0, sa,sb,d_c); break;
            default:hipLaunchKernelGGL((probe<3,3>), dim3(1),64,0,0, sa,sb,d_c); break;
        }
        hipDeviceSynchronize();
        hipMemcpy(hC.data(), d_c, 256*sizeof(float), hipMemcpyDeviceToHost);
        return hC[0];
    };

    auto f2i = [](float f){ int i; __builtin_memcpy(&i,&f,4); return i; };

    // ---- Q1: scale is the E8M0 *exponent byte replicated*, hypothesis: value = 2^(byte-127).
    //   HW may instead read byte0 as an 8-bit e8m0 directly: value = 2^(byte-127).
    printf("=== Q1: e8m0-byte interpretation (scale = 0xBBBBBBBB), expect C=128*2^(b-127)^2 ===\n");
    for (int b = 125; b <= 129; b++) {
        unsigned p = pack4(b,b,b,b);
        printf("  byte=%3d -> C=%g  (pred if e8m0: 128*2^(2*(b-127))=%g)\n",
               b, run(0,(int)p,(int)p), 128.0*exp2(2.0*(b-127)));
    }
    printf("\n");

    (void)f2i;

    // ---- Q2: ROUTING. A=1, scale_a=unit(0x7F7F7F7F). scale_b bytes={127,128,129,130}
    //   => per-byte factors {1,2,4,8}. Sweep op_sel_b (a stays op_sel=b here since template
    //   ties them; A unit so A's byte choice is irrelevant).
    //   R2  (op_sel picks 1 byte for ALL 128K): C = 128 * 2^(byte[op]-127) = 128,256,512,1024
    //   R1a (auto per-32-group, byte s -> K-seg s): C = 32*(1+2+4+8) = 480, ~const across op
    printf("=== Q2: routing. scale_a=unit, scale_b bytes={127,128,129,130} ===\n");
    unsigned sap = pack4(127,127,127,127);
    unsigned sbp = pack4(127,128,129,130);
    for (int op = 0; op < 4; op++) {
        float c = run(op, (int)sap, (int)sbp);
        printf("  op_sel=%d -> C=%g   [R2 pred=%g]\n", op, c, 128.0*exp2((double)op));
    }
    printf("  R2 signature: 128,256,512,1024   |   R1a signature: ~480 constant\n\n");

    // ---- Q2b: single hot byte in seg0 only: bytes={127,-inf,-inf,-inf} approximated by
    //   {127,0,0,0} (byte0=2^0, others=2^-127~0). If per-128 -> C=128 (op0) ; if per-32-group
    //   -> only seg0 contributes -> C=32.
    printf("=== Q2b: scale_b={127,0,0,0} (only byte0 hot). per-128 -> ~128, per-32grp -> ~32 ===\n");
    unsigned sbhot = pack4(127,0,0,0);
    for (int op = 0; op < 4; op++)
        printf("  op_sel=%d -> C=%g\n", op, run(op,(int)sap,(int)sbhot));

    hipFree(d_c);
    return 0;
}
#endif
