// DECISIVE: map how the per-lane scale VGPR feeds V_MFMA_SCALE_F32_16X16X128_F8F6F4.
// Hypothesis (colleague's claim): scale is per-lane; lane L supplies the E8M0 scale for
// output row = L%16 and K-block = L/16 (block size 32, 4 blocks over K=128), byte = op_sel.
// => one instruction consumes A_scale[16 rows][4 kblocks] and B_scale[16 cols][4 kblocks].
//
// Test 1: ONE lane hot. A=B=1, scale_b unit. scale_a: lane HOT gets byte0=128 (x2), rest 127.
//   If mapping holds, ONLY outputs with m == HOT%16 change, and only the K-block HOT/16
//   contributes an extra 32*(2-1)=32 to C[m][*]. Since all n share the same k-sum,
//   every C[HOT%16][n] += 32. Other rows unchanged.
//
// Test 2: give block g factor 2^g via per-lane byte0 = 127+(L/16). Expect C=480 everywhere
//   (already seen) — but here we ALSO vary op_sel to prove byte selection is orthogonal.

#include "opus/opus.hpp"
#ifndef __HIP_DEVICE_COMPILE__
#include "opus/hip_minimal.hpp"
#include <cstdio>
#include <vector>
#endif
using namespace opus;

template<int OPSEL>
__global__ void probe(const int* __restrict__ dSA, fp32_t* __restrict__ dCraw){
#if defined(__gfx950__)
    int lane = (int)__builtin_amdgcn_workitem_id_x();
    const fp8_t one = __builtin_bit_cast(fp8_t,(unsigned char)0x38);
    fp8x32_t a,b;
    #pragma unroll
    for(int i=0;i<32;i++){a[i]=one;b[i]=one;}
    volatile int sa=dSA[lane]; int s_a=sa;
    volatile int sb=0x7F7F7F7F; int s_b=sb;
    fp32x4_t c{0};
    c=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        __builtin_bit_cast(i32x8_t,a),__builtin_bit_cast(i32x8_t,b),
        c,0,0, OPSEL,s_a, 0,s_b);
    #pragma unroll
    for(int i=0;i<4;i++) dCraw[lane*4+i]=c[i];
#endif
}
template __global__ void probe<0>(const int*,fp32_t*);
template __global__ void probe<1>(const int*,fp32_t*);

#ifndef __HIP_DEVICE_COMPILE__
// c_reg -> (m,n): m=lane%16, n=(lane/16)*4 + i   (canonical 16x16x128 acc layout)
static void dumpC(std::vector<float>& raw,const char* tag){
    // reconstruct C[16][16]
    int C[16][16];
    for(int L=0;L<64;L++)for(int i=0;i<4;i++){int m=L%16,n=(L/16)*4+i;C[m][n]=(int)(raw[L*4+i]+0.5f);}
    printf("%s  C[m][n]:\n",tag);
    for(int m=0;m<16;m++){printf(" m%02d:",m);for(int n=0;n<16;n++)printf(" %4d",C[m][n]);printf("\n");}
}
int main(){
    int* dS; fp32_t* dC;
    hipMalloc(&dS,64*sizeof(int)); hipMalloc(&dC,256*sizeof(fp32_t));
    auto run=[&](std::vector<int>&hS,int opsel,std::vector<float>&raw){
        hipMemcpy(dS,hS.data(),64*sizeof(int),hipMemcpyHostToDevice);
        if(opsel==0) hipLaunchKernelGGL((probe<0>),dim3(1),64,0,0,dS,dC);
        else         hipLaunchKernelGGL((probe<1>),dim3(1),64,0,0,dS,dC);
        hipDeviceSynchronize();
        raw.assign(256,-1.f); hipMemcpy(raw.data(),dC,256*sizeof(fp32_t),hipMemcpyDeviceToHost);
    };
    unsigned unit=0x7F7F7F7F;

    // Test 1: single hot lane
    for(int HOT : {0, 5, 16, 21, 32, 48+7}){
        std::vector<int> hS(64,(int)unit);
        hS[HOT]=(int)((128u)|(127u<<8)|(127u<<16)|(127u<<24)); // byte0=128 -> x2 on that lane
        std::vector<float> raw; run(hS,0,raw);
        printf("=== Test1 HOT lane=%d (expect row=%d, Kblock=%d) ===\n",HOT,HOT%16,HOT/16);
        dumpC(raw,"");
        printf("\n");
    }

    // Test 2: op_sel orthogonality. all lanes byte0=127+(L/16), byte1=127 (const)
    {
        std::vector<int> hS(64);
        for(int L=0;L<64;L++){unsigned b0=127+(L/16); hS[L]=(int)(b0|(127u<<8)|(127u<<16)|(127u<<24));}
        std::vector<float> r0,r1; run(hS,0,r0); run(hS,1,r1);
        printf("=== Test2 per-block byte0=127+g ; op_sel0 vs op_sel1(=const127) ===\n");
        printf("op_sel0 C[0][0]=%d (expect 480)   op_sel1 C[0][0]=%d (expect 128)\n",
               (int)(r0[0]+0.5f),(int)(r1[0]+0.5f));
    }
    hipFree(dS);hipFree(dC);
    return 0;
}
#endif
