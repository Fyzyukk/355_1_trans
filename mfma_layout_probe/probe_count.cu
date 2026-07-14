// Cleanest possible: A=B=all 1.0 (value independent of any layout guess).
// One instruction 16x16x128. Baseline every C = 128 (sum of 128 k, unit scale).
// Boost ONE lane's scale x2 (byte0 128). The delta on the single output line that
// changes = number of K-elements that lane contributes.
//   delta==32  => that lane holds 32 contiguous K = ONE full 32-block (BLOCK model)
//   delta==8   => that lane holds 8 K per block spread over 4 blocks (REP4 model)
// Also report WHICH (m,n) reacted, to nail lane -> (index, block) map.

#include "opus/opus.hpp"
#ifndef __HIP_DEVICE_COMPILE__
#include "opus/hip_minimal.hpp"
#include <cstdio>
#include <vector>
#endif
using namespace opus;

__global__ void probe(const int* __restrict__ dSA, fp32_t* __restrict__ dC){
#if defined(__gfx950__)
    int lane=(int)__builtin_amdgcn_workitem_id_x();
    const fp8_t one=__builtin_bit_cast(fp8_t,(unsigned char)0x38);
    fp8x32_t a,b;
    #pragma unroll
    for(int i=0;i<32;i++){a[i]=one;b[i]=one;}
    volatile int sa=dSA[lane]; int s_a=sa;
    volatile int sb=0x7F7F7F7F; int s_b=sb;
    fp32x4_t c{0};
    c=__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        __builtin_bit_cast(i32x8_t,a),__builtin_bit_cast(i32x8_t,b),c,0,0,0,s_a,0,s_b);
    #pragma unroll
    for(int i=0;i<4;i++) dC[lane*4+i]=c[i];
#endif
}

#ifndef __HIP_DEVICE_COMPILE__
int main(){
    int* dS; fp32_t* dC;
    hipMalloc(&dS,64*sizeof(int)); hipMalloc(&dC,256*sizeof(fp32_t));
    unsigned unit=0x7F7F7F7F;
    auto run=[&](std::vector<int>&hS,std::vector<float>&o){
        hipMemcpy(dS,hS.data(),64*sizeof(int),hipMemcpyHostToDevice);
        hipLaunchKernelGGL(probe,dim3(1),64,0,0,dS,dC); hipDeviceSynchronize();
        o.assign(256,-1.f); hipMemcpy(o.data(),dC,256*sizeof(fp32_t),hipMemcpyDeviceToHost);
    };
    std::vector<int> hUnit(64,(int)unit); std::vector<float> base; run(hUnit,base);
    printf("baseline all C = %.0f (expect 128)\n\n",base[0]);
    printf("boost scale_A lane L (x2). Report reacting (m,n) and delta:\n");
    for(int L : {0,1,7,16,17,32,48,63}){
        std::vector<int> hS(64,(int)unit); hS[L]=(int)(128u|(127u<<8)|(127u<<16)|(127u<<24));
        std::vector<float> r; run(hS,r);
        // find reacting outputs
        int rm=-1,rn=-1,cnt=0; double dmax=0;
        for(int lane=0;lane<64;lane++)for(int i=0;i<4;i++){
            double d=r[lane*4+i]-base[lane*4+i];
            if(d>0.5){cnt++; dmax=d; int m=lane%16,n=(lane/16)*4+i; if(rm<0){rm=m;rn=n;}}
        }
        printf("  L=%2d (L%%16=%2d, L/16=%d): delta=%.0f on %d outputs, first at (m=%d,n=%d)\n",
               L,L%16,L/16,dmax,cnt,rm,rn);
    }
    printf("\nInterpretation: delta=32 => lane = one 32-K block (BLOCK). delta=8 => REP4.\n");
    hipFree(dS);hipFree(dC);
    return 0;
}
#endif
