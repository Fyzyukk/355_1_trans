// Determine the A/B register K-layout of the SINGLE gfx950 instruction
// V_MFMA_SCALE_F32_16X16X128_F8F6F4 (wave64, one lane holds 32 fp8 = i32x8).
//
// Method: fill a_reg/b_reg from row-major global A[16][128] / B[128][16] using a
// CANDIDATE (row,k) mapping, run one unscaled MMA, dump raw c_reg. On the host,
// for every (A/B-layout, C-layout) candidate combo, predict raw[lane*4+i] from a
// host reference C = A*B and report the combo that reproduces HW output exactly.
// The matching combo IS the true hardware layout.
//
// A,B entries are 0/1 (deterministic hash) so C is small and fp32-exact.

#include "opus/opus.hpp"
#ifndef __HIP_DEVICE_COMPILE__
#include "opus/hip_minimal.hpp"
#include <cstdio>
#include <vector>
#include <cstring>
#endif
using namespace opus;

__host__ __device__ inline int Aval(int m,int k){ return ((m*131 + k*17) % 3 == 0) ? 1 : 0; }
__host__ __device__ inline int Bval(int k,int n){ return ((k*13  + n*7 ) % 3 == 0) ? 1 : 0; }

// candidate (row,k) for A given lane,slot i (0..31). kk = lane/16 (0..3), row=lane%16.
__device__ inline void mapA(int ALAY,int lane,int i,int& m,int& k){
    int row = lane % 16, kk = lane / 16;
    m = row;
    if      (ALAY==0){ int g=i/8, j=i%8; k = g*32 + kk*8 + j; } // user hypo: 8-contig, stride-32, x4
    else if (ALAY==1){ k = kk*32 + i; }                         // contiguous quarter per lane
    else             { k = i*4 + kk; }                          // interleaved
}
__device__ inline void mapB(int BLAY,int lane,int i,int& k,int& n){
    int col = lane % 16, kk = lane / 16;
    n = col;
    if      (BLAY==0){ int g=i/8, j=i%8; k = g*32 + kk*8 + j; }
    else if (BLAY==1){ k = kk*32 + i; }
    else             { k = i*4 + kk; }
}

template<int ALAY,int BLAY>
__global__ void probe(const unsigned char* __restrict__ dA,
                      const unsigned char* __restrict__ dB,
                      fp32_t* __restrict__ dCraw){
#if defined(__gfx950__)
    int lane = (int)__builtin_amdgcn_workitem_id_x();
    fp8x32_t a_reg, b_reg;
    #pragma unroll
    for(int i=0;i<32;i++){
        int m,k; mapA(ALAY,lane,i,m,k);
        a_reg[i] = __builtin_bit_cast(fp8_t, dA[m*128 + k]);
        int kk,n; mapB(BLAY,lane,i,kk,n);
        b_reg[i] = __builtin_bit_cast(fp8_t, dB[kk*16 + n]);
    }
    volatile int su = 0x7F7F7F7F; int s = su;   // unit E8M0 scale, defeat const-fold
    fp32x4_t c{0};
    c = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            __builtin_bit_cast(i32x8_t,a_reg), __builtin_bit_cast(i32x8_t,b_reg),
            c, 0,0, 0, s, 0, s);
    #pragma unroll
    for(int i=0;i<4;i++) dCraw[lane*4+i] = c[i];
#endif
}
template __global__ void probe<0,0>(const unsigned char*,const unsigned char*,fp32_t*);
template __global__ void probe<1,1>(const unsigned char*,const unsigned char*,fp32_t*);
template __global__ void probe<2,2>(const unsigned char*,const unsigned char*,fp32_t*);

#ifndef __HIP_DEVICE_COMPILE__
static unsigned char e4m3(int v){ // exact small ints
    switch(v){case 0:return 0x00;case 1:return 0x38;case 2:return 0x40;case 3:return 0x44;default:return 0x48;}
}
// C-layout candidates: given lane,i -> (m,n)
static void mapC(int CLAY,int lane,int i,int& m,int& n){
    int row=lane%16, gk=lane/16;
    if      (CLAY==0){ m=row;        n=gk*4+i; }
    else if (CLAY==1){ m=gk*4+i;     n=row;    }
    else if (CLAY==2){ m=row;        n=i*4+gk; }
    else             { m=i*4+gk;     n=row;    }
}
int main(){
    std::vector<unsigned char> hA(16*128), hB(128*16);
    for(int m=0;m<16;m++)for(int k=0;k<128;k++) hA[m*128+k]=e4m3(Aval(m,k));
    for(int k=0;k<128;k++)for(int n=0;n<16;n++)  hB[k*16+n]=e4m3(Bval(k,n));
    // host reference C (exact int)
    std::vector<int> ref(16*16,0);
    for(int m=0;m<16;m++)for(int n=0;n<16;n++){int s=0;for(int k=0;k<128;k++)s+=Aval(m,k)*Bval(k,n);ref[m*16+n]=s;}

    unsigned char *dA,*dB; fp32_t* dC;
    hipMalloc(&dA,hA.size()); hipMalloc(&dB,hB.size()); hipMalloc(&dC,256*sizeof(fp32_t));
    hipMemcpy(dA,hA.data(),hA.size(),hipMemcpyHostToDevice);
    hipMemcpy(dB,hB.data(),hB.size(),hipMemcpyHostToDevice);

    auto run=[&](int LAY,std::vector<float>& raw){
        switch(LAY){
            case 0: hipLaunchKernelGGL((probe<0,0>),dim3(1),64,0,0,dA,dB,dC);break;
            case 1: hipLaunchKernelGGL((probe<1,1>),dim3(1),64,0,0,dA,dB,dC);break;
            default:hipLaunchKernelGGL((probe<2,2>),dim3(1),64,0,0,dA,dB,dC);break;
        }
        hipDeviceSynchronize();
        raw.assign(256,-1.f);
        hipMemcpy(raw.data(),dC,256*sizeof(fp32_t),hipMemcpyDeviceToHost);
    };

    const char* names[3]={"A0=user(8-contig,stride32,x4)","A1=contiguous-quarter","A2=interleaved-stride4"};
    for(int LAY=0;LAY<3;LAY++){
        std::vector<float> raw; run(LAY,raw);
        for(int CLAY=0;CLAY<4;CLAY++){
            bool ok=true; int checked=0;
            for(int lane=0;lane<64&&ok;lane++)for(int i=0;i<4;i++){
                int m,n; mapC(CLAY,lane,i,m,n);
                if((int)(raw[lane*4+i]+0.5f)!=ref[m*16+n]){ok=false;break;}
                checked++;
            }
            if(ok) printf("MATCH: A/B-layout %s  +  C-layout %d  (checked %d)\n",names[LAY],CLAY,checked);
        }
    }
    printf("(no MATCH lines above => none of the tested candidates is the true layout)\n");
    // also dump raw for LAY0 for inspection
    std::vector<float> raw; run(0,raw);
    printf("raw c_reg dump (layout A0), lane:reg = value\n");
    for(int lane=0;lane<64;lane++){printf("L%02d:",lane);for(int i=0;i<4;i++)printf(" %3d",(int)(raw[lane*4+i]+0.5f));printf(lane%4==3?"\n":"   ");}
    hipFree(dA);hipFree(dB);hipFree(dC);
    return 0;
}
#endif
