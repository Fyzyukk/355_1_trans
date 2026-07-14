// Nail the C register layout of the UNSCALED gfx950 instruction
// V_MFMA_F32_16X16X32_FP8_FP8 (wave64: elem_a=8, elem_b=8, elem_c=4 per lane).
// a/b registers are 8 packed fp8 = 64-bit long.
//
// Trick: make C[m][n] = m*16 + n (UNIQUE 0..255) so each C element self-identifies.
//   A[m][0]=m, A[m][1]=1, else 0 ;  B[0][n]=16, B[1][n]=n, else 0.
//   C[m][n] = A[m][0]*B[0][n] + A[m][1]*B[1][n] = m*16 + n.  (fp32-exact)
// Only k=0,1 are non-zero, so the A/B slot<->k permutation is irrelevant as long
// as A and B agree (they do). Read raw[lane*4+i]; value = m*16+n => reveals the
// true C map (lane, slot i) -> (m, n). Then match against candidate C-layouts.

#include "opus/opus.hpp"
#ifndef __HIP_DEVICE_COMPILE__
#include "opus/hip_minimal.hpp"
#include <cstdio>
#include <vector>
#endif
using namespace opus;

// A/B fed by a contiguous candidate (kk*8+i). Only k=0,1 matter.
__global__ void probe(const unsigned char* __restrict__ dA,
                      const unsigned char* __restrict__ dB,
                      fp32_t* __restrict__ dCraw){
#if defined(__gfx950__)
    int lane = (int)__builtin_amdgcn_workitem_id_x();
    int row = lane % 16, kk = lane / 16;
    fp8x8_t a_reg, b_reg;
    #pragma unroll
    for(int i=0;i<8;i++){
        int k = kk*8 + i;              // this lane's K slot
        a_reg[i] = __builtin_bit_cast(fp8_t, dA[row*32 + k]);   // A[row][k]
        b_reg[i] = __builtin_bit_cast(fp8_t, dB[k*16 + row]);   // B[k][row]
    }
    fp32x4_t c{0};
    c = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(
            __builtin_bit_cast(long,a_reg), __builtin_bit_cast(long,b_reg), c, 0,0,0);
    #pragma unroll
    for(int i=0;i<4;i++) dCraw[lane*4+i] = c[i];
#endif
}

#ifndef __HIP_DEVICE_COMPILE__
// exact e4m3 for the integers we need: 0..16
static unsigned char e4m3(int v){
    switch(v){
        case 0:return 0x00; case 1:return 0x38; case 2:return 0x40; case 3:return 0x44;
        case 4:return 0x48; case 5:return 0x4A; case 6:return 0x4C; case 7:return 0x4E;
        case 8:return 0x50; case 9:return 0x51; case 10:return 0x52; case 11:return 0x53;
        case 12:return 0x54; case 13:return 0x55; case 14:return 0x56; case 15:return 0x57;
        case 16:return 0x58; default:return 0x00;
    }
}
// candidate C-layouts: (lane, slot i) -> (m,n). row=lane%16, gk=lane/16 (0..3).
static void mapC(int CLAY,int lane,int i,int& m,int& n){
    int row=lane%16, gk=lane/16;
    if      (CLAY==0){ m=row;    n=gk*4+i; }
    else if (CLAY==1){ m=gk*4+i; n=row;    }
    else if (CLAY==2){ m=row;    n=i*4+gk; }
    else if (CLAY==3){ m=i*4+gk; n=row;    }
    else if (CLAY==4){ m=gk+i*4; n=row;    }
    else             { m=row;    n=gk+i*4; }
}
int main(){
    std::vector<unsigned char> hA(16*32,0), hB(32*16,0);
    for(int m=0;m<16;m++){ hA[m*32+0]=e4m3(m); hA[m*32+1]=e4m3(1); }
    for(int n=0;n<16;n++){ hB[0*16+n]=e4m3(16); hB[1*16+n]=e4m3(n); }

    unsigned char *dA,*dB; fp32_t* dC;
    hipMalloc(&dA,hA.size()); hipMalloc(&dB,hB.size()); hipMalloc(&dC,256*sizeof(fp32_t));
    hipMemcpy(dA,hA.data(),hA.size(),hipMemcpyHostToDevice);
    hipMemcpy(dB,hB.data(),hB.size(),hipMemcpyHostToDevice);
    hipLaunchKernelGGL(probe,dim3(1),64,0,0,dA,dB,dC); hipDeviceSynchronize();
    std::vector<float> raw(256,-1.f);
    hipMemcpy(raw.data(),dC,256*sizeof(fp32_t),hipMemcpyDeviceToHost);

    // sanity: every value should be a unique integer in [0,255]
    printf("raw C (value = m*16+n if layout gives unique cover):\n");
    for(int lane=0;lane<64;lane++){
        printf("L%02d(r%2d,gk%d):",lane,lane%16,lane/16);
        for(int i=0;i<4;i++){int v=(int)(raw[lane*4+i]+0.5f); printf(" %3d(m%2d,n%2d)",v,v/16,v%16);}
        printf("\n");
    }
    // match candidate C-layouts: predicted value m*16+n must equal raw
    printf("\nCandidate C-layout match:\n");
    for(int CLAY=0;CLAY<6;CLAY++){
        bool ok=true;
        for(int lane=0;lane<64&&ok;lane++)for(int i=0;i<4;i++){
            int m,n; mapC(CLAY,lane,i,m,n);
            if((int)(raw[lane*4+i]+0.5f)!=m*16+n){ok=false;break;}
        }
        printf("  C-layout %d : %s\n",CLAY,ok?"MATCH":"-");
    }
    hipFree(dA);hipFree(dB);hipFree(dC);
    return 0;
}
#endif
