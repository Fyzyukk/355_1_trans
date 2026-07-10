// End-to-end probe of the ACTUAL tiled swap_ab mma C-fragment layout.
// Feeds the exact make_layout_ra/rb + tiled mma the kernel uses, with A/B chosen
// so that C[m][n] = m*STRIDE_C + n is an fp32-exact unique fingerprint, then reads
// which (lane, fragment i) each C value lands in. This is the ground truth for the
// C-store layout (no hand-derived table).
//
// Construction (single 128x128x32 sub-slice, wave0):
//   For a single 16x16x32 tile at (wave_id_m=wave_id_n=0), we want to know for each
//   lane and each accumulator register i, what (m,n) it holds. We build A[m][k],
//   B[n][k] (B row-major NxK, since the kernel treats B as [N,K]) such that
//   C[m][n] = sum_k A[m][k]*B[n][k] = m*16 + n uniquely.
//     A[m][0]=m, A[m][1]=1, else 0
//     B[n][0]=16, B[n][1]=n, else 0
//   => C[m][n] = A[m][0]*B[n][0] + A[m][1]*B[n][1] = m*16 + n.
// We only need one 16x16x32 (E_M=E_N=E_K=1 view), so run a minimal tiled mma with
// the kernel's adaptor but expand=1. Read raw c registers; value reveals (m,n).

#include "opus/opus.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace opus;

#define CHECK_HIP(call) do { hipError_t s_ = call; if (s_ != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(s_)); exit(1);} } while(0)

// exact e4m3 for integers 0..16
__host__ static unsigned char e4m3(int v){
    switch(v){
        case 0:return 0x00; case 1:return 0x38; case 2:return 0x40; case 3:return 0x44;
        case 4:return 0x48; case 5:return 0x4A; case 6:return 0x4C; case 7:return 0x4E;
        case 8:return 0x50; case 9:return 0x51; case 10:return 0x52; case 11:return 0x53;
        case 12:return 0x54; case 13:return 0x55; case 14:return 0x56; case 15:return 0x57;
        case 16:return 0x58; default:return 0x00;
    }
}

__global__ void probe(const unsigned char* __restrict__ dA,
                      const unsigned char* __restrict__ dB,
                      float* __restrict__ dCraw){
#if defined(__gfx950__)
    int lane = (int)__builtin_amdgcn_workitem_id_x();
    int row = lane % 16, kk = lane / 16;   // matches probe_16x16x32 feeding

    // Build fp8x8 operands exactly like the raw probe, but through the swap_ab adaptor.
    fp8x8_t a_reg, b_reg;
    #pragma unroll
    for(int i=0;i<8;i++){
        int k = kk*8 + i;
        a_reg[i] = __builtin_bit_cast(fp8_t, dA[row*32 + k]);   // A[row][k]
        b_reg[i] = __builtin_bit_cast(fp8_t, dB[row*32 + k]);   // B[row][k]  (B is [N,K])
    }

    // Single-tile mma via the adaptor the kernel uses (swap_ab), expand=1.
    auto mma = make_tiled_mma<fp8_t, fp8_t, fp32_t>(
        seq<1,1,1>{}, seq<1,1,1>{}, seq<16,16,32>{}, mfma_adaptor_swap_ab{});
    typename decltype(mma)::vtype_a va;
    typename decltype(mma)::vtype_b vb;
    #pragma unroll
    for (int i=0;i<8;i++){ va[i]=a_reg[i]; vb[i]=b_reg[i]; }
    auto vc = mma(va, vb);
    #pragma unroll
    for (int i=0;i<4;i++) dCraw[lane*4+i] = vc[i];
#endif
}

int main(){
    std::vector<unsigned char> hA(16*32,0), hB(16*32,0);
    for(int m=0;m<16;m++){ hA[m*32+0]=e4m3(m); hA[m*32+1]=e4m3(1); }
    for(int n=0;n<16;n++){ hB[n*32+0]=e4m3(16); hB[n*32+1]=e4m3(n); }

    unsigned char *dA,*dB; float* dC;
    CHECK_HIP(hipMalloc(&dA,hA.size())); CHECK_HIP(hipMalloc(&dB,hB.size()));
    CHECK_HIP(hipMalloc(&dC,256*sizeof(float)));
    CHECK_HIP(hipMemcpy(dA,hA.data(),hA.size(),hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB,hB.data(),hB.size(),hipMemcpyHostToDevice));
    probe<<<1,64>>>(dA,dB,dC);
    CHECK_HIP(hipGetLastError()); CHECK_HIP(hipDeviceSynchronize());
    std::vector<float> raw(256,-1.f);
    CHECK_HIP(hipMemcpy(raw.data(),dC,256*sizeof(float),hipMemcpyDeviceToHost));

    printf("raw C via swap_ab tiled mma (value = m*16+n):\n");
    for(int lane=0;lane<64;lane++){
        printf("L%02d(r%2d,gk%d):",lane,lane%16,lane/16);
        for(int i=0;i<4;i++){int v=(int)(raw[lane*4+i]+0.5f); printf(" %3d(m%2d,n%2d)",v,v/16,v%16);}
        printf("\n");
    }
    // candidate layouts (lane,i)->(m,n)
    auto mapC=[](int CLAY,int lane,int i,int&m,int&n){
        int row=lane%16, gk=lane/16;
        if      (CLAY==0){ m=row;    n=gk*4+i; }
        else if (CLAY==1){ m=gk*4+i; n=row;    }
        else if (CLAY==2){ m=row;    n=i*4+gk; }
        else if (CLAY==3){ m=i*4+gk; n=row;    }
    };
    printf("\nCandidate C-layout match (i=fragment):\n");
    for(int CLAY=0;CLAY<4;CLAY++){
        bool ok=true;
        for(int lane=0;lane<64&&ok;lane++)for(int i=0;i<4;i++){
            int m,n; mapC(CLAY,lane,i,m,n);
            if((int)(raw[lane*4+i]+0.5f)!=m*16+n){ok=false;break;}
        }
        printf("  C-layout %d %s: %s\n",CLAY,
            CLAY==0?"(m=row,n=gk*4+i)":CLAY==1?"(m=gk*4+i,n=row)":CLAY==2?"(m=row,n=i*4+gk)":"(m=i*4+gk,n=row)",
            ok?"MATCH":"-");
    }
    return 0;
}
