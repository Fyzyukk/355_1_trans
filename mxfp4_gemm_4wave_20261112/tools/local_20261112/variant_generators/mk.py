import sys,re,shutil,os
src='/root/workspace/355_1_trans/mxfp4_gemm_4wave_20261112/candidate_m5'
C11_E1='''__device__ inline void sched_c11() {
    opus::static_for<7>([&](auto) { sched_m<1, 1>(); sched_m<0, 1>(); sched_m<0, 1>(); sched_m<0, 0>(); });
    sched_m<1, 1>();
    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);   // e1: scale ds_write after the last ds_read
    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);   // e1: scale panel buffer_load
    sched_m<0, 1>(); sched_m<0, 1>(); sched_m<0, 0>();
}'''
C00={'e2a':'''__device__ inline void sched_c00() {
    opus::static_for<8>([&](auto) { sched_m<1, 1>(); sched_m<1, 1>(); sched_m<0, 1>(); });   // e2a: 16 DS over 24 MFMAs
    opus::static_for<8>([&](auto) { sched_m<0, 0>(); });
}''','e2b':'''__device__ inline void sched_c00() {
    opus::static_for<16>([&](auto) { sched_m<1, 1>(); sched_m<0, 1>(); });   // e2b: 16 DS over 32 MFMAs
}'''}
name=sys.argv[1]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
if 'e1' in name:
    s,n=re.subn(r'__device__ inline void sched_c11\(\) \{.*?\n\}',C11_E1,s,flags=re.S); assert n==1
for k,v in C00.items():
    if k in name:
        s,n=re.subn(r'__device__ inline void sched_c00\(\) \{.*?\n\}',v,s,flags=re.S); assert n==1
open(f,'w').write(s); print(name,'ok')
