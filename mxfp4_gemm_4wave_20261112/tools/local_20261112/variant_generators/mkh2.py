import sys,shutil
src,name=sys.argv[1],sys.argv[2]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
def rep(a,b):
    global s; assert s.count(a)==1,(s.count(a),a[:60]); s=s.replace(a,b)
# both halves pending
rep("opus::vector_t<D_C, H1_VEC> v_pend_c[H1_COPIES];","opus::vector_t<D_C, H1_VEC> v_pend_c[H1_COPIES], v_pend_c0[H1_COPIES];   // h2: + top half")
rep("""            store<H1_VEC>(g_cp, v_pend_c[i.value], r * kargs.stride_c + c, 0, opus::number<2>{});
        });""","""            store<H1_VEC>(g_cp, v_pend_c0[i.value], (r - T::HALF_B_M) * kargs.stride_c + c, 0, opus::number<2>{});   // h2: top half
            store<H1_VEC>(g_cp, v_pend_c[i.value], r * kargs.stride_c + c, 0, opus::number<2>{});
        });""")
# sched: 1 store per MFMA, 1 DS per 2 MFMAs
rep("""__device__ inline void sched_c00_store() {
    opus::static_for<8>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x040, 1, 0);   // VMEM write
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x040, 1, 0);
    });
}""","""__device__ inline void sched_c00_store() {   // h2: 32 stores, 1 per MFMA
    opus::static_for<16>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x040, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(0x040, 1, 0);
    });
}""")
# write-back half 0: read into v_pend_c0 instead of storing
rep("""    copy_output_half(0);""","""    opus::static_for<c_copies>([&](auto i) {   // h2: top half -> registers
        const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
        v_pend_c0[i.value] = load<VEC_OUT>(s_c, (linear / T::B_N) * c_lds_stride + linear % T::B_N);
    });""")
open(f,'w').write(s); print(name,'ok')
