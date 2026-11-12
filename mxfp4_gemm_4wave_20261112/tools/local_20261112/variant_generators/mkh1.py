import sys,shutil
src,name=sys.argv[1],sys.argv[2]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
def rep(a,b):
    global s; assert s.count(a)==1,(s.count(a),a[:60]); s=s.replace(a,b)
# helper sched: prologue C00 with 2 DS + 2 stores per 4 MFMAs
rep('// One fp4 MFMA operand per lane','''// h1: prologue C00 -- MFMA, DS, MFMA, store, MFMA, DS, MFMA, store.
__device__ inline void sched_c00_store() {
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
}
// One fp4 MFMA operand per lane''')
# state outside the persistent loop
rep("""    int prefetched = 0;""","""    int prefetched = 0;
    // h1: bottom half of the previous tile's C (bf16), read from LDS in its write-back and
    // stored during this tile's prologue C00.
    constexpr int H1_VEC = 16 / sizeof(D_C);
    constexpr int H1_COPIES = T::HALF_B_M * T::B_N / (T::BLOCK_SIZE * H1_VEC);   // 16
    opus::vector_t<D_C, H1_VEC> v_pend_c[H1_COPIES];
    D_C* pend_c_ptr = reinterpret_cast<D_C*>(kargs.ptr_c);
    int pend_c = 0;
    auto store_pend_c = [&](int tid) {
        auto g_cp = make_gmem(pend_c_ptr, pend_c ? 0xffffffffu : 0u);
        opus::static_for<H1_COPIES>([&](auto i) {
            const int linear = tid * H1_VEC + decltype(i)::value * T::BLOCK_SIZE * H1_VEC;
            const int r = linear / T::B_N + T::HALF_B_M, c = linear % T::B_N;
            store<H1_VEC>(g_cp, v_pend_c[i.value], r * kargs.stride_c + c, 0, opus::number<2>{});
        });
    };""")
# prologue C00: issue the stores
rep("""        // C00 of tile 0 + A1, B1 of tile 0.
        load_a(rd_a, number<1>{});
        load_b(rd_b, number<1>{});""","""        // C00 of tile 0 + A1, B1 of tile 0 (+ h1: the previous tile's bottom-half stores).
        load_a(rd_a, number<1>{});
        load_b(rd_b, number<1>{});
        store_pend_c(wave_id * T::WARP_SIZE + lane_id);""")
open(f,'w').write(s)
# the sched_x4<8, 2, 0, 0>() right after: first occurrence after the prologue C00 comment
i=s.index('store_pend_c(wave_id * T::WARP_SIZE + lane_id);'); j=s.index('sched_x4<8, 2, 0, 0>();',i)
s=s[:j]+'sched_c00_store();   // h1'+s[j+len('sched_x4<8, 2, 0, 0>();'):]
# write-back half 1: only read into v_pend_c
rep("""    copy_output_half(1);""","""    opus::static_for<c_copies>([&](auto i) {   // h1: bottom half -> registers, stored in the next prologue
        const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
        v_pend_c[i.value] = load<VEC_OUT>(s_c, (linear / T::B_N) * c_lds_stride + linear % T::B_N);
    });
    pend_c_ptr = reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col;
    pend_c = 1;""")
# after the persistent loop: flush
rep("""    }   // persistent tile loop
}""","""    }   // persistent tile loop
    if (pend_c) {   // h1: the last tile's bottom half
        unsigned int lm = ~0u; asm volatile("" : "+s"(lm));
        const int ln = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lm, __builtin_amdgcn_mbcnt_lo(lm, 0u)));
        s_waitcnt_lgkmcnt(0_I);
        store_pend_c(wave_id * T::WARP_SIZE + ln);
    }
}""")
open(f,'w').write(s); print(name,'ok')
