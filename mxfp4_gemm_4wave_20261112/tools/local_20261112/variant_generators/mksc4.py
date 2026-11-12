import sys,shutil
src,name=sys.argv[1],sys.argv[2]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
def rep(a,b):
    global s; assert s.count(a)==1,(s.count(a),a[:70]); s=s.replace(a,b)
rep("""        v_nsfa0 = load<T::VEC_GLOBAL_SCALE>(ng_sfa, gsfa_offset(0));
        v_nsfb0 = load<T::VEC_GLOBAL_SCALE>(ng_sfb, gsfb_offset(0));""",
"""        {   // sc4: lane offsets re-derived here (opaque lane id) so the per-tile ones need not stay live / spill
            unsigned int lm = ~0u; asm volatile("" : "+s"(lm));
            const int ln = static_cast<int>(__builtin_amdgcn_mbcnt_hi(lm, __builtin_amdgcn_mbcnt_lo(lm, 0u)));
            const int blk = 2 * wave_id + ln / 32, in_blk = (ln % 32) * T::VEC_GLOBAL_SCALE;
            v_nsfa0 = load<T::VEC_GLOBAL_SCALE>(ng_sfa, blk * kargs.stride_sfa + in_blk);
            v_nsfb0 = load<T::VEC_GLOBAL_SCALE>(ng_sfb, blk * kargs.stride_sfb + in_blk);
        }""")
open(f,'w').write(s); print(name,'ok')
