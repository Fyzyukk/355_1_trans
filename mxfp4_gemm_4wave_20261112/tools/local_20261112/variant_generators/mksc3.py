import sys,shutil
src,name=sys.argv[1],sys.argv[2]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
def rep(a,b):
    global s; assert s.count(a)==1,(s.count(a),a[:70]); s=s.replace(a,b)
rep("        opus::static_for<2>([&](auto h) { v_sfa_raw[h.value] = sf_load_raw(rd_sfa_, h); });",
    "        v_sfa_raw[0] = sf_load_raw(rd_sfa_, number<0>{});   // sc3: half 1 is read in read_sf(0)")
rep("        opus::static_for<2>([&](auto h) { v_sfb_raw[h.value] = sf_load_raw(rd_sfb_, h); });",
    "        v_sfb_raw[0] = sf_load_raw(rd_sfb_, number<0>{});")
rep("    auto read_sf = [&](auto&, auto&, auto h_) {","    auto read_sf = [&](auto& rd_sfa_, auto& rd_sfb_, auto h_) {")
rep("        opus::static_for<T::E_K>([&](auto kk) { v_sfa[h][kk.value] = a[kk.value]; v_sfb[h][kk.value] = b[kk.value]; });",
    """        opus::static_for<T::E_K>([&](auto kk) { v_sfa[h][kk.value] = a[kk.value]; v_sfb[h][kk.value] = b[kk.value]; });
        if constexpr (h == 0) {   // sc3: raw half 1 only now -> 8 fewer live dwords between C01 and C11
            v_sfa_raw[1] = sf_load_raw(rd_sfa_, number<1>{});
            v_sfb_raw[1] = sf_load_raw(rd_sfb_, number<1>{});
        }""")
open(f,'w').write(s); print(name,'ok')
