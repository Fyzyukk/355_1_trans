import sys,shutil
src,name=sys.argv[1],sys.argv[2]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
def rep(a,b):
    global s; assert s.count(a)==1,(s.count(a),a[:70]); s=s.replace(a,b)
i=s.index("    SF_KK v_sfa_next[2], v_sfb_next[2];\n    int sf_read_off = 0;\n    auto sf_gather"); j=s.index("    auto load_sfb = [&](auto& rd_sfb_, int, int) {",i); j=s.index("    };",j)+len("    };")
s=s[:i]+"""    // sc2: load_sf* only issues the LDS reads (raw dwords, C01); the v_perm gather runs in read_sf (C11 / C00),
    // long after the data landed, so no lgkmcnt wait lands in the main loop.
    using SF_RAW = opus::vector_t<opus::u32_t, 4>;
    SF_RAW v_sfa_raw[2], v_sfb_raw[2];
    int sf_read_off = 0;
    auto sf_load_raw = [&](auto& rd_, auto h_) {
        constexpr int h = decltype(h_)::value;
        SF_RAW r;
        opus::static_for<4>([&](auto e) {
            r[e.value] = __builtin_bit_cast(unsigned int, load<4>(rd_, sf_rd_base + (h * 4 + e.value) * 512 + sf_read_off));
        });
        return r;
    };
    auto sf_gather = [&](const SF_RAW& dw, unsigned int sel) {
        const unsigned int t01 = __builtin_amdgcn_perm(dw[1], dw[0], sel);
        const unsigned int t23 = __builtin_amdgcn_perm(dw[3], dw[2], sel);
        return SF_KK{__builtin_amdgcn_perm(t23, t01, 0x05040100u), __builtin_amdgcn_perm(t23, t01, 0x07060302u)};
    };
    auto load_sfa = [&](auto& rd_sfa_, int buf, int tile_in_panel) {
        sf_read_off = buf * T::SFA_PANEL_BYTES + tile_in_panel * 256;
        opus::static_for<2>([&](auto h) { v_sfa_raw[h.value] = sf_load_raw(rd_sfa_, h); });
    };
    auto load_sfb = [&](auto& rd_sfb_, int, int) {
        opus::static_for<2>([&](auto h) { v_sfb_raw[h.value] = sf_load_raw(rd_sfb_, h); });
    };"""+s[j:]
rep("        opus::static_for<T::E_K>([&](auto kk) { v_sfa[h][kk.value] = v_sfa_next[h][kk.value]; v_sfb[h][kk.value] = v_sfb_next[h][kk.value]; });",
    "        const SF_KK a = sf_gather(v_sfa_raw[h], sfa_sel), b = sf_gather(v_sfb_raw[h], sfb_sel);   // sc2\n        opus::static_for<T::E_K>([&](auto kk) { v_sfa[h][kk.value] = a[kk.value]; v_sfb[h][kk.value] = b[kk.value]; });")
open(f,'w').write(s); print(name,'ok')
