# m9-scale: scales passed in aiter e8m0_shuffle (gfx950 shuffle_scale) layout.
import sys,shutil
src,name=sys.argv[1],sys.argv[2]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log','README.md','ldscal'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
def rep(a,b,cnt=1):
    global s; n=s.count(a); assert n==cnt,(n,a[:70]); s=s.replace(a,b)
# 1. global bases: row-block (32 rows) granularity; stride_sf* = bytes per 32-row block
rep("auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row * kargs.stride_sfa, static_cast<unsigned int>((kargs.m - row) * kargs.stride_sfa));",
    "auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + (row / 32) * kargs.stride_sfa, static_cast<unsigned int>(ceil_div_scale(kargs.m - row, 32) * kargs.stride_sfa));   // sc: e8m0_shuffle layout")
rep("auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + col * kargs.stride_sfb, static_cast<unsigned int>((kargs.n - col) * kargs.stride_sfb));",
    "auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / 32) * kargs.stride_sfb, static_cast<unsigned int>(ceil_div_scale(kargs.n - col, 32) * kargs.stride_sfb));")
rep("auto ng_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + n_batch * kargs.stride_sfa_batch + n_row * kargs.stride_sfa, has_next ? static_cast<unsigned int>((kargs.m - n_row) * kargs.stride_sfa) : 0u);",
    "auto ng_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + n_batch * kargs.stride_sfa_batch + (n_row / 32) * kargs.stride_sfa, has_next ? static_cast<unsigned int>(ceil_div_scale(kargs.m - n_row, 32) * kargs.stride_sfa) : 0u);")
rep("auto ng_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + n_batch * kargs.stride_sfb_batch + n_col * kargs.stride_sfb, has_next ? static_cast<unsigned int>((kargs.n - n_col) * kargs.stride_sfb) : 0u);",
    "auto ng_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + n_batch * kargs.stride_sfb_batch + (n_col / 32) * kargs.stride_sfb, has_next ? static_cast<unsigned int>(ceil_div_scale(kargs.n - n_col, 32) * kargs.stride_sfb) : 0u);")
# 2. per-lane global offset: a panel (2 K tiles) = 8 row blocks x 512 contiguous bytes; wave w loads blocks 2w, 2w+1
rep("    auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * T::NUM_KGROUPS; };\n    auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin * T::NUM_KGROUPS; };",
"""    // sc: e8m0_shuffle layout = [32-row block][K tile (8 groups)][k_lane 4][row 16][kk 2][row16-pair 2], 256 B per
    // (block, K tile).  A panel (2 K tiles x 256 rows) is 8 runs of 512 contiguous bytes: wave w copies blocks 2w, 2w+1.
    static_assert(T::SCALE_PANEL_TILES == 2 && T::NUM_KGROUPS == 8 && T::B_M == 256 && T::B_N == 256);
    const int sfa_g_lane = (2 * wave_id + lane_id / 32) * kargs.stride_sfa + (lane_id % 32) * T::VEC_GLOBAL_SCALE;
    const int sfb_g_lane = (2 * wave_id + lane_id / 32) * kargs.stride_sfb + (lane_id % 32) * T::VEC_GLOBAL_SCALE;
    auto gsfa_offset = [&](int panel_k_begin) { return sfa_g_lane + panel_k_begin * 256; };
    auto gsfb_offset = [&](int panel_k_begin) { return sfb_g_lane + panel_k_begin * 256; };""")
rep("auto fetch_sfa = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfa, u_gsfa, gsfa_offset(panel_k_begin)); };",
    "auto fetch_sfa = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfa, gsfa_offset(panel_k_begin)); };")
rep("auto fetch_sfb = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfb, u_gsfb, gsfb_offset(panel_k_begin)); };",
    "auto fetch_sfb = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfb, gsfb_offset(panel_k_begin)); };")
rep("v_nsfa0 = load<T::VEC_GLOBAL_SCALE>(ng_sfa, u_gsfa, gsfa_offset(0));","v_nsfa0 = load<T::VEC_GLOBAL_SCALE>(ng_sfa, gsfa_offset(0));")
rep("v_nsfb0 = load<T::VEC_GLOBAL_SCALE>(ng_sfb, u_gsfb, gsfb_offset(0));","v_nsfb0 = load<T::VEC_GLOBAL_SCALE>(ng_sfb, gsfb_offset(0));")
# 3. put_sf: verbatim copy (no transpose); LDS panel image = [block 8][K tile 2][256 B]
i=s.index("    auto put_sf = [&](auto& s_sf_, int wave_base, const SF_PANEL& raw, int buf) {"); j=s.index("    };",i)+len("    };")
s=s[:i]+"""    auto put_sf = [&](auto& s_sf_, int wave_base, const SF_PANEL& raw, int buf) {   // sc: no transpose, contiguous 1 KB per wave
        store<T::VEC_GLOBAL_SCALE>(s_sf_, raw, buf * T::SFA_PANEL_BYTES + wave_base);
    };"""+s[j:]
rep("    const int ssfa_wbase = (wave_id_n * T::T_M + wave_id_m) * T::WARP_SIZE * T::VEC_LDS_SCALE;\n    const int ssfb_wbase = (wave_id_m * T::T_N + wave_id_n) * T::WARP_SIZE * T::VEC_LDS_SCALE;",
    "    const int ssfa_wbase = wave_id * (T::WARP_SIZE * T::VEC_GLOBAL_SCALE) + lane_id * T::VEC_GLOBAL_SCALE;   // sc\n    const int ssfb_wbase = ssfa_wbase;")
# 4. LDS reads: per half h, e = 0..3: dword at (h*4+e)*512 + tile*256 + lane*4; v_perm picks row-pair byte (kk*2 + w)
i=s.index("    const int sf_rd_lane = lane_id ^"); j=s.index("    auto load_sfb = [&](auto& rd_sfb_, int, int) {"); j=s.index("    };",j)+len("    };")
s=s[:i]+"""    // sc: each dword D_e (row block h*4+e) holds bytes [kk0 w0, kk0 w1, kk1 w0, kk1 w1] of lane (g, r); the wave
    // needs w = its wave_m (A) / wave_n (B).  Two v_perm stages build [D0, D1, D2, D3] at byte w (kk0) / 2 + w (kk1).
    const unsigned int sfa_sel = 0x06020400u + static_cast<unsigned int>(wave_id_m) * 0x01010101u;
    const unsigned int sfb_sel = 0x06020400u + static_cast<unsigned int>(wave_id_n) * 0x01010101u;
    const int sf_rd_base = lane_id * 4;
    SF_KK v_sfa_next[2], v_sfb_next[2];
    int sf_read_off = 0;
    auto sf_gather = [&](auto& rd_, unsigned int sel, auto h_) {
        constexpr int h = decltype(h_)::value;
        unsigned int dw[4];
        opus::static_for<4>([&](auto e) {
            dw[e.value] = __builtin_bit_cast(unsigned int, load<4>(rd_, sf_rd_base + (h * 4 + e.value) * 512 + sf_read_off));
        });
        const unsigned int t01 = __builtin_amdgcn_perm(dw[1], dw[0], sel);
        const unsigned int t23 = __builtin_amdgcn_perm(dw[3], dw[2], sel);
        return SF_KK{__builtin_amdgcn_perm(t23, t01, 0x05040100u), __builtin_amdgcn_perm(t23, t01, 0x07060302u)};
    };
    auto load_sfa = [&](auto& rd_sfa_, int buf, int tile_in_panel) {
        sf_read_off = buf * T::SFA_PANEL_BYTES + tile_in_panel * 256;
        opus::static_for<2>([&](auto h) { v_sfa_next[h.value] = sf_gather(rd_sfa_, sfa_sel, h); });
    };
    auto load_sfb = [&](auto& rd_sfb_, int, int) {
        opus::static_for<2>([&](auto h) { v_sfb_next[h.value] = sf_gather(rd_sfb_, sfb_sel, h); });
    };"""+s[j:]
open(f,'w').write(s)
# host: shuffle the scales (aiter shuffle_scale, gfx950) for the kernel; reference keeps the raw ones
h=f'{d}/gemm_a4w4_mxfp4_4wave_host.cc'; s=open(h).read()
rep("""    CHECK_HIP(hipMalloc(&d_sfa_raw, sfa_cnt));""","""    CHECK_HIP(hipMalloc(&d_sfa_raw, sfa_cnt));
    // sc: kernel scales in aiter e8m0_shuffle (shuffle_scale, gfx950) layout: rows padded to 32, K/32 to 8 (pad 0x7F),
    // [rows/32][kg/8][k_lane 4][row16 16][kk 2][row-pair 2]; stride_sf* = bytes per 32-row block.
    const int kg8 = (kg + 7) / 8 * 8;
    auto shuffle_sf = [&](const std::vector<e8m0_t>& raw, int rows) {
        const int rows32 = (rows + 31) / 32 * 32;
        std::vector<e8m0_t> out(static_cast<size_t>(batch) * rows32 * kg8, static_cast<e8m0_t>(0x7F));
        for (int b = 0; b < batch; ++b) for (int r = 0; r < rows; ++r) for (int g = 0; g < kg; ++g) {
            const size_t o = static_cast<size_t>(b) * rows32 * kg8 + (static_cast<size_t>(r / 32) * (kg8 / 8) + g / 8) * 256
                           + (g % 4) * 64 + (r % 16) * 4 + ((g % 8) / 4) * 2 + (r % 32) / 16;
            out[o] = raw[(static_cast<size_t>(b) * rows + r) * kg + g];
        }
        return out;
    };
    const std::vector<e8m0_t> h_sfa_s = shuffle_sf(h_sfa, M), h_sfb_s = shuffle_sf(h_sfb, N);
    e8m0_t *d_sfa_s, *d_sfb_s;
    CHECK_HIP(hipMalloc(&d_sfa_s, h_sfa_s.size())); CHECK_HIP(hipMalloc(&d_sfb_s, h_sfb_s.size()));
    CHECK_HIP(hipMemcpy(d_sfa_s, h_sfa_s.data(), h_sfa_s.size(), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_sfb_s, h_sfb_s.data(), h_sfb_s.size(), hipMemcpyHostToDevice));""")
rep("""    kargs.ptr_sfa = d_sfa_raw; kargs.ptr_sfb = d_sfb_raw;   // native [rows, K/32]
    kargs.stride_sfa = kg; kargs.stride_sfb = kg;
    kargs.stride_sfa_batch = M * kg; kargs.stride_sfb_batch = N * kg;""","""    kargs.ptr_sfa = d_sfa_s; kargs.ptr_sfb = d_sfb_s;   // sc: e8m0_shuffle layout
    kargs.stride_sfa = 32 * kg8; kargs.stride_sfb = 32 * kg8;
    kargs.stride_sfa_batch = (M + 31) / 32 * 32 * kg8; kargs.stride_sfb_batch = (N + 31) / 32 * 32 * kg8;""")
open(h,'w').write(s); print(name,'ok')
