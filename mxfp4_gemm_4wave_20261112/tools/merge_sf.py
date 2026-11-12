# Apply the scale-panel transpose-at-write scheme (sf5) to a kernel template; persistent-aware.
import sys
src,dst=sys.argv[1],sys.argv[2]
s=open(src).read()
def rep(o,n,c=None):
    global s; k=s.count(o); assert k>0 and (c is None or k==c),(k,o[:90]); s=s.replace(o,n)
def body(name):
    i=s.index(f'__device__ inline auto {name}('); j=s.index('\n}\n',i)+3; return i,j
TPL='''__device__ inline auto NAME(int lane_id, int wave_id_m, int wave_id_n, int ST) {
    // Writer wave (P0, P1), lane (e = lane / 16, r = lane % 16) loads row P0 * 128 + e * 32 +
    // P1 * 16 + r: after the 4x4 byte transpose across e, lane (g, r) holds byte e = repeat e,
    // K group g -- the MFMA scale layout of wave (h = P0, w = P1).
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p P0: half (128 rows)
        opus::number<4>{},                                             // p e (32 rows)
        opus::number<2>{},                                             // p P1 (16 rows)
        opus::number<16>{},                                            // p r
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // y 16 bytes (panel)
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    static_assert(2 * 4 * 2 * 16 == T::B_M && T::B_M == T::B_N);
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{ST, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{P0, lane_id / 16, P1, lane_id % 16}));
}
'''
for nm,st,p0,p1 in [('make_layout_gsfa','stride_sfa','wave_id_n','wave_id_m'),('make_layout_gsfb','stride_sfb','wave_id_m','wave_id_n')]:
    i,j=body(nm); s=s[:i]+TPL.replace('NAME',nm).replace('ST',st).replace('P0',p0).replace('P1',p1)+s[j:]
rep('''    auto prefetch_sfa = [&](int panel_k_begin, int buf) {
        async_load<T::VEC_GLOBAL_SCALE>(g_sfa, s_sfa.ptr, u_gsfa, u_ssfa + ssfa_offset(buf, 0), gsfa_offset(panel_k_begin));
    };
    auto prefetch_sfb = [&](int panel_k_begin, int buf) {
        async_load<T::VEC_GLOBAL_SCALE>(g_sfb, s_sfb.ptr, u_gsfb, u_ssfb + ssfb_offset(buf, 0), gsfb_offset(panel_k_begin));
    };''','''    // Scale panels go global -> VGPR -> 4x4 byte transpose -> LDS, so LDS holds the MFMA scale
    // layout and the main loop reads it as is (one transpose per panel instead of one per wave
    // and tile).  A panel is loaded in iteration t and written in t + 1 (its load has landed by
    // B_b of t + 1).
    using SF_PANEL = opus::vector_t<D_SF, T::VEC_GLOBAL_SCALE>;
    using SF_DW4 = opus::vector_t<opus::u32_t, 4>;
    SF_PANEL v_sf_pend;
    auto fetch_sfa = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfa, u_gsfa, gsfa_offset(panel_k_begin)); };
    auto fetch_sfb = [&](int panel_k_begin) { return load<T::VEC_GLOBAL_SCALE>(g_sfb, u_gsfb, gsfb_offset(panel_k_begin)); };
    auto put_sf = [&](auto& s_sf_, const auto& u_ssf, const SF_PANEL& raw, int buf) {
        const auto dw = __builtin_bit_cast(SF_DW4, raw);
        SF_DW4 t;
        opus::static_for<4>([&](auto d) { t[d.value] = transpose_scale_dword_4x4(dw[d.value], lane_id / 16); });
        store<T::VEC_GLOBAL_SCALE>(s_sf_, __builtin_bit_cast(SF_PANEL, t), u_ssf + (buf * T::SFA_PANEL_BYTES + lane_id * T::VEC_GLOBAL_SCALE));   // the LDS-DMA layout has no lane term
    };''')
i=s.index('    opus::vector_t<D_SF, 2 * T::VEC_LDS_SCALE> v_sfa_raw, v_sfb_raw;'); j=s.index('\n',s.index('auto load_sfb',i))+1
s=s[:i]+'''    // Scale reads: lane reads the 2 dwords (kk = 0, 1) of half h at LDS index (h * 2 + w) * 64 +
    // lane (+ 8 B for the second tile of the panel), early (C01) into v_sf*_next; read_sf moves
    // half h into v_sfa / v_sfb when the previous values are no longer used.
    using SF_KK = opus::vector_t<D_SF_PACK, T::E_K>;
    const int sfa_rd_base = (wave_id_m * T::WARP_SIZE + lane_id) * T::VEC_GLOBAL_SCALE;
    const int sfb_rd_base = (wave_id_n * T::WARP_SIZE + lane_id) * T::VEC_GLOBAL_SCALE;
    constexpr int sf_half_stride = 2 * T::WARP_SIZE * T::VEC_GLOBAL_SCALE;
    SF_KK v_sfa_next[2], v_sfb_next[2];
    int sf_read_off = 0;
    auto load_sfa = [&](auto& rd_sfa_, int buf, int tile_in_panel) {
        sf_read_off = buf * T::SFA_PANEL_BYTES + tile_in_panel * T::NUM_KGROUPS;
        opus::static_for<2>([&](auto h) { v_sfa_next[h.value] = __builtin_bit_cast(SF_KK, load<T::VEC_LDS_SCALE>(rd_sfa_, sfa_rd_base + h.value * sf_half_stride + sf_read_off)); });
    };
    auto load_sfb = [&](auto& rd_sfb_, int, int) {
        opus::static_for<2>([&](auto h) { v_sfb_next[h.value] = __builtin_bit_cast(SF_KK, load<T::VEC_LDS_SCALE>(rd_sfb_, sfb_rd_base + h.value * sf_half_stride + sf_read_off)); });
    };
'''+s[j:]
i=s.index('    using SF_DW = opus::vector_t<D_SF_PACK, 2 * T::E_K>;'); j=s.index('    };\n',s.index('auto transpose_sf',i))+7
s=s[:i]+'''    auto read_sf = [&](auto&, auto&, auto h_) {
        constexpr int h = decltype(h_)::value;
        opus::static_for<T::E_K>([&](auto kk) { v_sfa[h][kk.value] = v_sfa_next[h][kk.value]; v_sfb[h][kk.value] = v_sfb_next[h][kk.value]; });
    };
'''+s[j:]
s=s.replace('transpose_sf(number<0>{});','read_sf(rd_sfa, rd_sfb, number<0>{});').replace('transpose_sf(number<1>{});','read_sf(rd_sfa, rd_sfb, number<1>{});')
# loop panel spots -> moved into the C11 region (even / odd copies), tail keeps the boundary
ind='                '
rep(ind+'prefetch_sfb(tile + 2, sf_wr);                 // even tile: sfb of the panel starting at tile + 2\n','')
rep(ind+'prefetch_sfa(tile + 3, sf_wr);\n','',1)
c11=ind+'read_sf(rd_sfa, rd_sfb, number<0>{});\n'
assert s.count(c11)==2
first=s.index(c11); second=s.index(c11, first+1)
ev=ind+'put_sf(rd_sfa, u_ssfa, v_sf_pend, sf_wr);      // even tile: sfa of the panel starting at tile + 2\n'+ind+'v_sf_pend = fetch_sfb(tile + 4);               //            load sfb of the panel starting at tile + 4\n'
od=ind+'put_sf(rd_sfb, u_ssfb, v_sf_pend, sf_wr);      // odd tile: sfb of the panel starting at tile + 3\n'+ind+'v_sf_pend = fetch_sfa(tile + 3);               //           load sfa of the panel starting at tile + 3\n'
s=s[:second+len(c11)]+od+s[second+len(c11):]
s=s[:first+len(c11)]+ev+s[first+len(c11):]
rep('''                if (tile + 3 < loops) prefetch_sfa(tile + 3, sf_wr);
                asm volatile("; panel copy: sfa" ::: "memory");''','''                put_sf(rd_sfb, u_ssfb, v_sf_pend, sf_wr);
                v_sf_pend = fetch_sfa(tile + 3);
                asm volatile("; panel copy: sfa" ::: "memory");''')
rep('''                prefetch_sfb(tile + 2, sf_wr);
                asm volatile("; panel copy: sfb" ::: "memory");''','''                put_sf(rd_sfa, u_ssfa, v_sf_pend, sf_wr);
                v_sf_pend = fetch_sfb(tile + 4);
                asm volatile("; panel copy: sfb" ::: "memory");''')
persistent='if (!prefetched)' in s
if persistent:
    # next tile's panel 0 -> VGPRs during the last K tile (written in the next prologue)
    rep('''        async_load<T::VEC_GLOBAL_SCALE>(ng_sfa, s_sfa.ptr, u_gsfa, u_ssfa + ssfa_offset(0, 0), gsfa_offset(0));
        async_load<T::VEC_GLOBAL_SCALE>(ng_sfb, s_sfb.ptr, u_gsfb, u_ssfb + ssfb_offset(0, 0), gsfb_offset(0));''',
'''        v_nsfa0 = load<T::VEC_GLOBAL_SCALE>(ng_sfa, u_gsfa, gsfa_offset(0));
        v_nsfb0 = load<T::VEC_GLOBAL_SCALE>(ng_sfb, u_gsfb, gsfb_offset(0));''')
    rep('''    int prefetched = 0;''','''    int prefetched = 0;
    opus::vector_t<opus::e8m0_t, Traits::VEC_GLOBAL_SCALE> v_nsfa0, v_nsfb0;   // next tile's scale panel 0''')
    pro_old=s[s.index('    if (!prefetched) {'):s.index('        load_a(rd_a, number<0>{});\n        load_b(rd_b, number<0>{});\n        load_sfa(rd_sfa, 0, 0);')]
    pro_new='''    static_assert(T::SFA_PANEL_BYTES == T::SFB_PANEL_BYTES);
    if (!prefetched) {
        prefetch_a(0, 0, 0);
        prefetch_b(0, 0, 0);
        prefetch_a(0, 0, 1);
        prefetch_b(0, 0, 1);
        v_nsfa0 = fetch_sfa(0);
        v_nsfb0 = fetch_sfb(0);
    }
    // Panel 0 and sfb of panel 1 are transposed into LDS here; sfa of panel 1 stays in v_sf_pend
    // and is written in iteration 0.
    SF_PANEL v_sfb1;
    if (loops > T::SCALE_PANEL_TILES) {
        v_sfb1 = fetch_sfb(T::SCALE_PANEL_TILES);
        v_sf_pend = fetch_sfa(T::SCALE_PANEL_TILES);
    }
    if (loops > 1) {
        prefetch_a(1, 1, 0);
        prefetch_b(1, 1, 0);
        prefetch_a(1, 1, 1);
        prefetch_b(1, 1, 1);
        s_waitcnt_vmcnt(number<4 * T::a_buffer_load_insts + 1>{});      // tile 1 + pending sfa may fly
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    views(s_a.ptr, s_b.ptr, s_sfa.ptr, s_sfb.ptr, [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
        put_sf(rd_sfa, u_ssfa, v_nsfa0, 0);
        put_sf(rd_sfb, u_ssfb, v_nsfb0, 0);
        if (loops > T::SCALE_PANEL_TILES) put_sf(rd_sfb, u_ssfb, v_sfb1, 1);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
'''
    s=s.replace(pro_old,pro_new)
else:
    raise SystemExit('non-persistent input: use sf5.hpp')
open(dst,'w').write(s)
