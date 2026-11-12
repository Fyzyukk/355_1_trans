import sys,shutil
name=sys.argv[1]; E4='34' in name
src='/tmp/exp/e1e2a'; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
def rep(a,b,cnt=1):
    global s; n=s.count(a); assert n==cnt,(n,a[:60]); s=s.replace(a,b)
# 1. next tile's scale panel 1
rep("opus::vector_t<opus::e8m0_t, Traits::VEC_GLOBAL_SCALE> v_nsfa0, v_nsfb0;   // next tile's scale panel 0",
    "opus::vector_t<opus::e8m0_t, Traits::VEC_GLOBAL_SCALE> v_nsfa0, v_nsfb0;   // next tile's scale panel 0\n"
    "    opus::vector_t<opus::e8m0_t, Traits::VEC_GLOBAL_SCALE> v_nsfb1, v_nsfp;    // e3: next tile's sfb / sfa of panel 1")
# 3. prologue uses them
rep("""        v_sfb1 = fetch_sfb(T::SCALE_PANEL_TILES);
        v_sf_pend = fetch_sfa(T::SCALE_PANEL_TILES);
    }""","""        if (prefetched) { v_sfb1 = v_nsfb1; v_sf_pend = v_nsfp; }   // e3: fetched in "2. Last tile"
        else {
            v_sfb1 = fetch_sfb(T::SCALE_PANEL_TILES);
            v_sf_pend = fetch_sfa(T::SCALE_PANEL_TILES);
        }
    }""")
# 2. fetch in last tile C11
rep("""        v_nsfb0 = load<T::VEC_GLOBAL_SCALE>(ng_sfb, u_gsfb, gsfb_offset(0));""",
"""        v_nsfb0 = load<T::VEC_GLOBAL_SCALE>(ng_sfb, u_gsfb, gsfb_offset(0));
        v_nsfb1 = load<T::VEC_GLOBAL_SCALE>(ng_sfb, u_gsfb, gsfb_offset(T::SCALE_PANEL_TILES));   // e3 (out of range -> 0)
        v_nsfp = load<T::VEC_GLOBAL_SCALE>(ng_sfa, u_gsfa, gsfa_offset(T::SCALE_PANEL_TILES));""")
rep("""        sched_x4<2, 0, 1, 0>();
        sched_x4<6, 0, 0, 0>();""","""        sched_x4<4, 0, 1, 0>();   // e3: 4 scale loads
        sched_x4<4, 0, 0, 0>();""")
if E4:
    # prologue: k-tile 1 already in flight
    rep("""    if (loops > 1) {
        prefetch_a(1, 1, 0);
        prefetch_b(1, 1, 0);
        prefetch_a(1, 1, 1);
        prefetch_b(1, 1, 1);
        s_waitcnt_vmcnt(number<4 * T::a_buffer_load_insts + 1>{});      // tile 1 + pending sfa may fly
    }""","""    if (loops > 1 && prefetched) {
        // e4: K tile 1 was copied during the write-back (after the half-0 stores, before the
        // half-1 stores); K tile 0 + scales are older than those 48 operations.
        s_waitcnt_vmcnt(number<4 * T::a_buffer_load_insts + 32>{});
    } else if (loops > 1) {
        prefetch_a(1, 1, 0);
        prefetch_b(1, 1, 0);
        prefetch_a(1, 1, 1);
        prefetch_b(1, 1, 1);
        s_waitcnt_vmcnt(number<4 * T::a_buffer_load_insts + 1>{});      // tile 1 + pending sfa may fly
    }""")
    # writeback half 1: reads, barrier, next tile's K tile 1 -> stage 1, stores
    rep("""    copy_output_half(1);""","""    {   // e4: half 1 -- LDS reads, barrier (stage 1 free), next tile's K tile 1 -> stage 1, stores
        opus::vector_t<D_C, VEC_OUT> v[c_copies];
        opus::static_for<c_copies>([&](auto i) {
            const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
            v[i.value] = load<VEC_OUT>(s_c, (linear / T::B_N) * c_lds_stride + linear % T::B_N);
        });
        sync_lds();
        if (loops > 1) {
            const int next_id = tile_id + grid_blocks;
            const bool has_next = next_id < total_tiles;
            const int nid = has_next ? next_id : tile_id;
            const int n_batch = nid / tiles_per_batch;
            const int n_in_batch = nid % tiles_per_batch;
            const int n_row = (n_in_batch / num_tiles_n) * T::B_M;
            const int n_col = (n_in_batch % num_tiles_n) * T::B_N;
            auto ng_a = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_a) + n_batch * kargs.stride_a_batch + n_row * kargs.stride_a, has_next ? 0xffffffffu : 0u);
            auto ng_b = make_gmem(reinterpret_cast<const D_AB*>(kargs.ptr_b) + n_batch * kargs.stride_b_batch + n_col * kargs.stride_b, has_next ? 0xffffffffu : 0u);
            async_load_imm<T::VEC_A>(ng_a, s_a.ptr, u_ga, u_sa + sa_offset(1, 0), ga_offset(0, 1));
            async_load_imm<T::VEC_B>(ng_b, s_b.ptr, u_gb, u_sb + sb_offset(1, 0), gb_offset(0, 1));
            async_load_imm<T::VEC_A>(ng_a, s_a.ptr, u_ga, u_sa + sa_offset(1, 1), ga_offset(1, 1));
            async_load_imm<T::VEC_B>(ng_b, s_b.ptr, u_gb, u_sb + sb_offset(1, 1), gb_offset(1, 1));
        }
        __builtin_amdgcn_sched_barrier(0);
        opus::static_for<c_copies>([&](auto i) {
            const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
            const int r = linear / T::B_N + T::HALF_B_M, c = linear % T::B_N;
            store<VEC_OUT>(g_c, v[i.value], r * kargs.stride_c + c, 0, opus::number<2>{});
        });
    }""")
open(f,'w').write(s); print(name,'patched')
