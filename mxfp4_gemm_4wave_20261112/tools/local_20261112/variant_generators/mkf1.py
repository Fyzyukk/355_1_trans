import sys,shutil
src,name=sys.argv[1],sys.argv[2]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
def rep(a,b):
    global s; assert s.count(a)==1,a[:50]; s=s.replace(a,b)
rep("""    opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
        clear(v_c[hm.value][hn.value][f.value]);
    }); }); });
""","""    auto clear_c = [&]() {   // f1: called in the prologue, under the scale-load wait
        opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
            clear(v_c[hm.value][hn.value][f.value]);
        }); }); });
    };
""")
rep("""        load_a(rd_a, number<0>{});
        load_b(rd_b, number<0>{});
        put_sf(rd_sfa, u_ssfa, v_nsfa0, 0);""","""        load_a(rd_a, number<0>{});
        load_b(rd_b, number<0>{});
        __builtin_amdgcn_sched_barrier(0);
        clear_c();                                     // f1: overlaps the wait for the scale loads
        __builtin_amdgcn_sched_barrier(0);
        put_sf(rd_sfa, u_ssfa, v_nsfa0, 0);""")
open(f,'w').write(s); print(name,'ok')
