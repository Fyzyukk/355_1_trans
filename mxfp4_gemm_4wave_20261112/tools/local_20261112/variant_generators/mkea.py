import sys,shutil,re
src,name=sys.argv[1],sys.argv[2]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log','README.md'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
def rep(a,b):
    global s; assert s.count(a)==1,(s.count(a),a[:80]); s=s.replace(a,b)
# load_b_kk lambda right after load_b
i=s.index("    auto load_b = [&](auto& sb_, auto half_) {"); j=s.index("    };",i)+len("    };")
lb=s[i:j]
kk=lb.replace("auto load_b = [&](auto& sb_, auto half_) {","auto load_b_kk = [&](auto& sb_, auto half_, auto kk_) {   // ea: one kk of a half\n        constexpr int kkv = decltype(kk_)::value;")
kk=kk.replace("opus::static_for<T::E_K>([&](auto kk) { opus::static_for<T::E_N>([&](auto e) {","opus::static_for<1>([&](auto) { constexpr opus::number<kkv> kk{}; opus::static_for<T::E_N>([&](auto e) {")
assert kk!=lb
s=s[:j]+"\n"+kk+s[j:]
# C01: kk = 0 only ; C11: + kk = 1   (loop-level indents only, not the prologue)
s,n1=re.subn(r'(?m)^( {12,})load_b\(rd_b, number<0>\{\}\);$', r'\1load_b_kk(rd_b, number<0>{}, number<0>{});   // ea: kk = 1 in C11', s)
s,n2=re.subn(r'(?m)^( {12,})load_a\(rd_a, number<0>\{\}\);$', r'\1load_a(rd_a, number<0>{});\n\1load_b_kk(rd_b, number<0>{}, number<1>{});   // ea', s)
print('C01',n1,'C11',n2); assert n1==4 and n2==4
# sched_c11: 12 DS (8 A0 + 4 B0 kk1)
i=s.index("__device__ inline void sched_c11() {"); j=s.index("\n}\n",i)+3
old=s[i:j]; print(old)
s=s[:i]+"""__device__ inline void sched_c11() {   // ea: 12 DS (A0 + B0 kk = 1); e1: scale ds_write + VMEM after the last DS
    opus::static_for<4>([&](auto) { sched_m<1, 1>(); sched_m<0, 1>(); sched_m<1, 1>(); sched_m<0, 0>(); });
    opus::static_for<3>([&](auto) { sched_m<1, 1>(); sched_m<0, 1>(); sched_m<0, 1>(); sched_m<0, 0>(); });
    sched_m<1, 1>();
    __builtin_amdgcn_sched_group_barrier(0x200, 1, 0);
    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);
    sched_m<0, 1>(); sched_m<0, 1>(); sched_m<0, 0>();
}
"""+s[j:]
open(f,'w').write(s); print(name,'ok')
