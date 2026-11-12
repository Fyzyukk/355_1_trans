import sys,shutil,re
src,name=sys.argv[1],sys.argv[2]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
# new sched helper: 2 VMEM per 4 MFMAs
helper='''
// g1: C01 with all 16 copies of tile t+2: MFMA, DS, MFMA, VMEM, MFMA, DS, MFMA, VMEM.
template<int REPS, int DS>
__device__ inline void sched_x4v2() {
    static_assert(DS <= 2);
    opus::static_for<REPS>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (DS >= 1) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (DS >= 2) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, 1, 0);
    });
}
'''
s=s.replace('// One fp4 MFMA operand per lane', helper+'\n// One fp4 MFMA operand per lane',1)
pat=re.compile(r'''(?P<i>[ ]+)// B_a: every wave is done reading stage\(t\).*?\n(?P=i)s_waitcnt_lgkmcnt\(0_I\);\n(?P=i)__builtin_amdgcn_s_barrier\(\);\n(?P=i)__builtin_amdgcn_sched_barrier\(0\);\n\n(?P=i)// C10\(t\) \+ A0, B0 of tile t\+2 -> stage\(t\), one buffer_load per 4 MFMAs.\n(?P=i)prefetch_a\(stage, tile \+ 2, 0\);\n(?P=i)prefetch_b\(stage, tile \+ 2, 0\);\n(?P<mf>.*?)(?P=i)sched_x4<8, 0, 1, 0>\(\);\n(?P=i)__builtin_amdgcn_sched_barrier\(0\);\n(?P=i)// B_b:[^\n]*\n(?P=i)s_waitcnt_vmcnt\(number<2 \* T::a_buffer_load_insts>\{\}\);\n(?P=i)__builtin_amdgcn_s_barrier\(\);\n(?P=i)__builtin_amdgcn_sched_barrier\(0\);\n\n(?P<c01>(?P=i)// C01.*?\n(?:(?P=i)load_[^\n]*\n)+)(?P=i)prefetch_a\(stage, tile \+ 2, 1\);\n(?P=i)prefetch_b\(stage, tile \+ 2, 1\);\n(?P<mf2>.*?)(?P=i)sched_x4<4, 2, 1, 0>\(\);\n(?P=i)sched_x4<4, 1, 1, 0>\(\);\n''',re.S)
def sub(m):
    i=m['i']
    return (f"{i}// g1: no B_a -- C10 issues no copies, so B_b (after C10) also orders the C00 reads of stage(t).\n"
            f"{i}__builtin_amdgcn_sched_barrier(0);\n\n"
            f"{i}// C10(t): MFMAs only.\n{m['mf']}{i}sched_x4<8, 0, 0, 0>();\n{i}__builtin_amdgcn_sched_barrier(0);\n"
            f"{i}// B_b (g1): tile t+1 + its scale panel landed, and every wave is done reading stage(t).\n"
            f"{i}s_waitcnt_lgkmcnt(0_I);\n{i}s_waitcnt_vmcnt(0_I);\n{i}__builtin_amdgcn_s_barrier();\n{i}__builtin_amdgcn_sched_barrier(0);\n\n"
            f"{m['c01']}{i}prefetch_a(stage, tile + 2, 0);\n{i}prefetch_b(stage, tile + 2, 0);\n{i}prefetch_a(stage, tile + 2, 1);\n{i}prefetch_b(stage, tile + 2, 1);\n"
            f"{m['mf2']}{i}sched_x4v2<4, 2>();\n{i}sched_x4v2<4, 1>();\n")
s,n=pat.subn(sub,s); print(name,'replaced',n); assert n==3
open(f,'w').write(s)
