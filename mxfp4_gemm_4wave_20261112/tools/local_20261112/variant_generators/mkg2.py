import sys,shutil,re
src,name,P=sys.argv[1],sys.argv[2],int(sys.argv[3]); d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
pat=re.compile(r'''(?P<i>[ ]+)s_waitcnt_lgkmcnt\(0_I\);\n(?P=i)__builtin_amdgcn_s_barrier\(\);\n(?P=i)__builtin_amdgcn_sched_barrier\(0\);\n\n(?P=i)// C10\(t\) \+ A0, B0 of tile t\+2 -> stage\(t\), one buffer_load per 4 MFMAs.\n(?P=i)prefetch_a\(stage, tile \+ 2, 0\);\n(?P=i)prefetch_b\(stage, tile \+ 2, 0\);\n(?P<mf10>(?P=i)opus::static_for<frags>.*?\n(?P=i)\}\);\n)(?P=i)sched_x4<8, 0, 1, 0>\(\);\n(?P=i)__builtin_amdgcn_sched_barrier\(0\);\n(?P<bb>(?P=i)// B_b:[^\n]*\n)(?P=i)s_waitcnt_vmcnt\(number<2 \* T::a_buffer_load_insts>\{\}\);\n(?P=i)__builtin_amdgcn_s_barrier\(\);\n(?P=i)__builtin_amdgcn_sched_barrier\(0\);\n\n(?P<c01>(?P=i)// C01.*?\n(?:(?P=i)load_[^\n]*\n)+)(?P=i)prefetch_a\(stage, tile \+ 2, 1\);\n(?P=i)prefetch_b\(stage, tile \+ 2, 1\);\n(?P<mf01>(?P=i)opus::static_for<frags>.*?\n(?P=i)\}\);\n)(?P=i)sched_x4<4, 2, 1, 0>\(\);\n(?P=i)sched_x4<4, 1, 1, 0>\(\);\n''',re.S)
def split(mf,off,cnt): return mf.replace('opus::static_for<frags>',f'opus::static_for<{cnt}>').replace('decltype(i)::value',f'(decltype(i)::value + {off})')
R=(32-2*P)//4
def sub(m):
    i=m['i']; sb=f"{i}__builtin_amdgcn_sched_barrier(0);\n"
    return (f"{i}// g2: {2*P} C10 MFMAs keep the MFMA pipe busy while B_a waits.\n{split(m['mf10'],0,P)}{sb}"
            f"{i}s_waitcnt_lgkmcnt(0_I);\n{i}__builtin_amdgcn_s_barrier();\n{sb}\n"
            f"{i}prefetch_a(stage, tile + 2, 0);\n{i}prefetch_b(stage, tile + 2, 0);\n{split(m['mf10'],P,16-P)}{i}sched_x4<{R}, 0, 1, 0>();\n{sb}"
            f"{i}// g2: {2*P} C01 MFMAs (operands already in registers) before B_b.\n{split(m['mf01'],0,P)}{sb}"
            f"{m['bb']}{i}s_waitcnt_vmcnt(number<2 * T::a_buffer_load_insts>{{}});\n{i}__builtin_amdgcn_s_barrier();\n{sb}\n"
            f"{m['c01']}{i}prefetch_a(stage, tile + 2, 1);\n{i}prefetch_b(stage, tile + 2, 1);\n{split(m['mf01'],P,16-P)}"
            f"{i}sched_x4<4, 2, 1, 0>();\n{i}sched_x4<{R-4}, 1, 1, 0>();\n")
s,n=pat.subn(sub,s); print(name,'replaced',n); assert n==3
open(f,'w').write(s)
