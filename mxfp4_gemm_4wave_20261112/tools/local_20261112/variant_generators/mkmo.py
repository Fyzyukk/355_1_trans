# mo: in every quadrant, issue the 16 kk=0 MFMAs, then the 16 kk=1 MFMAs (adjacent MFMAs never share C).
import sys,shutil,re
src,name=sys.argv[1],sys.argv[2]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log','README.md'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
pat=re.compile(r'(?P<i>[ ]*)opus::static_for<frags>\(\[&\]\(auto i\) \{\n(?P<c>[ ]*constexpr int m = decltype\(i\)::value / T::E_N, n = decltype\(i\)::value % T::E_N;\n)(?P<k0>[ ]*mma_scale_one<[^\n]*// kk = 0\n)(?P<k1>[ ]*mma_scale_one<[^\n]*// kk = 1\n)(?P=i)\}\);\n')
def sub(m):
    i=m['i']
    return (f"{i}opus::static_for<frags>([&](auto i) {{   // mo: kk = 0 for all fragments first\n{m['c']}{m['k0']}{i}}});\n"
            f"{i}opus::static_for<frags>([&](auto i) {{   // mo: then kk = 1\n{m['c']}{m['k1']}{i}}});\n")
s,n=pat.subn(sub,s); print(name,'quadrants rewritten:',n)
open(f,'w').write(s)
