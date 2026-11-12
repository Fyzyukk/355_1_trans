import sys,shutil,re
src,name=sys.argv[1],sys.argv[2]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log','README.md'))
t=f'{d}/gemm_a4w4_mxfp4_4wave_traits.hpp'; s=open(t).read()
assert s.count('static constexpr int smem_padding = 32;')==1
s=s.replace('static constexpr int smem_padding = 32;','static constexpr int smem_padding = 64;   // s1: chunk stride 1088 B; with the 16 B swizzle the A/B ds_read_b128 are bank-conflict free')
s=s.replace('static_assert(LDS_BYTES == 159744 && LDS_BYTES <= 163840);','static_assert(LDS_BYTES == 163840 && LDS_BYTES <= 163840);   // s1: padding 64 -> exactly 160 KB')
open(t,'w').write(s)
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
# write side (A and B): K chunk of the lane swizzled by bit 1 of its row in the LDS chunk
a='opus::tuple{wave_id_n, wave_id_m, lane_id / threads_k, lane_id % threads_k}'
b='opus::tuple{wave_id_m, wave_id_n, lane_id / threads_k, lane_id % threads_k}'
sw='(lane_id % threads_k) ^ ((((lane_id / threads_k) >> 1) & 1) << 1)'
assert s.count(a)==1 and s.count(b)==1, (s.count(a),s.count(b))
s=s.replace(a,a.replace('lane_id % threads_k',sw)).replace(b,b.replace('lane_id % threads_k',sw))
# read side: g ^ s(row), row = lane_m / 2
n=0
for m in re.finditer(r'opus::unfold_p_coord\(r[ab]_block_dim, opus::tuple\{(wave_id_[mn]), (lane_[mn]) % T::T_[MN], (lane_[mn]) / T::T_[MN], 0_I, lane_id / T::W_[MN]\}\)',s):
    n+=1
s,k=re.subn(r'(opus::unfold_p_coord\(r[ab]_block_dim, opus::tuple\{wave_id_[mn], (lane_[mn]) % T::T_[MN], lane_[mn] / T::T_[MN], 0_I, )(lane_id / T::W_[MN])\}\)',
            lambda m: f"{m.group(1)}({m.group(3)}) ^ ((({m.group(2)} >> 2) & 1) << 1)}})", s)
assert k==2,(k,n)
assert s.count('static_assert(T::HALF_B_M * c_lds_stride * sizeof(D_C) == stage_elems * sizeof(D_AB));')==1
s=s.replace('static_assert(T::HALF_B_M * c_lds_stride * sizeof(D_C) == stage_elems * sizeof(D_AB));','static_assert(T::HALF_B_M * c_lds_stride * sizeof(D_C) <= stage_elems * sizeof(D_AB));   // s1: stage is 2 KB larger')
open(f,'w').write(s); print(name,'ok')
