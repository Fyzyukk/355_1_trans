import sys,shutil
src,name=sys.argv[1],sys.argv[2]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log','README.md'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
def rep(a,b):
    global s; assert s.count(a)==1,(s.count(a),a[:80]); s=s.replace(a,b)
rep("// sched_group_barrier masks (LLVM AMDGPU)","""// rs: tile order.  With 256 persistent blocks (block b -> XCD b % 8), the 256 tiles in flight form a 16 x 16
// square instead of 8 rows x 32 columns: per XCD still 4 x 8 tiles (12 A/B blocks per K tile), but the whole
// GPU touches 32 instead of 40 A/B blocks per K tile (less traffic behind the L2s).  Other shapes: row-major.
__device__ inline void tile_coord(int t, int num_tiles_m, int num_tiles_n, int grid_blocks, int& bm, int& bn) {
    if (grid_blocks == 256 && num_tiles_m % 16 == 0 && num_tiles_n % 16 == 0) {
        const int wv = t / 256, x = t % 8, j = (t % 256) / 8;
        const int sm = wv / (num_tiles_n / 16), sn = wv % (num_tiles_n / 16);
        bm = sm * 16 + (x / 2) * 4 + j / 8;
        bn = sn * 16 + (x % 2) * 8 + j % 8;
    } else {
        bm = t / num_tiles_n;
        bn = t % num_tiles_n;
    }
}
// sched_group_barrier masks (LLVM AMDGPU)""")
rep("""    const int block_m = tile_in_batch / num_tiles_n;
    const int block_n = tile_in_batch % num_tiles_n;""","""    int block_m, block_n;
    tile_coord(tile_in_batch, tiles_per_batch / num_tiles_n, num_tiles_n, grid_blocks, block_m, block_n);   // rs""")
rep("""        const int n_row = (n_in_batch / num_tiles_n) * T::B_M;
        const int n_col = (n_in_batch % num_tiles_n) * T::B_N;""","""        int n_bm, n_bn;
        tile_coord(n_in_batch, tiles_per_batch / num_tiles_n, num_tiles_n, grid_blocks, n_bm, n_bn);   // rs
        const int n_row = n_bm * T::B_M;
        const int n_col = n_bn * T::B_N;""")
open(f,'w').write(s); print(name,'ok')
