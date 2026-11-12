import sys,shutil
src,name,units,mode=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4]); d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
a="""#pragma clang loop unroll(disable)
    for (int tile_id = block_id_x(); tile_id < total_tiles; tile_id += grid_blocks) {"""
assert s.count(a)==1
# mode 0: block parity (whole XCDs); mode 1: (block/8)&1 (half of each XCD); mode 2: 4 phases (block/8)%4 * units
grp={0:"(block_id_x() & 1)",1:"((block_id_x() >> 3) & 1)",2:"((block_id_x() >> 3) & 3)"}[mode]
b=f"""    {{   // stagger: offset the tile transitions of part of the CUs (units x s_sleep(32) ~ 2048 clk)
        const int st = {grp} * {units};
        for (int i = 0; i < st; ++i) __builtin_amdgcn_s_sleep(32);
    }}
"""+a
s=s.replace(a,b); open(f,'w').write(s); print(name)
