import json,collections,sys
c=json.load(open(sys.argv[1]))['code']
W=16
ops=[r[0].split()[0] if r[0] else '' for r in c]
# loop region = between the first and last instruction with the max hit count
mx=max(r[6] for r in c); idx=[i for i,r in enumerate(c) if r[6]==mx]; a,b=idx[0],idx[-1]
def k(op):
    return 'mfma' if 'mfma' in op else 'waitcnt' if 'waitcnt' in op else 'barrier' if op=='s_barrier' else 'buf_load' if op.startswith('buffer_load') else 'buf_store' if op.startswith('buffer_store') else 'ds' if op.startswith('ds_') else 'nop' if op=='s_nop' else 'salu' if op.startswith('s_') else 'valu' if op.startswith('v_') else 'other'
tot=sum(r[7] for r in c)/W
print(f'per wave total {tot:.0f}')
for name,(x,y) in [('prologue',(0,a)),('main loop',(a,b+1)),('tail+epilogue',(b+1,len(c)))]:
    L=c[x:y]; lat=sum(r[7] for r in L)/W; cat=collections.Counter()
    for r,op in zip(L,ops[x:y]): cat[k(op)]+=r[7]/W
    print(f'{name:14s} {lat:8.0f} ({lat/tot:5.1%})', {kk:round(v) for kk,v in cat.most_common()})
iters=mx/W
print(f'main loop trips {iters:.0f}')
print('top stalls in loop (per hit):')
for r in sorted(c[a:b+1],key=lambda r:-r[8])[:12]: print(f'  line {r[2]:5d} stall/hit {r[8]/r[6]:6.1f} {r[0][:80]}')
print('top stalls outside loop (per wave):')
for r in sorted(c[:a]+c[b+1:],key=lambda r:-r[8])[:8]: print(f'  line {r[2]:5d} stall/wave {r[8]/W:7.0f} {r[0][:80]}')
