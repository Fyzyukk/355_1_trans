import json,glob,sys,collections
d=sys.argv[1]; want=[int(x) for x in sys.argv[2].split(',')]; W=16
c=json.load(open(glob.glob(f'{d}/att/ui_output*/code.json')[0]))['code']
mx=max(r[6] for r in c if r[0] and 'mfma' in r[0])
segs=[]; cur=[]
for r in c:
    if not r[0] or r[6]==0: continue
    op=r[0].split()[0]
    if r[6]==mx:
        if cur and cur[-1]!='LOOP': segs.append(cur); cur=[]
        if not segs or segs[-1]!='LOOP': segs.append('LOOP')
        continue
    cur.append(r)
    if op=='s_barrier' or op=='s_endpgm' or op.startswith('s_cbranch'): segs.append(cur); cur=[]
if cur: segs.append(cur)
for i in want:
    s=segs[i]; cnt=collections.Counter(r[0].split()[0] for r in s)
    print(f'==== #{i}: {len(s)} 条, 源码行 {min(r[2] for r in s)}-{max(r[2] for r in s)}')
    print('   指令构成:', ', '.join(f'{k}×{v}' for k,v in cnt.most_common(12)))
    for r in sorted(s,key=lambda r:-r[8])[:5]:
        if r[8]/W>=15: print(f'   stall {r[8]/W:6.0f} (每次 {r[8]/max(1,r[6]):5.1f}, hit {r[6]})  {r[0][:85]}')
