# Per-item comparison of targeted stalls across traces (cycles per wave per output tile = sum / 16).
import csv,glob,os,sys,json
def load(d):
    f=glob.glob(os.path.join(d,'att','stats_ui_output_agent_*_dispatch_1.csv'))[0]
    rows=[r for r in csv.DictReader(open(f)) if r['Instruction'] and not r['Instruction'].startswith(';')]
    for r in rows:
        for k in ('Hitcount','Latency','Stall','Idle'): r[k]=int(r[k] or 0)
    return rows
def regions(rows):
    mx=max(r['Hitcount'] for r in rows if 'mfma' in r['Instruction'])
    idx=[i for i,r in enumerate(rows) if r['Hitcount']==mx]
    return rows[:idx[0]], rows[idx[0]:idx[-1]+1], rows[idx[-1]+1:]
def S(rows,pred,key): return sum(r[key] for r in rows if pred(r['Instruction']))/16
def waves(d):
    w=[]
    for f in glob.glob(os.path.join(d,'att','ui_output*','se0_sm*_sl0_wv*.json')):
        x=json.load(open(f))['wave']; tl=json.loads(x['timeline']) if isinstance(x['timeline'],str) else x['timeline']
        w.append((int(x['simd']) if 'simd' in x else 0, int(x['begin']), int(x['end']), tl, f))
    return w
names=sys.argv[1:]
res={}
for d in names:
    rows=load(d); pro,loop,epi=regions(rows)
    tot=sum(r['Latency'] for r in rows)/16
    it={}
    it['合计']=tot
    it['T1 prologue lgkmcnt 等待']=S(pro,lambda s:s.startswith('s_waitcnt') and 'lgkmcnt' in s and 'vmcnt' not in s,'Stall')
    it['T1 prologue MFMA idle']=S(pro,lambda s:'mfma' in s,'Idle')
    it['T2 main loop s_barrier 等待']=S(loop,lambda s:s=='s_barrier','Stall')
    it['T3 main loop VALU 时间']=S(loop,lambda s:s.startswith('v_') and 'mfma' not in s and 'accvgpr' not in s,'Latency')
    it['T3 main loop permlane 条数/tile']=sum(r['Hitcount'] for r in loop if 'permlane' in r['Instruction'])/16/2
    it['T3 main loop s_nop 时间']=S(loop,lambda s:s.startswith('s_nop'),'Latency')
    it['T4 写回 ds_write 时间']=S(pro+epi,lambda s:s.startswith('ds_write'),'Latency')
    it['T5 写回 vmcnt(0) 等待']=S(epi+pro,lambda s:s.startswith('s_waitcnt vmcnt(0)'),'Stall')
    it['T5 s_endpgm 等待']=S(rows,lambda s:s.startswith('s_endpgm'),'Latency')
    it['T5 prologue buffer_load 时间']=S(pro,lambda s:s.startswith('buffer_load'),'Latency')
    w=waves(d); bysimd={}
    for simd,b,e,tl,f in w: bysimd.setdefault(f.split('_sm')[1].split('_')[0],[]).append((b,e,tl))
    gaps=[];inits=[]
    for k,v in bysimd.items():
        v.sort()
        for i,(b,e,tl) in enumerate(v):
            if tl and tl[0][0]==1: inits.append(tl[0][1])
            if i: gaps.append(b-v[i-1][1])
    it['T5 block 间空档(平均)']=sum(gaps)/len(gaps) if gaps else 0
    it['T5 新 wave 初始化(平均)']=sum(inits)/len(inits) if inits else 0
    it['T5 每 SIMD 的 wave 数']=len(w)/max(1,len(bysimd))
    res[d]=it
keys=list(next(iter(res.values())).keys())
print('| 项目 | '+' | '.join(os.path.basename(n.rstrip('/')) for n in names)+' |')
print('|---|'+'---|'*len(names))
for k in keys: print(f'| {k} | '+' | '.join(f'{res[n][k]:.0f}' for n in names)+' |')
