import csv,sys,collections,glob,os
def cat(op):
    if 'mfma' in op: return 'MFMA'
    if op.startswith('buffer_load'): return 'buffer_load'
    if op.startswith('buffer_store'): return 'buffer_store'
    if op.startswith('scratch_'): return 'scratch'
    if op.startswith('s_waitcnt'): return 'waitcnt'
    if op=='s_barrier': return 'barrier'
    if op=='s_nop': return 'nop'
    if op.startswith('ds_read'): return 'ds_read'
    if op.startswith('ds_write'): return 'ds_write'
    if op.startswith('v_accvgpr'): return 'accvgpr'
    if op.startswith('v_'): return 'VALU'
    if op.startswith('s_'): return 'SALU'
    return 'other'
def load(d):
    f=glob.glob(os.path.join(d,'att','stats_ui_output_agent_*_dispatch_1.csv'))[0]
    rows=[r for r in csv.DictReader(open(f)) if r['Instruction'] and not r['Instruction'].startswith(';')]
    return rows
res={}
for d in sys.argv[1:]:
    rows=load(d); c=collections.defaultdict(lambda:[0,0])
    for r in rows:
        op=r['Instruction'].split()[0]; k=cat(op); c[k][0]+=int(r['Hitcount']); c[k][1]+=int(r['Latency'])
    tot=sum(v[1] for v in c.values())
    res[d]=(tot/16,c,tot,rows)
names=list(res)
print('| 类别 | '+' | '.join(f'{os.path.basename(n.rstrip("/"))} cycle/wave (占比)' for n in names)+' |')
print('|---|'+'---|'*len(names))
print('| **合计** | '+' | '.join(f'**{res[n][0]:.0f}**' for n in names)+' |')
keys=sorted({k for n in names for k in res[n][1]}, key=lambda k:-res[names[-1]][1].get(k,[0,0])[1])
for k in keys:
    print(f'| {k} | '+' | '.join(f'{res[n][1].get(k,[0,0])[1]/16:.0f} ({res[n][1].get(k,[0,0])[1]/res[n][2]:.1%})' for n in names)+' |')
for n in names:
    rows=res[n][3]; tot=res[n][2]
    print(f'\n{os.path.basename(n.rstrip("/"))} 最耗时指令（hits>=200，按 latency）：')
    for r in sorted([r for r in rows if int(r['Hitcount'])>=200], key=lambda r:-int(r['Latency']))[:8]:
        print(f'  {r["Vaddr"]:>6} {int(r["Latency"])/tot:5.1%} {int(r["Latency"])/int(r["Hitcount"]):6.0f} cyc/次  {r["Instruction"][:70]}')
