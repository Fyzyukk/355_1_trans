import itertools
def addrs_PS(P,S): return [(l%P)*16+(l//P)*S for l in range(64)]
def kern(H,kk):
    out=[]
    for l in range(64):
        r,g=l&15,l>>4; h=(H>>(2*r))&3
        out.append((r&1)*1056+(r>>1)*128+((g^h)*16)+kk*64)
    return out
obs=[(addrs_PS(1,256),60),(addrs_PS(64,0),0),(addrs_PS(1,32),4),(addrs_PS(16,4224),4),(addrs_PS(16,4096),0),(addrs_PS(16,128),4),
     (addrs_PS(32,4096),0),(addrs_PS(8,4096),4),(addrs_PS(8,128),0),(addrs_PS(4,4096),12)]
for n,H,v in [('none',0x0,4),('row&3',0xfa50fa50,4),('(row>>1)&3',0xffaa5500,4),('row&3^par<<1',0x72d872d8,4),('row>>1&3^par',0xbbee1144,12),
              ('par*2',0x88888888,8),('row&1|par<<1',0xd8d8d8d8,8),('(row>>1)&3^par<<1',0x7722dd88,12),('rev',0x5af05af,4),('row>>2|par',0xdddd8888,8)]:
    obs.append((kern(H,0),v))
def cost(addrs,cyc,nb,bw):
    # each lane reads 16 B = 16/bw banks starting at addr/bw
    tot=0
    for c in range(4):
        cnt={}
        for l in range(64):
            if cyc(l)!=c: continue
            for j in range(16//bw):
                b=(addrs[l]//bw+j)%nb; cnt[b]=cnt.get(b,0)+1
        tot+=max(cnt.values())-1
    return tot
res=[]
for nb,bw in [(64,4),(128,4),(32,4),(64,8),(32,8)]:
  for b1,b2 in itertools.combinations(range(6),2):
    cyc=lambda l,b1=b1,b2=b2: ((l>>b1)&1)|(((l>>b2)&1)<<1)
    err=sum(abs(cost(a,cyc,nb,bw)-v) for a,v in obs)
    res.append((err,nb,bw,b1,b2))
res.sort(); print(res[:8])
print('--- 扩展模型 ---')
def cost2(addrs,lane_bits,split,nb=64,bw=4):
    # groups: by chosen lane bits (and, if split, by which 8-B half); each group is one sub-cycle of weight 4/ngroups
    ng=(1<<len(lane_bits))*(2 if split else 1); w=4.0/ng; tot=0
    groups={}
    for l in range(64):
        key=tuple((l>>b)&1 for b in lane_bits)
        halves=[(0,8),(8,8)] if split else [(0,16)]
        for hi,(o,sz) in enumerate(halves):
            k2=key+((hi,) if split else ())
            d=groups.setdefault(k2,{})
            for j in range(sz//bw):
                b=((addrs[l]+o)//bw+j)%nb; d[b]=d.get(b,0)+1
    for d in groups.values(): tot+=(max(d.values())-1)*w
    return tot
res=[]
for split in (False,True):
  for k in (1,2,3):
    for bits in itertools.combinations(range(6),k):
      for nb in (64,128):
        e=sum(abs(cost2(a,bits,split,nb)-v) for a,v in obs)
        res.append((round(e,2),split,bits,nb))
res.sort(); print(res[:8])
best=res[0]
print('逐项对比（最佳模型）:')
for a,v in obs: print(v, round(cost2(a,best[2],best[1],best[3]),2))
print('--- bank = (addr>>2) ^ (高位 XOR)，16 连续 lane 一拍 ---')
def cost3(addrs,xf,group=16):
    tot=0
    for c in range(64//group):
        cnt={}
        for l in range(c*group,(c+1)*group):
            for j in range(4):
                a=addrs[l]+4*j; b=((a>>2) ^ xf(a)) & 63; cnt[b]=cnt.get(b,0)+1
        tot+=(max(cnt.values())-1)*(group/16)
    return tot
X={}
for sh in range(8,13):
  for mbits in (1,2,3):
    for dst in range(0,6):
      if dst+mbits>6: continue
      X[f'((a>>{sh})&{(1<<mbits)-1})<<{dst}']=lambda a,sh=sh,mb=mbits,dst=dst: ((a>>sh)&((1<<mb)-1))<<dst
X['none']=lambda a:0
res=[]
for n,f in X.items():
  for grp in (16,):
    e=sum(abs(cost3(a,f,grp)-v) for a,v in obs); res.append((e,n,grp))
res.sort(); print(res[:6])
f=X[res[0][1]]
print([ (v, cost3(a,f)) for a,v in obs])
