import sys
offs = {}
for line in open('/tmp/w2/dump/offs.txt'):
    if ':' in line:
        t, rest = line.split(':'); offs[int(t)] = [int(x) for x in rest.split()]
pats = []
def add(name, mode, f): pats.append((name, mode, [f(t) for t in range(256)]))
def cw_swz(t, extra=0):
    b = offs[t][0]*2 + extra          # bytes; row = b // 528, col byte = b % 528
    row, cb = b // 528, b % 528
    return row*528 + (cb ^ (((row >> 3) & 1) << 3))
add('Cw2_old', 0, lambda t: offs[t][0]*2)
add('Cw2_swz', 0, lambda t: cw_swz(t))
add('Cw2_swz_+128', 0, lambda t: cw_swz(t, 128))
add('Cw2_swz_+256', 0, lambda t: cw_swz(t, 256))
crd = lambda t, i: ((t*8 + i*2048)//256)*528 + ((t*8)%256)*2
add('Cr_b128_i0', 1, lambda t: crd(t, 0))
add('Cr_b128_i1', 1, lambda t: crd(t, 1))
add('Cr_read2_b64_i0', 9, lambda t: crd(t, 0))
add('Cr_read2swp_b64_i1', 6, lambda t: crd(t, 1))
add('putsf_old', 2, lambda t: (t//64)*1024 + ((t%4)*16 + (t%64)//4)*16)
def phys(s): return s ^ (((s >> 4) & 3) << 2)
add('putsf_new_w2st64', 7, lambda t: (t//64)*512 + phys((t%4)*16 + (t%64)//4)*8)
add('putsf_new_w2st64_+1024', 7, lambda t: 1024 + (t//64)*512 + phys((t%4)*16 + (t%64)//4)*8)
add('sfrd_old', 3, lambda t: ((t//64)%2*64 + t%64)*16)
add('sfrd_new', 8, lambda t: ((t//64)%2*64 + phys(t%64))*8)
add('sfrd_new_tip1', 8, lambda t: 2048 + ((t//64)%2*64 + phys(t%64))*8)
with open(sys.argv[1], 'w') as f:
    for n, m, a in pats: f.write('%d %s\n' % (m, ' '.join(map(str, a))))
with open(sys.argv[1] + '.names', 'w') as f:
    for n, m, a in pats: f.write(n + '\n')
