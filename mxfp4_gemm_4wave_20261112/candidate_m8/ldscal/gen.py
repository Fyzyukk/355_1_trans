import sys
offs = {}
for line in open('/tmp/w2/dump/offs.txt'):
    if ':' in line:
        t, rest = line.split(':'); offs[int(t)] = [int(x) for x in rest.split()]
pats = []
def add(name, mode, f): pats.append((name, mode, [f(t) for t in range(256)]))
add('cal_contig_b128', 1, lambda t: (t%64)*16)
add('cal_samebank_b128', 1, lambda t: (t%64)*256)
add('Cwrite2_b64', 0, lambda t: offs[t][0]*2)
add('Cwrite2_b64_+128B', 0, lambda t: offs[t][0]*2+128)
add('Cread_b128', 1, lambda t: ((t*8)//256)*528 + ((t*8)%256)*2)
add('putsf_b128', 2, lambda t: (t//64)*1024 + ((t%4)*16 + (t%64)//4)*16)
add('putsf_lin_b128', 2, lambda t: (t//64)*1024 + (t%64)*16)
add('sfread2st64', 3, lambda t: ((t//64)%2*64 + t%64)*16)
add('sfread2st64_+8', 3, lambda t: ((t//64)%2*64 + t%64)*16 + 8)
add('sfread_b64', 4, lambda t: ((t//64)%2*64 + t%64)*16)
add('cal_contig_b64', 4, lambda t: (t%64)*8)
add('Cwrite_b64_single', 5, lambda t: offs[t][0]*2)
with open(sys.argv[1], 'w') as f:
    for n, m, a in pats: f.write('%d %s\n' % (m, ' '.join(map(str, a))))
with open(sys.argv[1] + '.names', 'w') as f:
    for n, m, a in pats: f.write(n + '\n')
