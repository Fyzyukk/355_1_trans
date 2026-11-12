import sys, itertools
pats = []
def add(name, mode, f): pats.append((name, mode, [f(t % 64) + (t // 64) * 4096 for t in range(256)]))
def compress(l, bits):  # remove given bits from 6-bit lane id
    out, k = 0, 0
    for b in range(6):
        if b in bits: continue
        out |= ((l >> b) & 1) << k; k += 1
    return out
# b64: lanes l and l^(1<<m) share bank
for m in range(6):
    add('b64w_pair_bit%d' % m, 5, lambda l, m=m: compress(l, {m}) * 8 + ((l >> m) & 1) * 256)
    add('b64r_pair_bit%d' % m, 4, lambda l, m=m: compress(l, {m}) * 8 + ((l >> m) & 1) * 256)
# b128: lanes differing only in bits a,b share banks
for a, b in itertools.combinations(range(6), 2):
    add('b128w_bits%d%d' % (a, b), 2, lambda l, a=a, b=b: compress(l, {a, b}) * 16 + (((l >> a) & 1) + 2 * ((l >> b) & 1)) * 256)
    add('b128r_bits%d%d' % (a, b), 1, lambda l, a=a, b=b: compress(l, {a, b}) * 16 + (((l >> a) & 1) + 2 * ((l >> b) & 1)) * 256)
with open(sys.argv[1], 'w') as f:
    for n, m, a in pats: f.write('%d %s\n' % (m, ' '.join(map(str, a))))
with open(sys.argv[1] + '.names', 'w') as f:
    for n, m, a in pats: f.write(n + '\n')
