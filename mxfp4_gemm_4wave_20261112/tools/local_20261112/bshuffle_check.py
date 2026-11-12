# CPU-only simulation of B global->LDS (LDS-DMA) -> ds_read_b128 for the mxfp4 4-wave kernel.
import numpy as np
N, KB = 512, 1024            # N rows, K bytes per row (K = 2048 fp4)
HALF_N, BKB, P = 128, 128, 1088   # half tile rows, K-tile bytes, chunk stride (1024 + 64)
rng = np.random.default_rng(0)
plain = rng.integers(0, 256, size=(N, KB), dtype=np.uint8)

def shuffle_weight_16_16(b):  # aiter shuffle_weight(layout=(16,16)) on a uint8 [N, KB] buffer
    n, kb = b.shape
    return b.reshape(n // 16, 16, kb // 32, 2, 16).transpose(0, 2, 3, 1, 4).copy().reshape(n, kb)
shuf = shuffle_weight_16_16(plain).reshape(-1)

def shuf_off(row, kbyte):     # byte offset of plain[row, kbyte] in the shuffled buffer
    return (row // 16) * 16 * KB + (kbyte // 32) * 512 + ((kbyte // 16) % 2) * 256 + (row % 16) * 16 + kbyte % 16
assert all(shuf[shuf_off(r, k)] == plain[r, k] for r in range(0, N, 7) for k in range(0, KB, 5))

# ---------------- DMA side: returns (chunk, list of 64 global byte offsets) per (wave, issue) ----------
def dma_today(src, half, tile, wm, wn, i):   # m8: plain row-major, swizzled 16 B K-chunk
    c = (wm * 2 + wn) * 4 + i
    offs = []
    for L in range(64):
        j = L // 8
        q = (L % 8) ^ (((j >> 1) & 1) << 1)
        row = half * HALF_N + (c // 2) * 16 + 2 * j + c % 2
        offs.append(row * KB + tile * BKB + 16 * q)
    return c, offs

def dma_a_keep_lds(src, half, tile, wm, wn, i):  # design (a): same LDS image, source = shuffled buffer
    c, _ = dma_today(src, half, tile, wm, wn, i)
    offs = []
    for L in range(64):
        j = L // 8
        q = (L % 8) ^ (((j >> 1) & 1) << 1)
        row = half * HALF_N + (c // 2) * 16 + 2 * j + c % 2
        offs.append(shuf_off(row, tile * BKB + 16 * q))
    return c, offs

def dma_b_pr6696(src, half, tile, wm, wn, i):   # design (b): chunk c = 2*nblk + khalf, 1 KB contiguous
    c = (wm * 2 + wn) * 4 + i                   # same chunk / imm assignment as today (make_layout_sb unchanged)
    nblk, khalf = c // 2, c % 2                 # = (wm*2+wn)*2 + i/2, i%2
    offs = [(half * HALF_N // 16 + nblk) * 16 * KB + tile * BKB * 16 + khalf * 1024 + 16 * L for L in range(64)]
    return c, offs

# ---------------- read side: LDS byte offset (within half) for lane, e, kk, wn --------------------------
def rd_today(L, e, kk, wn):
    r, g = L % 16, L // 16
    c = 4 * e + 2 * wn + r % 2
    return c * P + (r // 2) * 128 + kk * 64 + 16 * (g ^ (((r >> 2) & 1) << 1))
def rd_b(L, e, kk, wn):
    return (2 * (2 * e + wn) + kk) * P + 16 * L     # rb: (2e+wn)*2P + 16*lane ; kk -> + kk*P

def run(name, src, dma, rd, tile=3):
    sectors, lines = [], []
    ok = True
    for half in range(2):
        lds = np.full(16 * P, -1, dtype=np.int32)
        for wm in range(2):
            for wn in range(2):
                for i in range(4):
                    c, offs = dma(src, half, tile, wm, wn, i)
                    sectors.append(len({o // 64 for o in offs}))
                    lines.append(len({o // 128 for o in offs}))
                    for L, o in enumerate(offs):           # LDS-DMA: lane L -> M0 + imm + 16*L
                        lds[c * P + 16 * L: c * P + 16 * L + 16] = src[o:o + 16]
        for wn in range(2):
            for e in range(4):
                for kk in range(2):
                    for L in range(64):
                        r, g = L % 16, L // 16
                        row = half * HALF_N + (2 * e + wn) * 16 + r
                        want = plain[row, tile * BKB + kk * 64 + 16 * g: tile * BKB + kk * 64 + 16 * g + 16]
                        a = rd(L, e, kk, wn)
                        if not np.array_equal(lds[a:a + 16], want):
                            ok = False
    # bank model: ds_read_b128 serviced 8 lanes (128 B) per pass; 64 banks x 4 B
    worst = 0
    for wn in range(2):
        for e in range(4):
            for kk in range(2):
                for grp in range(8):
                    cnt = np.zeros(64, int)
                    for L in range(8 * grp, 8 * grp + 8):
                        a = rd(L, e, kk, wn)
                        for d in range(4):
                            cnt[((a + 4 * d) // 4) % 64] += 1
                    worst = max(worst, cnt.max())
    print(f"{name:34s} data_ok={ok}  64B-sectors/inst={sorted(set(sectors))}  128B-lines/inst={sorted(set(lines))}  "
          f"max lanes per bank per 8-lane pass={worst}")

src_plain = plain.reshape(-1)
run("today (plain B, m8 layout)", src_plain, dma_today, rd_today)
run("(a) shuffled B, keep m8 LDS layout", shuf, dma_a_keep_lds, rd_today)
run("(b) shuffled B, PR6696-style chunks", shuf, dma_b_pr6696, rd_b)

# print per-lane formula sample for (b)
c, offs = dma_b_pr6696(shuf, 0, 0, 1, 0, 3)
print("(b) wave(wm=1,wn=0) issue 3 -> chunk", c, " lane0..3 / 63 global offs:", offs[:4], offs[63])
c, offs = dma_a_keep_lds(shuf, 0, 0, 0, 0, 0)
print("(a) wave(0,0) issue 0 lane0..9 global offs:", offs[:10])
