#!/usr/bin/env python3
"""Analyze an aiter GEMM config PR (tuned/untuned CSV update) against the 4 standard checkpoints.

Checkpoints:
  1. No hipblaslt/blaslt (and no pure-torch) among NEW tuned rows.
  2. M-value discipline: prefill rows (M>256) must be powers of two; decode rows (M<=256)
     may be exact CUDA-graph sizes; flag shape-count explosion.
  3. Data integrity of new rows: recompute tflops/bw from us; splitK column must match the
     kernel-name ksN; list rows with err_ratio>0 (split-K accuracy flag).
  4. tuned <-> untuned sync: every new tuned (M,N,K) must exist in the PR untuned file;
     no duplicate rows in either file; new_untuned == new_tuned_shapes - pre_existing.

Also reports: rows on main but not in the PR branch (rebase currency), and same-(M,N,K)
rows that differ only by gfx (OK) vs same-gfx key collisions (FAIL).

Exit code 0 when every checkpoint passes, 1 otherwise.
"""
import argparse
import csv
import sys
from collections import Counter

TUNED_HEADER = "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw"
UNTUNED_HEADER = "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle"


def load_csv(path, header_prefix, name):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith(header_prefix):
            start = i
            break
    if start is None:
        sys.exit(f"ERROR: header '{header_prefix}' not found in {path} ({name})")
    rdr = csv.reader(lines[start:])
    header = next(rdr)
    rows = [r for r in rdr if r and any(c.strip() for c in r)]
    return header, rows


def is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pr-tuned", required=True, help="PR-branch tuned CSV (plain file or fetch-spill txt)")
    ap.add_argument("--main-tuned", required=True, help="main-branch tuned CSV")
    ap.add_argument("--pr-untuned", required=True, help="PR-branch untuned CSV")
    ap.add_argument("--main-untuned", required=True, help="main-branch untuned CSV")
    ap.add_argument("--perf-tol", type=float, default=0.02, help="relative tolerance for tflops/bw recompute (default 0.02)")
    args = ap.parse_args()

    _, pr_tuned = load_csv(args.pr_tuned, "gfx,", "pr-tuned")
    _, main_tuned = load_csv(args.main_tuned, "gfx,", "main-tuned")
    _, pr_untuned = load_csv(args.pr_untuned, "M,N,K", "pr-untuned")
    _, main_untuned = load_csv(args.main_untuned, "M,N,K", "main-untuned")

    key = lambda r: tuple(r[:10])                      # gfx..bpreshuffle
    mnk = lambda r: (r[2], r[3], r[4])                 # M,N,K
    ukey = lambda r: tuple(r[:8])

    main_tuned_keys = set(map(key, main_tuned))
    pr_tuned_keys = set(map(key, pr_tuned))
    new_rows = [r for r in pr_tuned if key(r) not in main_tuned_keys]
    gone_rows = [r for r in main_tuned if key(r) not in pr_tuned_keys]

    fails = []

    print(f"main_tuned rows: {len(main_tuned)} | pr_tuned rows: {len(pr_tuned)}")
    print(f"main_untuned rows: {len(main_untuned)} | pr_untuned rows: {len(pr_untuned)}")
    print(f"NEW tuned rows (full 10-col key): {len(new_rows)} | rows on main missing from PR branch: {len(gone_rows)}")
    if gone_rows:
        print("  [rebase note] main moved after the PR branched; these rows are not in the PR branch:")
        for r in gone_rows:
            print(f"    {','.join(r[:10])} | lib={r[10]}")

    # ---------------- Checkpoint 1: blaslt ----------------
    print("\n=== Checkpoint 1: no blaslt among NEW rows ===")
    lib_hist = Counter(r[10] for r in new_rows)
    print(f"libtype histogram of new rows: {dict(lib_hist)}")
    blas_new = [r for r in new_rows if "blas" in r[10].lower()]
    torch_new = [r for r in new_rows if r[10].lower() == "torch"]
    blas_all = [r for r in pr_tuned if "blas" in r[10].lower()]
    print(f"blas among NEW rows: {len(blas_new)} | pure-torch among NEW rows: {len(torch_new)}")
    print(f"pre-existing blas rows in file (not introduced by this PR): {len(blas_all)} "
          f"-> shapes {sorted(set(mnk(r) for r in blas_all))}")
    if blas_new:
        fails.append("C1: new rows use blaslt")
        for r in blas_new:
            print(f"  FAIL blas new row: {','.join(r[:11])}")
    if torch_new:
        fails.append("C1: new rows select the torch path (same as main default; should be dropped)")
        for r in torch_new:
            print(f"  FAIL torch new row: {','.join(r[:11])}")
    if not blas_new and not torch_new:
        print("PASS")

    # ---------------- Checkpoint 2: M discipline ----------------
    print("\n=== Checkpoint 2: M-value discipline ===")
    ms = sorted(set(int(r[2]) for r in new_rows))
    nonpow2 = [m for m in ms if not is_pow2(m)]
    bad_prefill = [m for m in nonpow2 if m > 256]
    decode_exact = [m for m in nonpow2 if m <= 256]
    print(f"distinct M in new rows: {ms}")
    print(f"non-pow2 M>256 (prefill odd-M, must be empty): {bad_prefill}")
    print(f"non-pow2 M<=256 (decode exact, OK): {decode_exact}")
    nk_groups = {}
    for r in new_rows:
        nk_groups.setdefault((r[3], r[4]), []).append(int(r[2]))
    for nk, mlist in sorted(nk_groups.items()):
        mlist = sorted(mlist)
        odd = [m for m in mlist if not is_pow2(m) and m > 256]
        flag = f"  <-- FAIL odd prefill M {odd}" if odd else ""
        print(f"  N={nk[0]:>6} K={nk[1]:>6}: M={mlist}{flag}")
    if len(new_rows) > 100:
        fails.append(f"C2: shape explosion ({len(new_rows)} new rows)")
        print(f"  FAIL: {len(new_rows)} new rows is a shape explosion; use pow2 buckets + getPaddedM")
    if bad_prefill:
        fails.append(f"C2: odd prefill M {bad_prefill}")
    if not bad_prefill and len(new_rows) <= 100:
        print("PASS")

    # ---------------- Checkpoint 3: data integrity ----------------
    print("\n=== Checkpoint 3: data integrity of NEW rows ===")
    perf_bad, sk_bad, err_rows = [], [], []
    for r in new_rows:
        M, N, K = int(r[2]), int(r[3]), int(r[4])
        try:
            us = float(r[13])
            tf_csv, bw_csv = float(r[16]), float(r[17])
        except (ValueError, IndexError):
            perf_bad.append((mnk(r), "unparseable us/tflops/bw", r[13:18]))
            continue
        tf = 2 * M * N * K / (us * 1e-6) / 1e12
        bw = 2 * (M * K + N * K + M * N) / (us * 1e-6) / 1e9
        if abs(tf - tf_csv) / max(tf_csv, 1e-9) > args.perf_tol or abs(bw - bw_csv) / max(bw_csv, 1e-9) > args.perf_tol:
            perf_bad.append((mnk(r), r[10], us, tf_csv, round(tf, 3), bw_csv, round(bw, 3)))
        if r[10] == "flydsl" and f"_ks{r[12]}_" not in r[14]:
            sk_bad.append((mnk(r), f"splitK={r[12]}", r[14]))
        if float(r[15] or 0) > 0:
            err_rows.append((mnk(r), r[10], r[12], r[15]))
    print(f"tflops/bw recompute mismatches (>{args.perf_tol:.0%}): {len(perf_bad)}")
    for b in perf_bad:
        print("  FAIL", b)
    print(f"splitK column vs kernel-name ks mismatch: {len(sk_bad)}")
    for b in sk_bad:
        print("  FAIL", b)
    print(f"rows with err_ratio>0 (split-K accuracy flag, needs e2e validation): {len(err_rows)}")
    for e in err_rows:
        print("  info", e)
    if perf_bad or sk_bad:
        fails.append("C3: data integrity")
    else:
        print("PASS (author A/B evidence still needs human scrutiny: total must beat main, no point below ~0.93x)")

    # ---------------- Checkpoint 4: tuned <-> untuned sync ----------------
    print("\n=== Checkpoint 4: tuned <-> untuned sync ===")
    pr_untuned_shapes = set((r[0], r[1], r[2]) for r in pr_untuned)
    main_untuned_shapes = set((r[0], r[1], r[2]) for r in main_untuned)
    new_tuned_shapes = set(mnk(r) for r in new_rows)
    missing = new_tuned_shapes - pr_untuned_shapes
    pre_existing = new_tuned_shapes & main_untuned_shapes
    new_untuned_shapes = pr_untuned_shapes - main_untuned_shapes
    sync_ok = new_untuned_shapes == (new_tuned_shapes - main_untuned_shapes)
    dup_t = [s for s, c in Counter(map(key, pr_tuned)).items() if c > 1]
    dup_u = [s for s, c in Counter(map(ukey, pr_untuned)).items() if c > 1]
    print(f"new tuned shapes: {len(new_tuned_shapes)} | missing from PR untuned: {len(missing)} {sorted(missing) if missing else ''}")
    print(f"new untuned shapes: {len(new_untuned_shapes)} | pre-existing in main untuned: {len(pre_existing)} {sorted(pre_existing) if pre_existing else ''}")
    print(f"new_untuned == new_tuned - pre_existing: {sync_ok}")
    print(f"duplicate rows inside PR tuned: {len(dup_t)} | inside PR untuned: {len(dup_u)}")
    # same-(M,N,K) collisions vs main tuned: same gfx = FAIL, different gfx = OK
    main_by_mnk = {}
    for r in main_tuned:
        main_by_mnk.setdefault(mnk(r), []).append(r)
    same_gfx_collide, cross_gfx = [], 0
    for r in new_rows:
        for mr in main_by_mnk.get(mnk(r), []):
            if mr[0] == r[0]:
                same_gfx_collide.append((mnk(r), r[0]))
            else:
                cross_gfx += 1
    print(f"same-(M,N,K) rows on main with different gfx (OK): {cross_gfx} | same-gfx key collisions (FAIL): {len(same_gfx_collide)}")
    for c in same_gfx_collide:
        print("  FAIL", c)
    c4_bad = missing or not sync_ok or dup_t or dup_u or same_gfx_collide
    if c4_bad:
        fails.append("C4: tuned/untuned sync")
    else:
        print("PASS")

    print("\n=== RESULT ===")
    if fails:
        for f in fails:
            print("FAIL:", f)
        sys.exit(1)
    print("ALL CHECKPOINTS PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
