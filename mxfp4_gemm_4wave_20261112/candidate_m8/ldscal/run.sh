#!/bin/bash
# usage: run.sh patfile counter
d=/tmp/w2/mb/prof_$(basename $1)_$2; rm -rf $d
HIP_VISIBLE_DEVICES=4 rocprofv3 --output-format csv --pmc $2 -d $d -o r -- /tmp/w2/mb/mb $1 >/dev/null 2>&1
f=$(find $d -name 'r_counter_collection.csv' | head -1)
python3 - "$f" "$1.names" <<'PY'
import csv, sys, collections
s = collections.OrderedDict(); lds = {}
for r in csv.DictReader(open(sys.argv[1])):
    if not r['Kernel_Name'].startswith('k('): continue
    i = int(r['Dispatch_Id']); s[i] = s.get(i, 0) + float(r['Counter_Value']); lds[i] = r.get('LDS_Block_Size')
names = [l.strip() for l in open(sys.argv[2])]
for (i, v), n in zip(sorted(s.items()), names):
    print('%-22s %10.0f  per-inst %.2f  lds=%s' % (n, v, v / (256 * 4 * 1000), lds[i]))
PY
