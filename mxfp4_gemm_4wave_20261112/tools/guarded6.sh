#!/bin/bash
# guarded6.sh <cmd...> : wait until GPU6 has been 0% busy for 20 s, run cmd, report GPU6 use before/after.
idle() { for i in 1 2 3 4 5 6 7 8; do u=$(rocm-smi -d 6 --showuse 2>/dev/null | grep -oE "GPU use \(%\): [0-9]+" | grep -oE "[0-9]+$"); [ "${u:-100}" -eq 0 ] || return 1; sleep 2.5; done; return 0; }
until idle; do sleep 20; done
echo "[guard] GPU6 idle at $(date -u +%T)"
"$@"
echo "[guard] GPU6 after: $(rocm-smi -d 6 --showuse 2>/dev/null | grep -oE 'GPU use \(%\): [0-9]+')"
