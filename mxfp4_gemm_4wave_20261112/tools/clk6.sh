#!/bin/bash
# clk6.sh label cmd... : run cmd in background on GPU6, sample sclk/power 3x mid-run
label=$1; shift
HIP_VISIBLE_DEVICES=6 OMP_NUM_THREADS=16 "$@" > /tmp/clk_$label.log 2>&1 & pid=$!
for k in $(seq 1 60); do sleep 1; p=$(amd-smi metric -g 6 -p 2>/dev/null | grep -oE "SOCKET_POWER: [0-9]+" | grep -oE "[0-9]+$"); [ "${p:-0}" -gt 600 ] && break; done; sleep 3
out=""; for k in 1 2 3; do out="$out $(amd-smi metric -g 6 -c -p 2>/dev/null | grep -oE "SOCKET_POWER: [0-9]+|CLK: [0-9]+" | head -2 | grep -oE "[0-9]+" | paste -sd/)"; sleep 1; done
wait $pid; echo "$label power/clk:$out  $(grep -oE '[0-9.]+ us' /tmp/clk_$label.log | tail -1)"
