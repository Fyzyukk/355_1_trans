#!/bin/bash
# usage: run.sh binary threads [steps] [launches]
# NOTE: HIP_VISIBLE_DEVICES=3 is PCI 0000:15:00.0 = rocm-smi GPU[1] on this host
bin=$1; th=$2; steps=${3:-1024}; L=${4:-15000}
HIP_VISIBLE_DEVICES=3 $bin $th $steps $L 300 > /tmp/f8w/out.$$ 2>&1 &
pid=$!
sleep 2.5
clk=""; pw=""
for i in 1 2 3; do
  o=$(rocm-smi -d 1 --showclocks --showpower 2>/dev/null)
  clk="$clk $(echo "$o" | grep -oP 'sclk clock level: \S+ \(\K[0-9]+')"
  pw="$pw $(echo "$o" | grep -oP 'Power \(W\): \K[0-9.]+' | head -1)"
  sleep 0.5
done
wait $pid
echo "$(cat /tmp/f8w/out.$$) | sclk:$clk | W:$pw"
rm -f /tmp/f8w/out.$$
