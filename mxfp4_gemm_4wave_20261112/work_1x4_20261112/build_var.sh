#!/bin/bash
# build_var.sh name "-DFOO -DBAR" : build this dir's sources into /tmp/exp_1x4/<name> with extra defines
n=$1; x=$2; D=$(cd $(dirname $0) && pwd); o=/tmp/exp_1x4/$n
rm -rf $o && mkdir -p $o && cp $D/*.cc $D/*.hpp $D/Makefile $o/ && cd $o && make EXTRA="$x" >build.log 2>&1 && echo "$n ok" || { echo "$n FAIL"; tail -20 build.log; }
