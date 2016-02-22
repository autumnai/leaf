#! /bin/bash
set -e
if [ $# -lt 2 ]
  then
    echo "No binary name or benchmark name supplied"
    exit 1
fi
binaryname=$1
benchname=$2
mkdir -p target/perf
perf record -a -g --output target/perf/${benchname}.data ${binaryname} --bench ${benchname}
perf script -f -i target/perf/${benchname}.data > target/perf/${benchname}.scripted
stackcollapse-perf target/perf/${benchname}.scripted | grep ${benchname} > target/perf/${benchname}.folded
flamegraph target/perf/${benchname}.folded > target/perf/${benchname}.svg
