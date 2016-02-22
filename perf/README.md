# Profiling

Collenchyma comes with scripts to help with profiling performance problems.

Run [perf](http://www.brendangregg.com/perf.html) on one of the benchmark test:

```sh
# compile latest version of benchmarks with DWARF information
cargo rustc --bench rblas_overhead -- -g
# benchmark binary is at target/debug/shared_memory-54e69b24ec0c2d04
# benchmark is called bench_256_sync_1mb_native_cuda
sudo ./perf/run_perf.sh target/debug/shared_memory-54e69b24ec0c2d04 bench_256_sync_1mb_native_cuda # perf needs sudo
```
