# Hillclimb Log: sys2 (Track 3 — Kernel / Memory / Orchestration / Build)

Environment: AMD Ryzen 5 7535HS (Zen 3, AVX2, no AVX-512), Linux 7.2, ext4 (`0xef53`), THP `always/madvise` (with `SS_NOHUGE=1` checks).
Benchmarking slice: `taskset -c 8-11`, `SS_THREADS=4 NCPU=4 SS_STATS=1`, `/dev/shm` output, >=7 interleaved A/B runs via `tmp/ab.py`.

| # | Hypothesis | Data | Before | After | Decision | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | W1-H2: `strip=true`, `-C relocation-model=static -C link-arg=-no-pie -C force-unwind-tables=no`, lazy `sys::prof::SAMPLES` (`AtomicPtr`, alloc only if `SS_PROF` set) | ELF + Docker + empty x50 + p20 x7x2 | ELF `ET_DYN` 1.74 MB, 1948 `RELATIVE`, `.bss` 32.02 MiB; empty startup 1.415 ms / 137 minflt; p20 parse 14723 Minstr / 8656 Mcyc, sessions 4065 Minstr / 2657 Mcyc | ELF `ET_EXEC` 1.47 MB, 0 `RELATIVE`, `.bss` 23.7 KB; empty startup 1.419 ms (min 1.089 vs 1.112) / 131 minflt; p20 parse 14714 Minstr / 8622 Mcyc, sessions 4068 Minstr / 2707 Mcyc; Docker dev 50.2 ms MATCH | accepted (per Tier 1 spec) | Removes 1948 PIE relocs, 32 MiB `.bss` THP hazard, and 266 KB binary size (-6 startup minor faults, -8..9M parse instr); `sessions` +50 Mcyc from static address code layout shifts once downstream code changes, while Docker build/run and full gate pass 100%. |
