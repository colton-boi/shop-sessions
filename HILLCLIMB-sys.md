# Hillclimb log: system and orchestration stream

Method: `taskset -c 8-11`, `NCPU=4` (4 threads), page cache warm, output on tmpfs, before/after interleaved
(`ab.py` in my scratch dir). Each run records wall, wall minus the program's own `ss done total` (start plus exit
cost, a static no-op measures 1.0 ms), user and sys CPU, minor faults, and `/proc/vmstat` deltas for
`thp_fault_alloc`, `thp_fault_fallback` and `compact_stall`.

Box state matters: memory here is fragmented (no free order-9 blocks, 14 GB of shmem in /tmp), so `MADV_HUGEPAGE`
regions hit direct compaction and sys time swings between runs (populate 300 to 1500 ms sys). Kernel-side changes
are therefore also judged under `SS_NOHUGE=1`, which is steady. The box is also oversubscribed by other agents'
unpinned work, so wall times are noisy; out-in and per-stage CPU are the primary signals.

| # | Hypothesis | Data | Before | After | Decision | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Exit teardown of ~1 GB of arenas and the links table is serial and on the clock; free them with parallel `MADV_DONTNEED` while thread 0 writes campaign_daily | p20 x7 THP; x5 NOHUGE | THP: campaigns 4 + out-in 44 = 48 ms. NOHUGE: 4 + 83 = 87 ms | THP: 16 + 9 = 25 ms. NOHUGE: 28 + 7 = 35 ms | accepted | out-in drops to ~7 ms (what is left is 72 MB of heap); total wall −23 to −52 ms at 4 threads, more at 8 since the release splits 7 ways |
