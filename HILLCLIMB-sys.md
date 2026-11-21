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
| 2 | 2 MB-aligned input mappings let the kernel PMD-map large page-cache folios, cutting input faults | p20 x5 NOHUGE | kflt 288k, `thp_file_mapped` +4/run | kflt 288k, +7/run | rejected | ext4 here maps no file PMDs for a private mapping; no change in faults or parse sys |
| 3 | Measure: how much parse sys is input mapping? (`MADV_POPULATE_READ` all inputs in the populate phase, not kept) | p20 x3 NOHUGE | populate sys 347, parse sys 283 | populate 576, parse 131 | info | input mapping costs about 150 to 230 ms sys of the ~610; read() copies cost more (C: 13 to 19% of CPU), so mmap stays |
| 4 | Dedup table probes miss cache; prefetch slots 16 to 32 records ahead, and try load factors 1/4, 1/2, 1/8 | p20 x5 SS_STATS Mcyc | 848 Mcyc (base) | 856 to 1122 Mcyc | rejected | the 256 to 512 KB per-partition table is cache-resident already; dedup is bound by streaming the 345 MB DRec arena back from DRAM |
| 5 | Links join is miss-bound (IPC 0.22); prefetch both slots of the pair 8 ahead, and in both sweeps the parent slot 16 ahead | p20 x5, x7 SS_STATS | links join stage 702, 707 Mcyc; user 191, 197 ms | 478, 484 Mcyc; user 143, 139 ms | accepted | −32% cycles on a barrier stage; join and sweep phases now cost about the same |
| 6 | Link table capacity (pairs × 1.5, 3 (current), 6, 12 slots → 2M, 4M, 8M, 16M) trades sweep cost for parse lookup probes | p20 x5 SS_STATS | 4M: links 489, parse 9389 Mcyc | 2M: parse +1000 Mcyc. 8M: links +106, parse −330. 16M: links +324, parse −333 | not taken | 8M nets about −220 Mcyc (~1.5% of total) and the parse number moves with the parse stream's work; a lead for after the merge |
