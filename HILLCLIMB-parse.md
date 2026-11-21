# Hillclimb log: parse workstream

Method: before/after binaries interleaved (`ab.sh`), `taskset -c 0-3`, 4 threads, p20, page cache warm.
Judged on the parse stage line of the run log (`SS_STATS=1`): Mcyc, Minstr and Mbr are the stable
signals, user ms swings about 15% on this shared box. Medians of 7 unless noted.

## Where parse time goes (base 9f4236e)

Knockouts on an instrumented build (env bitmask that skips one piece per run; correctness ignored):

| Removed | Parse Mcyc | Minstr | Mbr | Saved Mcyc |
| --- | --- | --- | --- | --- |
| nothing | 9440 | 16380 | 65.4 | |
| all row work (index + scan only) | 2580 | 5280 | 3.5 | 6860 |
| scatter pushes | 7740 | 14290 | 53.0 | 1700 |
| person lookup | 8250 | 15650 | 41.2 | 1190 |
| event `ts` parse | 8600 | 14980 | 51.6 | 840 |
| event and user `tok` | 8520 | 14520 | 53.9 | 920 |
| `ingest_ts` parse | 9050 | 15390 | 65.6 | 390 |

The index alone is about 630 user ms at 4 threads, so 800 ms needs the scanner cut too.

## Log

| # | Hypothesis | Before | After | Decision | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | Branch-free first probe in `LinkTable::person` (hit or empty selected by cmov) | 9260 Mcyc, 66.0 Mbr | 9233 Mcyc, 61.9 Mbr | rejected | only 0.3%; 20% of first probes collide, so the slow path still mispredicts |
| 2 | Regular stores instead of non-temporal in `Scatter::push` | 9445 Mcyc | 10077 Mcyc | rejected | +6.7% cycles, +20% LLC misses |
| 3 | 512 or 256 partitions (`SS_DPARTS`/`SS_PPARTS`) instead of 1024 | 9248 Mcyc | 9227 / 9081 | rejected | under 2%; staging misses are not the scatter cost |
| 4 | Link table at 6x pairs instead of 3x | 9153 Mcyc, 63.1 Mbr | 9013, 58.2 | not taken | 1.5%, outside my files; lead for the links owner |
| 5 | Branch-free `ids::pack`: bzhi mask, unconditional ninth byte, one combined bad test | 9470 / 9352 Mcyc, 15803 / 15720 Minstr | 9340 / 9042, 15352 / 15181 | **accepted** | -2.9 to -3.4% instructions, -1.4 to -3.3% cycles over two batches |
| 6 | Port B's AVX2 F1/F2 parser with a month table instead of the date cache, computing ISO and F3 both and selecting | 9282 Mcyc, 15182 Minstr | 10004, 16914 | rejected | the SIMD ISO path is about 50 instructions, more than the cached scalar one; Mbr fell only 4 M |
| 7 | Same ISO parser with a branch on length 20 or 25 | 9040 Mcyc, 15183 Minstr | 9387, 15399 | rejected | still more instructions than the date cache, and more LLC misses |
| 8 | Hash the user once for prefetch and lookup (`LinkTable::person_hashed`), and partition by a single multiply (`part_of_key`) instead of `part_of(mix(..))` | 9154 / 9080 Mcyc, 15182 Minstr | 8637 / 8651, 14706 | **accepted** | -5.6 and -4.7% cycles over two batches, -3.1% instructions; dedup and sessions unchanged (862 vs 870, 3922 vs 4023 Mcyc) |
