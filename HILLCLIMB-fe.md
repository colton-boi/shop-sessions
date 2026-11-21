# Hillclimb log: Track 1 Front-End / Parse (`impl-fe`)

Method: interleaved A/B runs (`tmp/abfe.py`), `taskset -c 0-3`, `SS_THREADS=4 NCPU=4 SS_STATS=1`, p20, page cache warm, output in `/dev/shm`.
Judged on per-stage `user+sys` CPU ms, `Mcyc`, `Minstr`, `Mbr`. Medians of 7 interleaved runs across 2 batches.

## Log

| # | Hypothesis | Before (parse medians) | After (parse medians) | Decision | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | **W2a-H4**: Direct L1d/L2 LUTs in `dims.rs` (`c_<N>` `Vec<u16>` <=65535, `p<N>` `Vec<u8>` <=262144, canonical validation + SWAR dec, KeyMap fallback) | B1: 8514.3 Mcyc, 62.1 Mbr, 3369.5 cpu ms; B2: 9458.9 Mcyc, 63.0 Mbr, 3605.0 cpu ms | B1: 8254.5 Mcyc, 60.6 Mbr, 3152.4 cpu ms (-3.05% Mcyc, -6.44% cpu); B2: 8724.3 Mcyc, 60.9 Mbr, 3445.0 cpu ms (-7.77% Mcyc, -4.44% cpu) | **accepted** | Eliminates KeyMap hash probing & 2 MiB product map misses for canonical `c_<N>` and `p<N>` IDs |
