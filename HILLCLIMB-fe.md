# Hillclimb log: Track 1 Front-End / Parse (`impl-fe`)

Method: interleaved A/B runs (`tmp/abfe.py`), `taskset -c 0-3`, `SS_THREADS=4 NCPU=4 SS_STATS=1`, p20, page cache warm, output in `/dev/shm`.
Judged on per-stage `user+sys` CPU ms, `Mcyc`, `Minstr`, `Mbr`. Medians of 7 interleaved runs across 2 batches.

## Log

| # | Hypothesis | Before (parse medians) | After (parse medians) | Decision | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | **W2a-H4**: Direct L1d/L2 LUTs in `dims.rs` (`c_<N>` `Vec<u16>` <=65535, `p<N>` `Vec<u8>` <=262144, canonical validation + SWAR dec, KeyMap fallback) | B1: 8514.3 Mcyc, 62.1 Mbr, 3369.5 cpu ms; B2: 9458.9 Mcyc, 63.0 Mbr, 3605.0 cpu ms | B1: 8254.5 Mcyc, 60.6 Mbr, 3152.4 cpu ms (-3.05% Mcyc, -6.44% cpu); B2: 8724.3 Mcyc, 60.9 Mbr, 3445.0 cpu ms (-7.77% Mcyc, -4.44% cpu) | **accepted** | Eliminates KeyMap hash probing & 2 MiB product map misses for canonical `c_<N>` and `p<N>` IDs |
| 2 | **W2b-H1+H5**: Pre-populated 128-slot `DateLut` (`ymdt=[b'\n';12]`, bit-0 parity guard), 19-uop SSE4.1 `parse_f1_rel`, `parse_ts_rel`, unsigned window/lateness collapse (`base_ms % 1000 == 0` guard) | B1: 8241.9 Mcyc, 14754.7 Minstr, 3758.4 cpu ms; B2: 8256.1 Mcyc, 14754.2 Minstr, 3820.3 cpu ms | B1: 7777.4 Mcyc, 13220.7 Minstr, 3575.8 cpu ms (-5.64% Mcyc, -10.40% Minstr, -4.86% cpu); B2: 7814.4 Mcyc, 13220.3 Minstr, 3538.1 cpu ms (-5.35% Mcyc, -10.40% Minstr, -7.39% cpu) | **accepted** | Replaces DateCache hash/scalar HH:MM:SS + 64-bit calendar mul with 19-uop SSE + 128-slot L1d DateLut and collapsed unsigned bounds |
