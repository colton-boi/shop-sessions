# Hillclimb log: post-parse stream (dedup, sessions, Parquet)

Method: `tmp`-free A/B via scratchpad `ab.py`: before/after binaries interleaved, `taskset -c 4-7`, NCPU=4,
SS_STATS=1, output on tmpfs, 7 runs each, medians. Primary metric is the stage's user-only Mcyc/Minstr from
perf counters (stable under the shared box's load) and user CPU ms; wall is secondary.
Box load during runs: 8 to 30, so wall varies a lot.

Baseline profile (p20, SIGPROF samples plus rdtsc per sub-phase, shares of the sessions stage):
gather events (hash person to local id) 17%, gather clicks 4%, counting scatter 9%, per-person event sort 19%,
click prep (by_campaign build and sort, equal-ts fix) 10%, session loop incl. day rows 25%, assists 19%.

| # | Hypothesis | Before median | After median | Decision | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | One-pass assists over ts-sorted clicks with two monotone session pointers and a per-campaign `covered` mark (no by_campaign sort, no binary searches, no all-days loop); crediting scans back over equal-ts runs instead of rewriting campaigns; `assisted` in its own u32 table | sessions Mcyc 3827 / 3831, Minstr 5410, user 985 / 983 ms | Mcyc 3544 / 3506, Minstr 4820, user 945 / 894 ms | kept | -8% cycles, -11% instructions on both batches |
| 2 | Radix for persons with >= 512 events (whales) instead of pdqsort | Mcyc 3537 | 3409 | superseded by 4 | -3.6%; thresholds 256/2048 the same |
| 3 | Person hash: one multiply instead of `mix`, load factor 1/4 | Mcyc 3512, Minstr 4496 | 3483, 4270 | rejected | -5% instructions but under 1% cycles |
| 4 | One sort per partition: 8-byte keys (local person << 32, ts << 1, purchase bit), two 11-bit LSD passes on ts skipping its low 9 bits, then the person counting pass (the old scatter). Purchases' payloads ride in a side list sorted by the same key; clicks sort the same way as 16-byte (key, campaign, CRec index) and fetch click_id only on equal-ts ties. Fix-up sorts only runs sharing a 512 ms bucket that hold an inversion, so cost is O(n + sum r log r) | Mcyc 3812 / 3945, Minstr 4820, user 1081 / 1104 ms | Mcyc 2971 / 2982, Minstr 3980, user 835 / 849 ms | kept | -22 to -24% cycles both batches. Gate also passed with TS_SKIP=20 (17-minute buckets), which forces the fix-up path constantly |
| 5 | Prefetch the campaign-day cell at session start (the cell update was 9% of stage samples) | sessions Mcyc 2881 / 2680 / 2869 / 2986 | 2845 / 2606 / 2830 / 2867 | kept (small) | -1.2 to -4% over four batches, two lines |
| 6 | 16-byte internal tally cells (assisted already split out) | Mcyc 2680 | 2631 | rejected | not better than 5 alone |
| 7 | Radix digit/skip variants: (skip 13, 9-bit), (15, 8), (11, 10) vs (9, 11) | Mcyc 2692 | 2948 / 3019 / 2828 | rejected | more fix-up work than saved scatter misses |
| 8 | Dedup: decomposition showed read+mix = 200 M cycles, memset 60 M, the probe into the 512 KB table ~600 M (L2 misses + 9 M branch misses). Two bitmaps (seen, seen-again) at 8 bits per record, buckets kept from pass 1, only flagged records (~14%) reach the exact table. Moved `dedup` from main.rs into group.rs as `Dedup::partition` | dedup Mcyc 904 / 941, user 258 / 265 ms | 821 / 819, user 232 / 240 ms | kept | -9 / -13% cycles |
| 9 | Dedup variants: bitmaps at 32 bits per record; candidates copied as DRec; interleaved bitmaps; hash prefetch 16 ahead on the original table; 64 L1-sized sub-buckets (indices, and DRec copies) | 868 | 861 / 893 / 845-908 / 868 / 892 / 1025 | rejected | none beat 8; sub-bucket copies add a full write pass |
