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
