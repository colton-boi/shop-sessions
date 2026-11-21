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
