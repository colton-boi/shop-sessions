# Hillclimb log: Track 2 (links, dedup, sessions, Parquet), swarm round 2

Method: `tmp/ab.py`: binaries interleaved, `taskset -c 4-7`, `SS_THREADS=4 NCPU=4 SS_STATS=1`, output on
tmpfs, 7 runs per batch, medians of per-stage user-only Mcyc (perf counters), user+sys ms, Mbr.
Accept: median stage gain > 2% repeated on a second batch, plus `verify/gate.sh` GATE PASS.
Box load during runs was 10 to 38 with other agents on cores 4-7, so wall is unreliable; Mcyc is the judge.
p60 runs (3 each) are a linearity and sanity check only: at load 38 even unchanged stages moved +/-10%.

Base for #1: `fa48eac` (swarm-integ).

| # | Hypothesis | Before (p20 median, batch 1 / batch 2) | After | Decision | Why |
| --- | --- | --- | --- | --- | --- |
| 1 | W2a-H3 + W5-H1 + W4-H7: 64 B SoA 4-slot LinkTable buckets (lanes filled 0..3 in order, so `keys[3]==0` proves no spill), branch-free AVX2 4-key probe in `person_hashed` (read-only during parse, 0 CAS: C4), single-pass bit-56 `compress_and_resolve` replacing compress+resolve (one barrier fewer), `SWEEP=16K` (release kept at 64K-slot / 1 MiB chunks), `mix` hashes cached in `link_all`, read-before-CAS on roots, self-links skipped in `parse_links`, sizing `pairs*5/2` (was `*3`) | links join Mcyc 431.5 / 462.4, Mbr 6.8; parse Mcyc 8558 / 8538, Mbr 62.1 | links join 295.9 / 317.5 (-31.4% / -31.3%), Mbr 4.2; parse 8325 / 8291 (-2.7% / -2.9%), Mbr 37.9 (-24 M) | kept | Both stages over 2% on both batches. p60 (3 runs): links join Mcyc 1367 -> 977 (-28.5%), parse Mbr 193 -> 117. Gate PASS. Parse Minstr +350 M (vector probe) but 24 M fewer mispredicts. Unit tests: 87.5%-full 2-bucket table under 4 threads, 400-chain + 5k hub + >9 B fallback ID |
| 2 | W4-H6: Dedup pass 2 writes every record index into a reserved candidate buffer and advances by the `twice` bit (no branch on the ~14% random hit); pass 3 prefetches the DRec 16 candidates ahead | dedup Mcyc 856.7 / 874.0 (h1), Mbr 7.55 | 723.2 / 717.6 (-15.6% / -17.9%), Mbr 3.27 | kept | Both batches. p60: dedup Mcyc 2946 -> 2470 (-16%), Mbr 19.0 -> 7.2. Gate PASS, fixtures 1/4/8 threads, NOHUGE, 192/4097 sweeps |
| 3 | W4-H3: SoA `sess` u32 table apart from cold revenue/purchasing cells, `const HAS_PURCH` person specialization, monotone `next_day_at` (no div_euclid per session), PDEP `tok_bytes_fast`, `assists` break past the last session start, `credit` cache keyed on `next_click` | sessions Mcyc 2669 / 2755 / 2704 / 2768 (h6) | 2624 / 2726 / 2741 / 2681 (-1.7 / -1.0 / +1.4 / -3.1%); variant without the SoA split: 2658 / 2693 (-1.7 / -2.7%) | rejected | Never > 2% on two batches. Minstr -86 M (-2.1%) every batch but Mbr +0.4 M (the `revenue > 0` branch). rdtsc split of the stage: gather events + radix about 40%, so the sessions gain lives in W4-H1/W4-H2. The credit cache and PDEP `tok_bytes` are carried into the Tier-2 rewrite |

Notes:
- No shared CSV fuzz harness exists in the repo (scripts/, reference/, verify/); checks run per commit are
  `verify/gate.sh`, fixtures at SS_THREADS 1/4/8, SS_NOHUGE=1 fixtures + seed 1, and SS_READ_BUF/SS_EXTENT
  192/4097 fixtures + seed 1 (`tmp/extra.sh`).
- Track 4: `links.rs` @ 03c184c is byte-identical (sha256 e741c200) to the blob A4a proved PASS. Note left in
  `verify/inbox/post-links-committed.md`.
- Cross-ownership edits in #1 (for the merge):
  - `src/main.rs` `parse_links` (Track 1): tokens bound to `a`, `b`; pair pushed only if `a != b` (self-link skip).
  - `src/main.rs` orchestration (Track 3): links-join `SWEEP` 1<<16 -> 1<<14; the two `sweep()` calls
    (`compress`, `resolve`) replaced by one `compress_and_resolve`; the phase-7 release loop now uses its own
    `REL_SWEEP = 1 << 16` / `rel_sweeps` instead of the join's `SWEEP` / `sweeps`, so release chunks stay 1 MiB.
