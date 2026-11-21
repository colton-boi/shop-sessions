# Hillclimb log

Method: before/after binaries run interleaved, `taskset -c 0-7`, page cache warm, output on tmpfs, wall measured
around the process (`tmp/ab.py`). Accept only a median gain above about 2% that repeats on a second batch.
All times in ms. Box: Ryzen 5 7535HS (Zen 3), shared desktop, so run-to-run noise is about 3 to 5%.

| # | Hypothesis | Data | Before median | After median | Decision | Why |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Always-on stderr diagnostics (host line about 0.12 ms) | p20 x7 | 704 | 708 | kept | required; difference is noise |
