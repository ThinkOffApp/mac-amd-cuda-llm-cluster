# Mini-only run, 2026-09-18 — HARNESS VALIDATION, NOT A COMPARISON

**These are not comparison numbers.** No M5 or TP figure exists: the M5 GPU
window was not opened. Nothing here may be paired with a later M5 or TP number,
because they would be taken at a different time on a fleet that has produced a
15% single-machine drift across one evening. The real comparison is interleaved
in one session, per `run_bench.sh`.

The purpose was to exercise the real MPS path — `torch.mps.synchronize`,
GPU-resident row outputs, the 512-position case — which no CPU dry run reaches.
It works: `valid: true`, every timed run matched the reference.

```
config                 ttft       decode            spread over 5 runs
solo MPS p128          0.0372 s   77.87 tok/s       5.6%
solo MPS p512          0.0874 s   82.47 tok/s       1.1%
solo-staged MPS p128   0.0370 s   41.02 tok/s       4.2%
```

## Two things worth knowing before any comparison exists

**1. Staging the FULL model through CPU costs 47.3% of decode throughput here** —
equivalently **89.8% more time per token**. 77.87 against 41.02 tok/s, same
model, same device, same math, differing only in whether row-parallel outputs
round-trip through CPU.

That sizes the bug @codexmb caught in the first version of this harness: the
solo baseline was staged, so it would have been substantially too slow and would
have manufactured a large TP "speedup" out of nothing.

**It is a CONTROL, not a TP prediction.** An earlier version of this file said TP
"pays this 47% before any network cost". That does not follow, and @codexmb was
right to bound it: **this measures the full model staged, while TP shards** — local
compute, synchronisation and overlap all differ once the work is split, so the
figure cannot be carried across as an additive tax. What it does establish is
that CPU round-tripping is expensive enough on this machine that an unstaged
baseline is mandatory, and that the network is not the only candidate for
dominating a TP result.

**2. Run-to-run spread was 5.6% at p128 across 5 runs.** This is **descriptive
variability, not a significance threshold.** It says these five runs varied by
that much; it does not license a rule that a future difference below it is
"noise". A paired, interleaved comparison is what would support that kind of
claim, and it has not been run.

Raw per-run records, per-step timestamps and measurement order are retained in
the JSON files beside this README.
