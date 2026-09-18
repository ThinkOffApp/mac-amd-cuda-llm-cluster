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

**1. CPU staging costs 47% of decode throughput on this machine.** 77.87 against
41.02 tok/s, same model, same device, same math, differing only in whether
row-parallel outputs round-trip through CPU.

That sizes the bug @codexmb caught in the first version of this harness: the
solo baseline was staged, so it would have been ~47% too slow and would have
manufactured a TP "speedup" of nearly 2x out of nothing.

It also frames the actual problem. **TP must stage — the transport moves CPU
tensors — so on the Mini side it pays this tax before any network cost.** For
two-host TP to beat the best single host it has to overcome roughly a 47%
handicap using a 2x split of the compute. The network may not be the thing that
decides this.

Measured on one machine, one model, at one prompt length. The ROCm side is
unmeasured and the TP figure does not exist.

**2. Run-to-run spread is 5.6% at p128.** Any eventual difference smaller than
that is inside the noise of a single machine, and 5 runs is not many. Worth
fixing in mind before the comparison arrives rather than after.
