# Where the time goes — corrected, 2026-09-18

**The first version of this file was wrong. It is kept corrected, not deleted,
because the error is instructive.**

@codexmb found two measurement bugs: the stage counters accumulated during
prefill, the correctness gate and the warmups, while the denominator counted
only decode forwards — 189 forwards over 63 tokens. And "compute" timed only the
two row-parallel matmuls, omitting embedding, layer norms, most of attention,
residuals and the LM head.

The profiler now switches on **only for the decode steps of a recorded run**, and
measures against a directly-timed `total_forward` with an explicit `unaccounted`
bucket, so a missing term cannot hide.

## Corrected: Mini alone, Metal, staged, decode only

```
  compute (row-parallel matmuls)   8.797 ms   28.2%
  device -> CPU                    4.389 ms   14.1%
  CPU -> device                    4.137 ms   13.3%
  lm_head matmul                   1.788 ms    5.7%
  logits -> CPU                    0.219 ms    0.7%
  embed                            0.395 ms    1.3%
  UNACCOUNTED                     11.456 ms   36.7%
  total forward                   31.191 ms
```

Staging is **27%**, not the 48% first reported. The largest single bucket is the
arithmetic that was never instrumented, plus the sync overhead profiling adds.

## The finding worth keeping: cost is per-transfer, not per-byte

```
  1 copy of 201,028 B   0.219 ms       918 MB/s effective
 24 copies of 3,072 B   0.183 ms each   17 MB/s effective
```

**The LM head copy moves 65x more bytes for 1.2x the time.** Device-to-host cost
here is dominated by fixed per-call overhead, not bandwidth.

**Consequence for optimisation: compressing the payload buys almost nothing.**
A narrower wire dtype halves bytes that were not the problem. **Fewer, larger
transfers is the lever** — fusing or batching the per-block collectives.

## How to read it

`--profile` takes a GPU sync around every stage, so absolute totals are inflated
(31.2 ms/token profiled against ~24.4 unprofiled). Read shares. Profiled runs
return no `timings` at all, so a profiled figure cannot be quoted as throughput.

## Bounds

One machine, Metal, no network, GPT-2 124M, staged control rather than real TP.
The real TP breakdown additionally pays wire time, synchronisation and waiting
for the peer, and has not been measured.
