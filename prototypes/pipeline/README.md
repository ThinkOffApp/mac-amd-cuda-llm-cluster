# Micro-batched pipelining across Mac↔Linux: it loses, and the cause is kernel launches

Provenance
- 2026-09-18, @claudemm, Mac mini M4 (Metal) rank 0 ↔ Bosgame M5 (ROCm) rank 1, home gigabit.
- Real GPT-2 weights, FP32, pipeline parallel: each rank owns whole layers, only the
  hidden state crosses the wire. Chunks are disjoint SEQUENCES, so they are genuinely
  independent and the chunked run computes the same arithmetic in a different order.
- Code: `pipeline.py` (prototype), `frame_cost.py` (transport in isolation), both here.

## Result: more micro-batches, less throughput — at both model sizes

| chunks | GPT-2 124M (6/6 layers) | GPT-2-large 774M (18/18) |
|-------:|------------------------:|-------------------------:|
| 1 (alternating) | 245.2 tok/s | 137.0 tok/s |
| 2 | 180.6 | 99.7 |
| 4 | 120.5 | 39.2 |

**Correctness gate:** identical output-token SHA at every chunk count, both models, and
identical bytes on the wire. Same payload, more frames.

## It is NOT the network

`frame_cost.py` replays the exact frame pattern with no model at all, across the same link:

| chunks | frames/step | median step |
|-------:|------------:|------------:|
| 1 | 2 | 0.758 ms |
| 2 | 4 | 0.672 ms |
| 4 | 8 | 0.685 ms |

**Four times the frames costs nothing measurable** — the sends and receives overlap exactly
as pipelining intends. Transport is ~1% of a 58 ms step. My earlier prediction that the cost
was "per round trip, so micro-batching multiplies it" was wrong, and this is what refuted it.

## It IS kernel-launch overhead

Stage-0 compute per call, gpt2-large, as the batch per call shrinks:

| batch per call | ms per call |
|---------------:|------------:|
| 8 | 18.39 |
| 4 | 16.99 |
| 2 | 14.07 |

**Quartering the work saves 23% of the time.** Each call pays 18 layers of launch overhead
whether 8 sequences ride along or 2, so 4 chunks cost 4 × 14.07 = 56 ms where 1 chunk cost
18.4 — which accounts for the slowdown on its own. Chunking divides the very thing that
amortises the launches.

This also retires the "~25 ms coordination tax" quoted earlier in the day: it was never
coordination, it is per-step launch overhead on the Metal stage, consistent with the separate
finding that a Metal GPU round trip is ~18x CUDA's.

## The generalisable test, for any model or runtime

Measure **ms-per-call against batch-per-call**:

- **flat** → launch-bound → pipelining loses, whatever the architecture
- **scales with batch** → work-bound → pipelining can win

GPT-2 at 124M and 774M are both flat. A model doing far more work per layer might not be —
and this curve answers it without implementing a second architecture.

Prediction on record: CUDA's launch overhead is ~18x smaller than Metal's, so a CUDA↔CUDA
pair should sit much closer to work-bound. If vLLM pipeline parallel wins on the two Sparks
and loses here, launch cost is the reason rather than the link.

## WITHDRAWN: "launch-bound, therefore pipelining loses for any model"

Do not cite the mechanism above. The same ms-per-step-vs-batch curve on **Qwen3.8-27B**,
a model actually in service, is NOT flat:

| batch | step time | work vs B=1 | time vs B=1 |
|------:|----------:|------------:|------------:|
| 1 | 84.2 ms | 1x | 1.0x |
| 4 | 118.4 | 4x | 1.4x |
| 32 | 451.8 | 32x | 5.4x |

32x the work costs 5.4x the time — **partly work-bound**, which is the regime where
pipelining is not hopeless. GPT-2 124M and 774M were nearly flat and the conclusion was
generalised from them. It should not have been.

@codexmb also pointed out that a flat stage-time curve does not by itself separate
kernel-launch overhead from weight-bandwidth limits, so "launch-bound" was never proven
even for GPT-2 — two mechanisms fit the same curve.

**What still stands:** the throughput numbers themselves, the identical-token correctness
gate across chunk counts, and the transport measurement showing the network is ~1% of a
step and flat in the number of frames. What does not stand is any claim about models other
than the ones measured here.

## Limits, stated
- Results describe GPT-2 124M and 774M ONLY. Benchmark on weights you serve.
- One prototype, one transport, FP32, no continuous batching.
- Layer split was 6/6 and 18/18 — balanced by LAYER, not by machine speed.
- Prefill is deliberately never chunked; only decode is under test.
