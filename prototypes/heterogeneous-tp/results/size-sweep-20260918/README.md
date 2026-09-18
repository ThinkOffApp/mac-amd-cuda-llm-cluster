# Model-size sweep and copy-cost sweep — 2026-09-18

Two measurements that between them close both payload-level optimisation levers.

## 1. Staging share barely moves with model size

Same architecture across the GPT-2 family, so only size changes. Mini alone,
Metal, staged, decode phase, 3 runs each.

```
  model          params   staging share of decode      runs
  gpt2             124M   median 29.0%   (30.8, 28.3, 29.0)
  gpt2-medium      355M   median 29.6%   (29.8, 29.6, 28.0)
  gpt2-large       774M   median 26.4%   (27.5, 25.9, 26.4)
```

**6.2x the parameters moved the staging share 2.6 points**, against a worst
within-model spread of 2.5. The 124M and 774M ranges do not overlap so the
direction is probably real, but medium is not below small and the magnitude is
small. **You cannot size your way out of this**, which contradicts the
hypothesis this file's author proposed.

Correctness held at every size: medium and large both generate token-identical
output to HuggingFace across the pair.

**gpt2-xl is excluded: it has 25 attention heads.** An odd head count cannot be
split evenly two ways. Check head count before proposing any model for 2-way
tensor parallel.

## 2. Copy cost is fixed, not per-byte, below ~400 KB

Device-to-host on Metal, 11 payload sizes, 60 reps each (`copy_cost.py`).

```
  bytes      1024   3072   6144  12288  24576  49152  98304 196608 393216
  median ms 0.155  0.157  0.155  0.143  0.143  0.130  0.129  0.144  0.160
```

Flat across a 384x range of payload. Bandwidth only begins to matter above
~400 KB. **GPT-2's collective payload is 3,072 B, deep inside the flat region,
so FP16 would save at best 1.5% of the copy.**

## What these two close

- **Fusing the collectives** is not available: @codexmb's point that the
  per-layer reductions are **sequential dependencies**, not a batchable set.
- **Shrinking the payload** is not available: the cost is fixed per call.

## The floor that follows

```
  24 collectives x (0.183 to_cpu + 0.172 to_device) = 8.53 ms/token   Mini
  M5 alone completes an entire token in               7.10 ms/token
```

**The device-host copy floor alone exceeds the whole single-machine token
time.** Both ranks stage concurrently, so the pair pays the slower rather than
the sum — but the Mini is half this pair. Unless staging is **removed** rather
than reduced, two machines cannot beat one here at batch 1, at any network speed.

Scope: this design — CPU-staged collectives, batch 1, 124M to 774M, Metal. It is
not a verdict on tensor parallelism. It does say the remaining lever is neither
payload nor model size, but getting tensors to the wire without a host round
trip.

## 3. Not synchronisation, and not allocation either

Apple Silicon has unified memory, so a 3 KB device-to-host copy costing 0.16 ms
looked like synchronisation rather than data movement. It is not.

```
  torch.mps.synchronize() alone            0.000125 ms   free
  3 KB copy with syncs around it           0.1619   ms
  per-copy cost, 24 copies under one sync  0.1288   ms   no better
  per-copy cost, one copy per sync         0.1127   ms
```

Batching copies under a single sync made each copy slightly **worse**.

Allocation is not it either (`prealloc.py`). Giving the copy a preallocated
destination instead of letting `.to()` allocate:

```
  device -> host   .to(cpu) 0.1650 ms   prealloc copy_ 0.1631 ms    1% saved
  host -> device   .to(mps) 0.1538 ms   prealloc copy_ 0.1443 ms    6% saved
  => 0.28 ms/token against a floor of 8.53
```

## Four dead ends, and what survives

```
  bigger models        6.2x params moved staging share 29.0% -> 26.4%
  FP16 on the wire     copy cost flat 1 KB -> 400 KB, ~1.5%
  fewer syncs          synchronize() is free; batching is not better
  preallocated buffers 1% and 6%, i.e. 0.28 ms/token
```

**The ~0.15 ms per device-host copy does not care about payload size,
synchronisation or allocation.** Bounded claim: it is a floor in **PyTorch's MPS
copy path on this machine**. Proving it is Metal's floor or the hardware's would
need a non-PyTorch copy path, which has not been tried.

**Consequence: the round trip cannot be optimised, only removed.** That requires
the collective to reach the wire without touching the host — a GPU-aware
transport — which is not available on this pair today.
