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

## 4. The bottleneck is the Mac, by 6.5x

Same probe on the M5's ROCm side. No GPU window was needed: a few hundred 3 KB
copies is negligible load, and Flash-Next stayed `active` with health 200,
checked before and after.

```
  per device<->host copy, 3 KB
    Mini  Metal (MPS)    0.1650 ms
    M5    ROCm           0.0255 ms      6.5x cheaper
```

Fixed cost on both — flat across payload size on both — but the Mac's fixed cost
is six and a half times larger.

```
  staging floor per token, 24 collectives, both directions
    Mini   7.65 ms/token
    M5     1.22 ms/token
    M5 alone completes an entire token in 7.10 ms
```

**The Mac's copy overhead alone exceeds the whole single-machine token time. The
PC's is under a fifth of it.** Ranks stage concurrently, so the pair pays the Mac.

The ROCm figures were taken under contention, which can only make ROCm look worse
than it is. A clean measurement would widen this gap, not narrow it.

### What follows

1. **This is a PyTorch-on-Metal cost, not a tensor-parallel one.** Same design,
   same payloads, same collective count; the PC side alone would not bottleneck.
2. **It reframes the Mac+PC pairing.** The property that makes it appealing —
   most people own both — is also what makes it slow on current evidence. A
   PC+PC pair carries a floor 6x lower and is worth measuring.
3. The bound from section 3 matters more now: this is a floor in **PyTorch's**
   MPS copy path. Whether a non-PyTorch path on Metal does better is unknown and
   decides whether this is fixable software or a hardware property.

## 5. It was never the copy: 81% is GPU round-trip latency

Everything above calls this a copy cost. That label was wrong, and the control
that shows it is simple — time a trivial GPU op whose result **never leaves the
GPU**:

```
  loop + sync overhead only      0.0001 ms
  GPU op, result stays on GPU    0.1428 ms
  copy of a settled tensor       0.1571 ms
  GPU op THEN copy to host       0.1772 ms

  attributable to the copy            0.034 ms
  attributable to the GPU round trip  0.143 ms   = 81%
```

A `tensor.add(1.0)` that stays on the GPU costs 0.143 ms if you wait for it.

MLX, Apple's own framework on the same hardware, corroborates:

```
  MLX read of an ALREADY-COMPUTED array to host   0.0006 ms
  MLX GPU op + read to host                       0.138  ms
  PyTorch MPS op + copy                           0.117  ms
```

**Reading unified memory is almost free.** It becomes expensive only when a GPU
operation must complete first — and both frameworks pay the same toll, so this
is not a PyTorch defect.

### What it actually means

Tensor parallel here does not have a transfer problem, it has a **pipeline**
problem. Every collective forces the GPU to drain before its result can be sent.
**24 drains per token at 0.143 ms is ~3.4 ms of pure waiting**, before copying,
framing or network. A single machine runs the whole token as one continuous
pipeline.

This explains, rather than merely observing, why neither a smaller payload nor a
faster wire helps. The only lever is **fewer GPU-wait points per token**, and
those points are sequentially dependent.

### Correction to section 2

The "8.53 ms/token copy floor" number stands as measured; **its label was
wrong**. It is roughly 3.4 ms of GPU round trips plus copying, not 8.5 ms of
copying.

And it re-reads section 4: AMD's dispatch-and-complete round trip is ~6x cheaper
than Apple's — a hardware and driver property, not something our code can
optimise around.
