# Helsinki pair: Mac mini (Metal) x Bosgame M5 (ROCm)

Correctness work on the framed TCP backend from `tcp_collectives.py`, run between
two machines whose PyTorch builds cannot share a Gloo transport.

```
rank 0  Petruss-Mini  TP_DEVICE=mps   torch 2.11.0                       macOS
rank 1  bosgamem5     TP_DEVICE=rocm  torch 2.12.0a0+rocm7.13.0a20260411 Linux
link    gigabit LAN, no torchrun, plain RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT
```

**Synthetic weights throughout. Generated token IDs are arithmetic, not language.
Nothing here is text generation, a trained-model result, or a speed measurement.**

## What is here

| file | gate | cross-host result |
|---|---|---|
| `block.py` | one TP decoder block, prefill only | passed, worst 1.9e-6 |
| `decode.py` | + per-layer KV cache, chunked prefill, decode steps | passed, worst 1.8e-6 |
| `model.py` | + 2 blocks, embeddings, LM head, 8 greedy tokens | passed, IDs match |
| `weight_hash.py` | hashes the rank-0 weight draw, for build comparison | see below |

Run any of them with the two-rank env contract, for example:

```bash
# rank 0
RANK=0 WORLD_SIZE=2 MASTER_ADDR=<rank0-ip> MASTER_PORT=29671 \
  TP_DEVICE=mps python3 model.py --transport tcp
# rank 1
RANK=1 WORLD_SIZE=2 MASTER_ADDR=<rank0-ip> MASTER_PORT=29671 \
  TP_DEVICE=rocm python3 model.py --transport tcp
```

## The rule these harnesses are built on

**The reference must share no machinery with the sharded path.** Not the norm
helper, not the causal mask, not the cache.

This is not a style preference. In the first version of `decode.py` the reference
called the same `rms_norm()` as the sharded path, and the norm weights were drawn
at scale 0.0, so they were all ones. A control that deleted the norm weight
multiply **passed**, because it deleted it from both sides at once and there was
nothing to delete anyway. The norm was untested and looked verified.

So every harness here keeps `_ref_norm()` written out with different arithmetic
from `rms_norm()`, and the reference recomputes the whole prefix with a plain
`triu(1)` mask rather than the offset mask the cached path uses.

## Negative controls

Every gate ships controls that **must fail**. A sharded test that cannot fail is
an expensive way to run the reference twice. Sources and both ranks' logs are in
`controls-helsinki/`; `SUMMARY.txt` in each directory carries the exit codes.

`decode.py` (`controls-helsinki/decode/`):

| control | mechanism that caught it |
|---|---|
| `nc1_missing_cache` | cache shape assertion |
| `nc2_wrong_cache` | numeric, right shape but time-reversed |
| `nc3_offbyone_future` | numeric, mask offset +1 |
| `nc4_offbyone_self` | numeric, mask offset -1 |
| `nc5_norm_weight_dropped` | numeric — **passed before the independence fix** |

`model.py` (`controls-helsinki/model/`):

| control | what only exists with >1 block |
|---|---|
| `mc1_shared_cache_across_layers` | both layers writing one cache |
| `mc2_layer0_weights_twice` | layer 0 weights applied to both blocks |
| `mc3_offbyone_future` | |
| `mc4_norm_weight_dropped` | |
| `mc5_missing_cache` | |

## Sharding

Sharded by rank: attention heads, MLP hidden channels. Row-parallel outputs are
summed with `all_reduce` and **their biases are added once, after the sum**.

Replicated: embeddings, LM head, norms, residual stream. Sharding the LM head
would need an `all_gather` this transport does not implement.

**Collectives: 2 `all_reduce` per block**, so 4 per generated token in `model.py`.
Payload is `tokens x dim x 4` bytes per collective — 512 B at dim 128, one token.
The two-rank exchange sends each payload both ways, so aggregate link payload is
4x that per decoded token, before framing and setup. **No speed claim follows
from these byte counts; nothing here has been timed.**

## Greedy match is reported with its margin

`model.py` records `min_top1_top2_margin` beside the logit error. Matching argmax
across a near-tie would be a coin flip presented as a result. Observed margin
~0.064, against a worst logit error of 2.1e-6.

## Weight hashes

`model.py` hashes **the actual post-transform tensors the run uses**, per layer,
and emits them in its report under `weight_sha256_post_transform`. They are
computed inside the run rather than replicated in a second script, because a
separate replica of the draw order drifts from the thing it claims to describe.

Two properties fall out of the cross-host run and are worth reading off:

```
combined hash, rank 0 == combined hash, rank 1     the broadcast really does
                                                   distribute identical weights
layer0 hashes != layer1 hashes                     the two blocks really are
                                                   independently weighted
```

Both were previously asserted in prose and are now evidenced.

`weight_hash.py` is a **diagnostic of the RNG draw stream only**. It draws the
norm weights at scale 0.0 and hashes values before the `+1` transform, so it
describes the draw stream of a superseded harness, not any current weight set.
Caveat raised by @codexmb. What it does still support is the cross-build
comparison: the same seed produces different draws on

```
Mini  torch 2.11.0                       combined d759bcde...
M5    torch 2.12.0a0+rocm7.13.0a20260411 combined fdd71511...
```

**This shows the two builds draw differently. It does not isolate the cause to
the version number** — build flags or platform would explain it equally well.

## Not done

No timing. No trained checkpoint, tokenizer or real text. Single toy dimensions
(dim 128, 8 heads, hidden 256, vocab 64). Two ranks only.
