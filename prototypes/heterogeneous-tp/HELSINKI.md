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
| `gpt2_tp.py` | **real GPT-2 124M, real text** | passed, IDs match HF exactly |
| `fetch_gpt2.py` | pinned-revision download + per-file hashes | identical on both hosts |

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

## GPT-2: the one part that is not synthetic

`gpt2_tp.py` runs `openai-community/gpt2` at pinned revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, sharded across the two machines, and
compares against **HuggingFace's own `GPT2LMHeadModel`** (fp32, eager, eval, full
recompute per step). That reference is independent by construction: different
authors, different code.

It is a GPT-2 adapter, not the toy fed with GPT-2 weights. It honours learned
positional embeddings, LayerNorm with weight and bias, the fused Conv1D QKV
layout with its `(in, out)` weight orientation, exact `gelu_new`, and the tied
LM head.

```
3 prompts x full and chunked prefill x 16 new tokens
token IDs match the reference EXACTLY in every case, no divergent step
worst logit error 0.00104     declared tolerance atol 2e-2, rtol 1e-3
collectives per generated token: 25  (2 per block x 12 blocks, + 1 broadcast)
```

Rank 0 selects each token and **broadcasts it**, so the two ranks cannot drift
onto different sequences while both believing they are correct.

**Read the margin, not just the match.** `min_top1_top2_margin` against the worst
logit error for that case:

| prompt | margin | error | headroom |
|---|---|---|---|
| `The capital of France is` | 0.0071 | 0.00032 | **22x** |
| `In a shocking finding, scientists discovered` | 0.0272 | 0.00035 | 78x |
| `def add(a, b):` | 0.4658 | 0.00104 | 449x |

**22x is the real safety margin on the first prompt, not the orders of magnitude
the toy model enjoyed.** The IDs match, and they are not matching by luck — but a
change that grew the numerical error by one order of magnitude could start
flipping tokens on prompts like that one. Worth knowing before anyone reads
"token IDs match" as unconditional.

### Controls for the GPT-2 adapter

| control | why it is the one to run |
|---|---|
| `gc1_qkv_contiguous_slice` | fused QKV sliced as one contiguous range instead of three per-block slices — the most likely way to get this adapter wrong |
| `gc2_attn_bias_twice` | row-parallel bias added per rank instead of once after the sum |
| `gc3_no_positional_embeddings` | `wpe` dropped; the toy had no positional embeddings, so this path is new |

Plus `gc4_reset_different_split`: the repeat run prefills with the other chunk
split, which the bitwise reset check must reject. It does. That control exists
because the reset check was strengthened from "same IDs" to "bitwise identical
logits", and a stricter check is worth nothing until you show it can fail.

All four fail as required. Run with `--new-tokens 4` for speed; detection is
the point, not sequence length.

### On reading the margin

@codexmb's bound, which is the correct way to state this: **a sufficient
condition for same-step argmax agreement is `margin > 2 * max_abs_error`.** It is
satisfied for every case here and is reported per case as
`argmax_bound_margin_gt_2x_error`. It is evidence **for these samples only** and
is not a guarantee about other prompts.

### Checkpoint provenance

`fetch_gpt2.py` downloads only named files (no remote custom code) at the pinned
revision and hashes each one. Downloaded **independently on both machines** and
compared rather than copied, so the hashes show the revision pin resolves to the
same bytes on both:

```
model.safetensors  248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707
config.json        0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4
```

Setup and download sit outside any timed path, and nothing here is timed.

## Benchmark harness: written, validated, NOT RUN

`bench.py` + `run_bench.sh` measure Mini-solo (MPS) against M5-solo (ROCm)
against Mini+M5 tensor parallel, same revision, tokenizer, FP32, eval, one torch
thread, same KV cache and the same greedy/EOS policy in all three.

**Solo runs the FULL model on its own GPU** — a `NullCollective` makes
`all_reduce` the identity over one rank, so the arithmetic path is the same and
only the sharding differs. Never a half shard, never CPU.

**The correctness gate runs before any timing and the timings are withheld if it
fails.** Load, tokenize, hash and the reference check sit outside every timed
region, and the reference uses its own KV cache rather than recomputing the
prefix, so it is not doing different work from the thing being measured.

Instrumentation: each stamp is taken **when a token becomes available**, after a
GPU sync, so `stamps[0]` is exactly TTFT and `stamps[-1] - stamps[0]` spans
`len(out) - 1` tokens. An earlier version stamped after the following forward
pass, which quietly misaligned both numbers.

`run_bench.sh` **interleaves the three configurations per repetition** instead of
running them in three blocks, so a machine drifting over the session perturbs all
three equally. This fleet has already produced a 15% drift on a Mac across one
evening, which is larger than the effect being measured.

**It has not been run, and it refuses to run.** Its preflight aborts unless
`/tmp/m5-gpu-window.open` exists, because `llm-server.service` (Flash-Next) is
serving on the M5 and `~/m5-gpu-window.sh open` stops both it and the room agent.
Timing GPT-2 against a contended GPU produces numbers that would need more
disclaimer than they are worth.

## Not done

No timing. No trained checkpoint, tokenizer or real text. Single toy dimensions
(dim 128, 8 heads, hidden 256, vocab 64). Two ranks only.
