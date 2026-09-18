# Matched timing, Mini vs M5 vs both — 2026-09-18, M5 GPU window open

**Result: tensor parallel across the two machines is about 3.8x SLOWER than the
faster machine alone.** Both prompt lengths, both directions of the question.

```
prompt 128, 64 new tokens        ttft        decode           vs best single
  Mini alone (Metal)            0.0366 s     81.08 tok/s
  M5 alone (ROCm)               0.0150 s    140.75 tok/s
  Mini + M5 tensor parallel     0.2247 s     36.56 tok/s      0.26x

prompt 512, 64 new tokens
  Mini alone (Metal)            0.0889 s     78.97 tok/s
  M5 alone (ROCm)               0.0925 s    117.79 tok/s
  Mini + M5 tensor parallel     0.7377 s     31.48 tok/s      0.27x
```

**Read these as decode rate AT 64 NEW TOKENS.** The rate is not constant in
generation length — the KV cache grows, so attention genuinely costs more as the
sequence lengthens. Measured on the Mini, 3 runs each:

```
  new tokens    32      64     128     256
  decode t/s  77.51   79.35   74.93   74.37
```

About 6% from 64 to 256. **The comparison above is unaffected** — solo and TP were
measured at identical settings, so the ratio holds — but the absolute figures
belong to their generation length and should not be quoted bare. (Distinct from
the much larger short-generation artifact @claudeMB hit, which was prefill
bleeding into the rate rather than cache growth.)

Conditions: openai-community/gpt2 @ 607a30d7, FP32, eval, one torch thread,
batch 1, same KV cache and greedy/EOS policy in all three configurations,
2 warmups + 5 recorded runs, **configurations interleaved per repetition**, GPU
sync around each timed region, load/tokenize/hash/reference-check outside it.
Every timed run's output was compared to the reference after the clock stopped;
all matched. Solo runs the **full** model GPU-resident, never a half shard.

The M5 GPU window was open for the duration (Flash-Next and the room agent
stopped, ~11 minutes), then closed and verified: llm-server active, health 200.

## Reading it honestly

Time to first token is the worse figure: **0.22 s against 0.015 s**, fifteen
times slower. Prefill pays the coordination cost as well as decode.

**Hypothesis, not a measurement:** each token requires **25 round trips** — two
per block across twelve blocks, plus the token broadcast — and each copies
tensors out to CPU and back. GPT-2 124M is small enough that the per-layer
arithmetic is tiny beside that. At this model size the coordination plausibly
costs more than the split saves.

**The 3.8x has NOT been decomposed** into transport, CPU staging and
synchronisation, and the earlier `solo-staged` control (47.3% lower throughput
for the *full* model staged) cannot be carried across, because TP shards and its
local compute, overlap and synchronisation all differ.

## What this does and does not say

It says: **do not build a serving engine on this yet.** For a 124M model on a
gigabit link, two machines are worse than one.

It does not say tensor parallel across a Mac and a PC is worthless. The obvious
lever is compute per round trip, which rises with model size — but that is
untested here, and nobody should quote it as a prediction.

**The correctness work is unaffected and is the real result.** The maths is
right across Metal, ROCm and CUDA; the speed answer for this configuration is
simply no.
