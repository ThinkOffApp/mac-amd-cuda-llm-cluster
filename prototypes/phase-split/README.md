# Cross-machine prefill/decode split

Stock `llama.cpp`, no patch: run `llama-server --slot-save-path DIR` on both ends,
`POST /slots/N?action=save` on the producer, copy the file, `action=restore` on the
decoder. The producer must prefill the first **N-1** prompt tokens and the decoder
receive all N, or the decoder re-prefills from scratch.

**SWA models (gemma-3 and friends) force a full re-prefill** unless both ends run
`--swa-full`. An earlier conclusion here that llama.cpp could not do this at all was
wrong: every test behind it used gemma-3, so five tests shared one hidden assumption
and were really one test.

## Measured, 2026-09-18

Mac mini (M4, **Metal**) + Bosgame M5 (**Vulkan**), gigabit ethernet.
The M5's `llama-server` links `libggml-vulkan.so.0` and no HIP/ROCm library --
an earlier version of this file said ROCm, which was wrong.
Qwen3.8-27B-UD-Q4_K_XL, 2107-token prompt, **both ends `-c 4096 -ngl 99 -np 1`**.

| step | time |
|---|---|
| M5 prefill, 2107 tokens | 7082.8 ms |
| save slot (295.0 MB = 281.4 MiB) | 92.2 ms |
| transfer M5 to Mini | 2.70 s = 109.3 MB/s (104 MiB/s) |
| restore on Mini | 91-107 ms |
| Mini finishes (1 prefill + 4 decode) | 0.97 s |
| **TTFT** | **~10.2 s** |
| Mini alone, same prompt | 38.15 s prefill / 38.78 s total |

**3.7x.** Same-machine save/restore: 38,152.7 ms cold to 271.0 ms restored = **141x**,
so the restore itself is nearly free and the wire sets the break-even. KV for this
model is **140,028 B/token** = 140.0 kB = 136.7 KiB.

Units matter here: a first pass at the break-even inherited a `MB`/`MiB` slip from me and
came out 5% low. Break-even link for this model is **9.50 MB/s = 76 Mbit/s** -- gigabit
has 11.5x headroom, so **the wire is not the constraint**; once transfer is small against
the fast machine's prefill, that prefill is the floor.

## What a faster link buys, and the term people drop

```
link             transfer   to 1st gen tok   speedup
100 Mbit          23.60 s        31.13 s      1.23x
gigabit, MEASURED  2.70 s        10.22 s      3.73x
10 GbE at 60%      0.39 s         7.92 s      4.82x   <- projection, not a result
```

Every row includes the **decoder's own prefill of token N, 241 ms**. That term is the
N-1 boundary itself -- the producer prefills N-1 and the decoder must still process
token N before it can generate -- so it is constant, and it grows as a share of the
total exactly where the win looks biggest. A table that omits it reads about 3% fast
at gigabit and puts the 10 GbE row at 4.98x instead of 4.82x.

**If a runner prefills all N on the producer instead of N-1, the decoder silently
re-prefills and the measurement is worthless.** That failure mode is quiet: the
numbers still come out, they are just the numbers for not having split at all.

Only the gigabit transfer is measured (109.3 MB/s against a nominal 125, so 87%
efficiency). The 10 GbE row assumes 60% and is a projection until somebody times a
file across that link.

## Correctness: matching tokens is not equivalence

The transfer is lossless (`sha256` of the slot file is identical on both machines),
but the two backends (Metal, Vulkan) do not produce the same KV for the same prompt, so the files
differ. `kvcheck.py` compares the next-token distributions. It replaces the first pair of
scripts, which @codexmb showed would print the headline result -- "same text True,
worst gap 0.000000" -- for two FAILED requests: `curl -sS` exits 0 on an HTTP error,
the restore response was never read, and `gap([], [])` returns 0.0. A zero from a dead
server was indistinguishable from a zero from a real control, and that zero was the
published evidence. `kvcheck.py --self-test` drives seven such cases and requires every
one to fail.

```
comparison              same text   worst |dlogprob| over shared top-10
pd.bin  vs pd.bin       True        0.000000    <- SAME FILE twice (control)
q27.bin vs q27.bin      True        0.000000    <- SAME FILE twice (control)
pd.bin  vs q27.bin      True        0.670641    <- Vulkan-KV vs Metal-KV
```

The same-file control is exactly zero, so the 0.67 is cross-backend and not run
noise. The top-10 sets agreed on only 8-9 of 10 entries. Greedy decoding matched for
four tokens and **that is the whole claim** — a 0.67 logprob gap will diverge under
sampling or over a long generation. Do not quote this result as "token-identical".

**This is a latency result, not a throughput one.** Decode after the restore ran at
5.33 tok/s against 4.81 solo, on n=3 tokens -- unchanged within noise, and it has to be.
The decoder does exactly what it did before; what disappears is 28 s of waiting. Do not
report this on a tok/s axis.

Also note the fast machine was **idle**: the M5's 7.08 s prefill was measured with its
serving workload stopped. Under real contention `prefill_fast` grows and break-even moves
up, so both terms of the rule must be measured under the load the box will actually carry.

### Is it the backend, or just a different chunk boundary?

That objection is not hypothetical -- @codexmb's own local failures moved when he
matched prefill chunk boundaries. Settled here without a second machine, by producing
two caches on the SAME machine that differ only in how the prompt was fed:

```
A   one request for all 2107 tokens
B   1107 tokens, then extended to 2107 (so the second pass prefilled 1000)

worst |dlogprob|, A vs B:  0.000000     <- chunking is not the confound
cross-machine, for comparison: 0.670641
```

So the cross-machine gap is not explained by chunking, at least on Metal. This does
not prove Vulkan is equally insensitive; it removes the confound on the side that
could be tested.

### What is still not established

The shared top-10 is **not a full-vocabulary logit gate**, and four matching greedy
tokens is a narrow positive observation, not a validated equivalence. Still owed: a
long greedy run, a sampled run, and a full-vocabulary comparison.

### How the cost scales

```
n=  512   prefill  9087 ms   slot 190.5 MB
n= 1024   prefill 18446 ms   slot 224.0 MB
n= 2048   prefill 38330 ms   slot 291.2 MB
n= 3072   prefill 56157 ms   slot 358.3 MB
```

Prefill is **linear** here (log-log slope 1.02, 1.06, 0.94), so long prompts do not
favour the split through superlinear attention at these lengths. The slot file is
**~157 MB fixed + ~65.5 kB/token**, NOT 140 kB/token: dividing the whole file by the
token count charges a constant block to the tokens. Any break-even rule needs
`KV_bytes = const + b*n`. The 9.50 MB/s figure above is still the right answer *at
2107 tokens* and does not generalise as `b*n`.

## Running it

Both servers up with `--slot-save-path`, the prompt saved as a token array in a
request body, then:

```bash
python3 kvcheck.py --self-test          # seven failure cases; all must fail
python3 kvcheck.py --expect-tokens 2107 \
    --pairs pd.bin:pd.bin q27.bin:q27.bin pd.bin:q27.bin --verbose
python3 chunk_control.py                # is it the backend or the chunk boundary?
python3 prefill_curve.py                # how the cost scales with prompt length
```

Raw receipts (hashes, both server logs, window closure, every run's output) are in
`receipts/`. `logit_check.py.bak` / `logit_control.py.bak` are the broken originals,
kept so the failure mode stays legible.
