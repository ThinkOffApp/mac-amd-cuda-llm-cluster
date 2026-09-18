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

## What the 3.7x actually measures

It measures how slow the decoder is, not how good the split is.

@grok's Berlin pair ran this identical path and got **0.84x -- a loss**. Same model,
same recipe, same code. The only variable that matters is `prefill_slow - prefill_fast`:
this Mac mini prefills at 55 tok/s, his Mac Studio at 704.

```
                        this pair      Berlin pair
fast machine prefill      7.083 s        2.662 s
slow machine, cold       38.153 s        3.001 s
ratio between machines      5.4x           1.13x
fixed cost (no wire)      7.516 s        2.896 s
budget left for the wire 30.637 s        0.105 s
break-even link           9.5 MB/s      2812 MB/s = 22.5 Gbit/s
measured link           109.3 MB/s       439 MB/s over 10G
result                      3.73x          0.84x
```

**Phase splitting is a fix for a slow decoder.** Where both machines are fast it is a
tax, and no link fixes that: the Berlin pair would need 22 Gbit before it broke even at
2k tokens. Quote the 3.7x only with the pair it came from.

## Correctness: matching tokens is not equivalence

The transfer is lossless (`sha256` of the slot file is identical on both machines),
but the two machines do not produce the same KV for the same prompt, so the files differ.
**Why they differ is not established**: the backends differ (Metal, Vulkan) and so do the
prefill chunk boundaries, and the control below shows chunking alone is enough to move the
distributions. `kvcheck.py` compares the next-token distributions. It replaces the first pair of
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
pd.bin  vs q27.bin      True        0.670641    <- cross-machine (cause NOT established)
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
matched prefill chunk boundaries.

**The first version of this control was VOID**, and he found it in the raw log
committed here. It saved a 2107-token cache and then sent back the same 2107 tokens,
which takes llama.cpp's rewind path: the server re-prefilled everything
(`receipts/decoder-server.log:244` and `:253`, tasks 158 and 166, 38.6 s and 39.2 s),
discarded the restored history and built a fresh native cache on both sides. Two fresh
native prefills agree exactly, so its `0.000000` said nothing about chunking. The void
script and its log are kept as `chunk_control_INVALID.py.bak` and
`receipts/chunk-control.txt`.

The corrected control saves N-1 and requests N, and **asserts `cache_n == N-1` and
`prompt_n == 1` before comparing anything**. With a confirmed cache hit on both sides
it does **not** reproduce zero:

```
A   one request for all 2106 tokens        cache_n=2106 prompt_n=1
B   1107 tokens, then extended to 2106     cache_n=2106 prompt_n=1

SAME BACKEND, DIFFERENT CHUNKING   worst |dlogprob| = 0.269895
cross-machine, for comparison                        0.670641
```

**A chunk-plan difference alone moves the distributions**, on one machine with one
backend, one build and one set of weights. A five-pattern sweep (`chunk_sweep.py`,
`receipts/chunk-sweep.txt`) puts the largest such difference at **0.449241**, with all
five plans emitting identical text:

```
half vs late            0.449241        one-shot vs late        0.193828
half vs early           0.422413        one-shot vs early       0.154447
quarter-steps vs late   0.378033        half vs quarter-steps   0.071208
quarter-steps vs early  0.351205        late vs early           0.039978
one-shot vs half        0.269895        one-shot vs quarter     0.198687
```

Per-step, the chunk effect is not even consistently smaller than the cross-machine one
(`receipts/per-step.txt`):

```
step     cross-machine    chunk plan
0             0.238789      0.269895   <- chunk plan LARGER
1             0.272204      0.053249
2             0.670641      0.024322
3             0.225183      0.047880
```

Those two columns come from different caches (2107 vs 2106 tokens), different prompt
lengths and different generated tokens, so they are shown together only to compare
shapes. (Setting 0.269895 against 0.670641 as a
percentage would be wrong: they are maxima from different experiments, at different
steps, over different tokens -- a ratio, not a decomposition. The chunk contribution to
the physical gap is unknown.) So the cross-machine difference is **NOT established as a
backend effect**, and the earlier claim in this file that chunking was
ruled out is withdrawn -- it rested on the void control above, which returned a clean
zero because it was comparing two fresh native prefills.

The producer and the decoder do not batch identically and those boundaries were never
matched. Separating backend from chunk plan needs the producer, matched boundaries, and
a GPU window.

### Does the fixed block belong to the model or to `-c`?

If the 156.9 MB were a slice of the allocated context, every crossover computed from
it would be valid only at `-c 4096`. Measured, same 512-token prompt, only `-c` varying:

```
-c 2048    190,463,552 B
-c 4096    190,463,552 B      difference: 0 B
```

**Equal at these two settings.** That is the whole claim: `-c 2048` and `-c 4096` give
the same state size for a 512-token prompt. It is **not** universal context portability
-- no other `-c` has been tested, and a rule computed at one context is evidence about
that context.

Kept separate, because it is a different fact: **`-c 8192` does not fit beside a 27B on
this 24 GB machine.** It loads, answers `/health` with 200, and returns `Compute error`
on every completion. That is a memory limit on this host, not a measurement of state
size. The two servers must run sequentially, and a health probe is never evidence that a
server works -- only a completion that returns content is.

### Crossover lengths are NOT bounds

An earlier version of this file, and of the room discussion, called a crossover computed
outside the calibrated range an "upper bound". **That is wrong.** Each machine's prefill
curve has a quadratic term, but the *difference* of two such curves need not grow
superlinearly and need not stay positive: kernels, hybrid attention, context settings and
memory pressure move either curve independently. Outside its data the fit gives neither a
value nor a bound nor even a sign, and a near-zero denominator does not establish "never
pays at any length" either. `calibrate.py` refuses rather than answering, and refuses
outside the calibrated model, build and `-c` as well as the calibrated length.

### The full-vocabulary gate FAILS (@codexmb, on the Berlin pair)

Run against @grok's preserved producer file with the producer's own chunk boundaries
matched `[512,1024,1536,1599,2107,2111]`:

```
16 greedy token IDs        reproduce native AND the producer's report
full-vocabulary gate       FAILS at atol 0.1 / rtol 0.01
step 0                     max abs error 1.31, 179,629 logits out of tolerance
across 16 steps            max 9.46
same-host Metal control    passes all 16 steps, EXACTLY zero logit error
```

That same-host control with boundaries matched is the isolation this repo's own sweep
could not do -- it shows only that *unmatched* plans move the distributions. With them
matched, same host is exactly zero and cross machine is not. So on that pair chunk plan
is excluded and something large remains, though @codexmb is explicit that it does not
isolate backend arithmetic from other server graph differences, and that pair is
CUDA->Metal where this one is Vulkan->Metal.

**It is also a different quantity from anything measured here.** 9.46 is a maximum
**raw-logit** difference across the full vocabulary; the 0.67 here is a **log-probability**
difference inside a top-10. Log-probs are logits minus `logsumexp`, so a uniform shift
disappears from one and not the other -- they cannot be compared, let alone divided. A
top-k comparison is a keyhole *and* it is looking at a different thing. **Do not relabel a greedy match as
correctness, and do not relax the tolerance.**

### What is still not established

The shared top-10 is **not a full-vocabulary logit gate**, and four matching greedy
tokens is a narrow positive observation, not a validated equivalence. Still owed: a
long greedy run, a sampled run, and a full-vocabulary comparison.

What the runs establish, in @codexmb's terms: **a real prefix-reuse path and a real
difference in the sampled distributions, with the cause not established.** Not a
validated inference speedup, and not validation of a universal install-time rule.

Distributions are keyed by **token ID, never by decoded text** -- distinct ids can
decode to the same string, and a text-keyed dict silently collapsed them. That bug is
why a request for the top 10 was reporting 9 shared entries, and it changed two of the
four per-step numbers first published (step 0: 0.102773 -> 0.238789, step 1: 0.097987
-> 0.272204). The headline 0.670641 and the exactly-zero controls survived it.

### How the cost scales

```
n=  512   prefill  9087 ms   slot 190.5 MB
n= 1024   prefill 18446 ms   slot 224.0 MB
n= 2048   prefill 38330 ms   slot 291.2 MB
n= 3072   prefill 56157 ms   slot 358.3 MB
```

**Confirmed on a second pair.** @grok ran the same N-1 path on a Spark + Mac Studio in
Berlin: 295,300,388 B at 2111 tokens against this file's 295,038,132 B at 2107. Four
tokens apart, on different hardware, a different link and a different build:

```
marginal KV from the two files    65,564 B/token
fit from the Mini's length curve  65,547 B/token    0.03% apart
implied fixed block               156.9 MB
whole file / token count         140,028 B/token    <- the wrong figure
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


## Server path beats the bench table, and bandwidth is not a constant

@grok measured the real handoff at three lengths (`-c 10240`, `-b 512 -ub 512`):

```
     n   mac_s  spark_s    gap  overhead  split_s   TTFT
  2111   3.001    2.662  0.339     0.906    3.568   0.84x  loss
  3072   4.227    3.736  0.491     0.950    4.686   0.90x  loss
  8192  12.921    9.915  3.006     1.556   11.471   1.13x  win
```

**Server crossover is between 3,072 and 8,192 (~4,303), not the 2,048-3,072 the
llama-bench conversion gave.** That conversion predicted 3,072 would win; the server
measured a 0.90x loss. Mac `llama-server` is less superlinear than `llama-bench`
(727 t/s at 3072 against the bench's 609), which is precisely why `--timing-source`
refuses bench-derived timings.

The overhead column also exposed a modelling error here:

```
     n   measured   this model    err   KV MB   implied MB/s
  2111      0.906        0.907  +0.001   295.3        439
  3072      0.950        1.050  +0.100   358.3        500
  8192      1.556        1.815  +0.259   693.9        525
```

**Effective bandwidth rises with file size** as startup cost amortises. Fixing it at the
439 MB/s measured on the *smallest* file over-charges every larger transfer and biases
the answer toward do-not-split -- by 0.26 s at 8k tokens. `calibrate.py` now requires
`--link-measured-at-mb` and refuses to apply a figure more than 2x away from the size
actually being moved.

The good news in the same table: `156.9 MB + 65,547/token` predicts that pair's overhead
at 2,111 tokens to within one millisecond, on hardware this repo has never touched.


## The row nobody has measured

The correctness evidence has a hole in the middle of it:

```
same HOST, boundaries matched            0.000000 exactly        measured
same BACKEND, DIFFERENT HOSTS            ** never tested **
different backend, different hosts       gate FAILS, max 9.46    measured
                                         (full-vocab RAW LOGITS -- not comparable
                                          to this repo's top-k log-prob numbers)
```

@codexmb's zero-error control was **same host**, which is not the same thing as same
backend on two machines: driver version, chip stepping, kernel selection and thread
count all still differ. **Until the middle row exists, "backend boundary" and "machine
boundary" are indistinguishable**, and any claim that same-backend pairs are lossless is
asserted rather than measured.

The experiment is cheap and the fleet already owns the hardware for it: two GB10 Sparks,
same CUDA build, different hosts. Save on one, restore on the other, run the full
vocabulary gate.

**Matched run, @claudeMB, 15:11: 0.000000 over a shared top-20, reproduced twice, gate
PASSES.** Both arms evaluate token 1,201 through the identical chunk plan
(512/512/176 then 1), both asserted at `cache_n=1200, prompt_n=1`, with an injected 0.4
detected before either arm ran. The only difference is where the 1,200-token prefix was
computed -- restored over the wire from asus1, or computed locally on asus2.

**THESE CELLS CANNOT BE PUT IN ONE TABLE**, and an earlier version of this file did it
anyway:

```
same host                            0.000000   FULL-VOCAB RAW LOGITS
same backend, different hosts        0.000000   top-20 LOG-PROBS, 8 steps
different backend, different hosts   9.46       FULL-VOCAB RAW LOGITS
```

@codexmb: those are not the same quantity, not merely different coverage. A log
probability is a logit minus `logsumexp`, so **a uniform shift across the vocabulary
vanishes entirely in log-prob space and is fully visible in raw logits.** Every number
measured in this repo -- the 0.67, the 0.449, the chunk sweep -- is a top-k log-prob
difference. The 1.31 and 9.46 are full-vocabulary raw-logit differences. Setting one
against the other is not a comparison.

Coverage is the second problem on top of that: 160 values is 0.1% of this vocabulary,
and at step 0 of the cross-backend run the top-10 moved 0.67 while **179,629 logits were
out of tolerance**. A quiet top-k is consistent with a loud vocabulary.

So the middle cell reads: **"shared top-20 log-probs agree in these runs; the
full-vocabulary gate is pending."** Not "PASS", not "lossless", not "deployable". The
acceptance test is full vocabulary at atol=0.1 **and** rtol=0.01 under its specified
formula, and a top-k atol-only diagnostic is not a substitute for it.

**First attempt, @claudeMB, 15:04: 0.403369 over a shared top-20, text identical, gate
FAILS -- and confounded.** The producer prefilled 1,200 tokens and the native reference
1,201, which under `-ub 512` are different chunk plans. 0.403 sits inside the 0.19-0.45
this repo measured for chunk plan alone on a single machine, so it is not evidence of a
host effect. He said so himself before anyone quoted it.

**Matching the prefix LENGTH is not matching the chunk PLAN**, and the difference is
easy to miss because the fix looks done:

```
producer:  prefill 1200            -> chunks 512, 512, 176   then save
consumer:  restore 1200, eval 1201 -> chunks 512, 512, 176,  then 1

native fed 1201 as ONE request     -> chunks 512, 512, 177   <- still different
native fed 1200, then 1201         -> chunks 512, 512, 176,  then 1   <- matches
```

**0.403 unmatched against 0.000000 matched, on identical hardware with everything else
held fixed, is the cleanest demonstration anyone produced that chunk plan alone moves
these numbers** -- worth keeping as a result in its own right, separately from what it
was run to test.

**The N-1 discipline applies to the REFERENCE arm too.** Every run today was careful
that the *split* arm hit the boundary and let the native arm run one-shot; @codexmb's is
the exception (*"Native prefill matched producer log boundaries ... then the final prompt
token separately"*), which is why his is the only cross-machine number with the schedule
controlled.

```
gate PASSES  ->  backend boundary. Same-backend pairs are safe, and a rule that
                 detects unequal same-backend hardware is deployable.
gate FAILS   ->  machine boundary. No cross-machine handoff is numerically safe
                 and refuse-by-default is permanent.
```

Two equally fast machines have nothing to win on performance, which is exactly what
makes them the right correctness control: no speed difference to confound it.

## What decides deployability

Not the gate. **Nobody has shown the divergence changes a user-visible answer.** The
measurement that settles it is a **flip rate**: a few hundred generations under real
sampling settings, split against un-split, counting how often the emitted text differs.
Sixteen matching greedy steps says almost nothing about that in either direction.

Both are now implemented rather than recommended:

- `calibrate.py --producer-backend/--consumer-backend` **refuses a cross-backend pair**
  by default, and the refusal names the flip rate as what would settle it.
  `--accept-numerical-divergence` overrides, for someone who has measured it.
- `flip_rate.py` measures it: N generations under *your* sampling settings, split
  against un-split, same seed each time, reported as a rate with a Wilson interval
  (which stays meaningful at zero -- 0/300 still reaches 1.26%).

`flip_rate.py` refuses to report a rate without three guards, and the third is the one
people leave out:

```
cache hit asserted   cache_n == N-1 and prompt_n == 1 on every split run
negative control     split vs split at one seed must AGREE
POSITIVE control     an INJECTED difference must be DETECTED
```

Without the positive control a zero flip rate is exactly what a harness comparing a
string with itself produces -- the failure mode that got a confident `0.000000`
published earlier the same day.

The first version of that positive control used a deliberately **altered prompt** and
was unsound: @codexmb pointed out an altered prompt is not guaranteed to change the
output, so it tests the model rather than the harness and fails spuriously. The
difference is now injected into the compared values and exercises the same `differs()`
the measurement uses.

Three further constraints, all his:

- **A flip is the whole generation differing in any token.** A per-token divergence
  rate is a different quantity and is not what this reports.
- **Vary prompts, not only seeds.** One prompt with many seeds measures that prompt,
  not a workload; the interval is labelled for the sampled workload and settings only.
- **A low flip rate does not authorize deployment** and does not override the failed
  numerical gate. It is one input to a decision that stays a human's, and
  `--accept-numerical-divergence` is worded that way.

Also from him, on the performance side: **~4,303 tokens is an interpolation between
timings, not a measured crossover**, and the 8,192-token 1.13x observation is
correctness-unqualified.

**Note on this pair specifically: it has never been through the full-vocabulary gate.**
The 3.73x here is a performance number. The only gate run is @codexmb's on CUDA->Metal;
what was measured here is 0.67 inside a top-10, which that result shows is a keyhole.


## Which comparison answers the question

Four quantities were argued over and three of them proxy for "does the split change the
answer" rather than answering it. `metric_ladder.py` shows why with no model and no
hardware:

```
                                  max RAW LOGIT  max LOG-PROB   TV distance
uniform shift of 9.46                  9.460000      0.000000     7.834e-17
non-uniform noise, sd 0.05             0.208582      0.208925     1.017e-02
one p=1e-17 token moved 3.0            3.000000      3.000000     1.809e-15
```

**Row 1**: softmax is shift-invariant, so a uniform shift across the vocabulary is a
large raw-logit difference that changes nothing. The number currently gating deployment
-- `9.46` -- is a raw-logit maximum, and **nobody has decomposed how much of it is shift**.

**Row 3**: a maximum over ~152k entries is set by whichever entry moved most, and the
tail moves most while mattering least. A token at p=1e-17 fails an atol=0.1 log-prob
gate while being unsamplable.

**Only TV is small in both.**

```
TV(p, q) = 0.5 * sum |p_i - q_i|
```

**TV is the MINIMUM probability that a single sampled token differs, over all
couplings** -- a LOWER bound, achieved only by the optimal coupling. An earlier version
of this file called it the maximum and wrote "whatever the sampler does". @claudeMB
caught it, and it was wrong in the direction that licenses a deployment. Measured:

```
TV(p,q)                              1.731%
optimal coupling (theory)            1.731%    <- TV IS THIS
shared uniform, same token order    23.863%    <- 13.8x TV
independent randomness              97.678%    <- 56x TV
```

**TV is still worth computing.** It is a floor, so a large TV settles the question
immediately and against us; `TV = 0` exactly means `p = q`, and identical distributions
give identical samples under a shared seed, so a zero is conclusive. It also separates
the shift-invariant part of `9.46` from the part that reshapes the distribution, which
nobody has decomposed. It is only the small-but-nonzero range where it says far less
than it looks like it says.

```
full-vocab RAW LOGITS    fails on differences that cannot change an output
full-vocab LOG-PROBS     right space, maximum decided by the irrelevant tail
TV DISTANCE per step     a FLOOR on disagreement; zero is conclusive, small is not
observed FLIP RATE       the only thing that answers the question   <- decisive
```

**There is no cheap bound standing in for the flip rate.** `flip_rate.py` uses the same
seed on both arms, which is the shared-uniform coupling above, where disagreement runs
many times TV -- so a rate well above TV is what a *correct* harness produces, and the
self-check an earlier version of this file described would have flagged good runs as
broken. It has been removed.

Caveat on the demo above: inverse-CDF over an untruncated 2,000-token distribution. Real
sampling truncates to `top_k` first, where orderings agree far more often, so the
multiple is illustrative rather than a prediction. Only the direction of the inequality
is certain.


## Validate the flip-rate harness before trusting it

@claudeMB measured **bit-identical full-vocabulary log-probs** for CUDA -> CUDA across
two hosts, chunk plan matched, with three distinct freshly-started server processes
verified by PID rather than inferred from timestamps.

Bit-identical log-probs means bit-identical probabilities, and every sampler transform
is a function of those, so **that pair must return exactly 0/N flips at any settings**.
That is a positive control at the experiment level: a single flip there means the
harness is broken, not the split. Run it there first, before pointing it at a pair whose
answer is unknown and where a wrong number cannot be recognised as wrong.

It also extends the depth for free -- that measurement was four steps, and the failure
mode for a product is a long answer.

Two caveats that survive the zero, neither a criticism of it:

- **Four steps is not four hundred.** Exactly-zero at step 4 is consistent with
  divergence at step 400.
- **A noise floor of exactly 0.000000 means the local arm is deterministic**, which
  makes the measurement clean and also means it cannot detect nondeterminism that only
  appears under concurrency. A serving box runs batched with other requests; a quiescent
  benchmark is a different condition.

**And the truncation amplification recorded above does not bind on this result.** It is
about small-but-nonzero TV, where two distributions can disagree about which tokens
clear the rank-k boundary. Exactly zero has no boundary to disagree about.
