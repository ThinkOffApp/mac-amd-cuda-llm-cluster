# What we are testing next: models that fit in no single machine

The plan, not a result: the capacity totals, the ladder of model sizes past the 121 GiB single-box line, and what a size does not prove.

> Moved verbatim from the [README](../README.md) on 23 Sep 2026 to keep the README short.
> Nothing was cut. Where the text says "above", "below" or "this README", it refers to
> the README as it was before the move; the [README's last section](../README.md#details-and-all-numbers)
> lists where each part now lives.

## What we are testing next: models that fit in no single machine

The models worth pointing this cluster at are the ones that fit nowhere in it on
their own. The most any single box here can hold is about **121 GiB**, once the
runtime's own overhead is counted, so anything above that line is the actual subject
and everything below it is a control. That addressable figure, not the installed
memory figure, is the one the model sizes below are compared against.

**None of this has been run.** It is a plan. The model sizes are read from published
model files; everything about what will actually execute is unverified.

### Three totals, and they are not the same number

Treating these as interchangeable is exactly the kind of error this repository
exists to avoid, so all three are stated separately:

| figure | what it actually is |
|---|---|
| **512 GB** | raw installed memory, four boxes at 128 GB each. **Not usable capacity** — no run will ever see this number. |
| **≈493 GB** (459 GiB) | what the runtimes can address, added up: 107 GiB on the Mac, 110 GiB on the Strix Halo, 121 GiB on each of the two Sparks (measured 20 Sep 2026) |
| **384 GB** | the practical target for inference that never swaps. Deliberately conservative, and the figure the ladder below is trying to approach. |

The distance between the first and second is the ordinary gap between memory a
machine contains and memory a GPU runtime is allowed to hold — and the per-box
figures agree with the device budgets already recorded independently in the
[capacity section](measurements-mac-spark-split.md#capacity-the-case-where-the-ratio-does-not-exist) above. The
distance between the second and third is deliberate headroom for everything that is
not model weights.

### The ladder

Climb model sizes one rung at a time and find where it stops working.

**The first rung is chosen so that failure is cheap.** DeepSeek V4-Flash-0731
UD-IQ4_XS, at 127.3 GiB, sits just past the single-box line. It is the least
expensive test that spreading a model across machines works at all: if the transport
is broken, this breaks immediately, before any long run has been paid for.

From there the rungs climb through quantizations of DeepSeek, Qwen3.8-Flash-Next and
GLM-5.3-Flash, to see how close to the 384 GB mark inference can get while staying
non-swapping.

One rung has effectively been climbed already. The capacity section above records
**GLM-5.3-Flash UD-Q4_K_XL at 185.98 GiB running across two boxes that could not hold
it individually** — no speedup to quote, but the model ran where it otherwise could
not. That result is the reason this ladder looks worth building.

### The rungs

Sizes are in GiB, summed across shards where a quantization ships in several parts,
and read from the published model files rather than estimated. **The line to compare
against is 121 GiB.** DeepSeek V4-Flash-0731 was published 2026-07-31;
Qwen3.8-Flash-Next and GLM-5.3-Flash both 2026-08-26.

**These are candidates, not predictions.** A size says a model might fit. It does not
say it will run, and the precision of these figures should not be mistaken for
confidence about that — see the limits at the end of this section.

**DeepSeek V4-Flash-0731**

| GiB | quantization | |
|---:|---|---|
| 97.1 | UD-IQ3_XXS | below the line — control |
| 108.1 | UD-IQ3_S | below the line |
| 119.3 | UD-Q3_K_M | below the line, barely |
| **127.3** | **UD-IQ4_XS** | **the first rung** |
| 127.3 | UD-IQ4_NL | |
| 144.3 | MXFP4 | |
| 144.4 | UD-Q4_K_XL | |
| 150.8 | UD-Q8_K_XL | |

**Qwen3.8-Flash-Next**

| GiB | quantization | |
|---:|---|---|
| **76.3** | **UD-IQ3_XXS** | **below the line — and running right now** |
| 103.7 | UD-Q4_K_XL | below the line |
| 147.4 | UD-Q5_K_XL | |
| 157.5 | UD-Q6_K_XL | |
| 175.3 | Q8_0 | |
| 329.7 | BF16 | unquantised |

**GLM-5.3-Flash**

| GiB | quantization | |
|---:|---|---|
| 86.7 | UD-IQ1_S | below the line |
| 101.3 | UD-Q2_K_XL | below the line |
| 112.1 | UD-IQ3_XXS | below the line |
| 137.4 | UD-Q3_K_XL | |
| 146.1 | UD-IQ4_XS | |
| 186.0 | UD-Q4_K_XL | already run across two boxes, see the capacity section |
| 223.8 | UD-Q5_K_XL | |
| 271.8 | UD-Q6_K_XL | |
| 317.6 | Q8_0 | |
| 597.6 | BF16 | **ruled out on arithmetic alone** |

**One rung already has a live baseline, which makes it the most useful place to start
asking about quality.** The 76.3 GiB Qwen file is what the Strix Halo box is serving
today — it is visible on the fleet dashboard pictured earlier in this README, beside
that machine's name. Climbing the same family to Q8_0 at 175.3 GiB, or to the
unquantised BF16 at 329.7, would compare against something measured daily rather than
against nothing.

**One row is excluded before any of this starts, and it is listed because it is
excluded.** GLM-5.3-Flash at BF16 is 597.6 GiB. That is larger than all four boxes
added together, and far larger than the roughly 459 GiB the runtimes can actually
address. No split, no transport and no backend changes that. The ceiling is real, and
recording it here is cheaper than having someone rediscover it on a plan later.

### Two questions, and the second one is open

**Speed, first.** Prefill and generation throughput as the model grows and more boxes
are brought in. This is the familiar question, and most of this README is already
about it.

**Whether splitting the work makes total memory use *smaller*, second — and this is
the non-obvious half.** We do not know. Spreading a model across machines plausibly
adds per-box overhead, and it plausibly avoids duplication a single box would have to
pay for. Those pull in opposite directions, and **nobody here has measured which one
wins.** It is written as an open question because that is what it is, and settling it
is part of what these runs are for.

### What a size on this ladder does not prove

A model's summed weight size establishes **candidate fit, and nothing beyond it.** It
does not establish:

- that every backend involved supports that model's architecture,
- that a layer split divides into proportions each box can actually hold,
- or that each device's share of the key-value cache fits alongside its share of the
  weights.

None of these are hypothetical. The pitfalls and flags sections above record what
each of them looks like when it goes wrong.
