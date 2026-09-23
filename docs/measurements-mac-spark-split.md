# Splitting Mac + Spark: the full measurements

Every table behind the README's "when splitting helps" summary: the prefill crossover sweeps, generation, the 186 GiB capacity run, and what we could not test (17 Sep 2026, MacBook Pro M5 Max + one ASUS Ascent GX10 over 10 GbE, llama.cpp RPC, TCP).

> Moved verbatim from the [README](../README.md) on 23 Sep 2026 to keep the README short.
> Nothing was cut. Where the text says "above", "below" or "this README", it refers to
> the README as it was before the move; the [README's last section](../README.md#details-and-all-numbers)
> lists where each part now lives.

## Splitting Mac + Spark: when it helps, when it does not (17 Sep 2026)

The sections above compare the two machines running *alone*. This one joins them with
`ggml-rpc-server` over the 10 GbE cable and asks the only question that decides whether the
cable is worth having: **does splitting a model across both boxes beat just using the better
one?**

![Mac + Spark: when does splitting a model across two machines make it faster?](../benchmarks/split-when-useful-2026-09-17.png)

**The public summary** we posted on 17 Sep 2026
([@petruspennanen](https://x.com/petruspennanen)), quoted here as the post rather than as this
section's conclusion:

> Prefill is faster than either machine alone once the prompt reaches roughly 700-2000 tokens,
> depending on the model. Generation never beats the faster machine alone, because layer split
> makes the two machines wait for each other. Speeding up generation needs tensor parallelism,
> which already works between two Sparks; whether it can be made to work Mac-to-Spark is open.

**What that compresses, stated precisely.** "Faster than either machine alone" holds above the
crossover for the two models tested, and the crossover differs between them. "Never beats the
faster machine" is a statement about the **three split configurations we measured**, not about
layer split in general. "Because layer split makes the machines wait" is our *explanation* for
the slowdown, not something these measurements isolate: we measured the slowdown, not its cause.
And tensor parallelism is **one route** to faster generation, reported by others on two Sparks
under their own configurations, untested by us on any pair. Each of these is unpacked below.

### Prefill: the split wins, but only past a crossover

Six passes, every configuration measured back to back, the order rotated three ways so drift
shows up inside the data rather than between runs. **`-ts` lists the RPC device first**, so
`50/50` is half the layers on the GB10.

Dense **Qwen3.8-27B UD-Q4_K_XL** (16.34 GiB, fits either box), split ÷ the better single machine:

| prompt tokens | 128 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|
| split ÷ GB10 | 0.807 ± 0.013 | 0.855 ± 0.034 | **1.102 ± 0.044** | **1.324 ± 0.050** | **1.391 ± 0.045** |
| passes the split won | 0/6 | 0/6 | 6/6 | 6/6 | 6/6 |

A second, denser sweep located the crossover (four passes, order rotated):

| prompt tokens | 640 | 768 | 896 | 1024 | 1280 |
|---|---:|---:|---:|---:|---:|
| split ÷ GB10 | 0.935 | 1.024 | 1.062 | 1.129 | 1.197 |
| passes the split won | 0/4 | **3/4** | 4/4 | 4/4 | 4/4 |

What the four passes support, stated as sample facts rather than as a crossing point:
**640 was below parity in all four passes; 768 was near parity; 896 was above parity in all four.**
**These samples do not resolve an exact crossing.** The mean at 768 is already above 1.0, so the
single losing pass there does not push the underlying crossing any higher — it only means 768 is
not far enough above parity for four passes to separate it from a tie.

Sparse **Qwen3.8-Flash-Next UD-IQ4_XS** (87.24 GiB, 177 B total / 3 B active, also fits either
box). Here the Mac is the machine to beat, and parity arrives much later. Only three lengths were
tested, so this is a coarser picture than the dense sweep:

| | 512 | 2048 | 4096 |
|---|---:|---:|---:|
| split ÷ Mac | 0.722 | 1.003 | **1.128** |

**2048 is near parity, not a resolved crossover** — 1.003 is indistinguishable from a tie at this
sample size, and the nearest tested points either side are 512 and 4096. What we can say is that
the split is clearly behind at 512 and clearly ahead at 4096.

So the "700-2000 tokens" range in the post is a range *across models*, not a window that closes:
the dense model is at parity around 768 and clearly above it by 896, the sparse one is at parity
around 2048, and past parity the advantage kept growing over the lengths we tested rather than
peaking.

### Generation: no split we tested won

Four passes, order rotated four ways, on an idle machine (the prefill numbers above were taken
with two large downloads running, so absolute rates are not comparable across the two panels;
in-pass ratios are).

Dense 27B, tg64, tok/s:

| configuration | tok/s | ÷ Mac | ÷ GB10 |
|---|---:|---:|---:|
| Mac alone | **27.05 ± 0.21** | 1.000 | 2.209 |
| split 15/85 | 21.79 ± 0.29 | 0.805 | 1.779 |
| split 50/50 | 16.71 ± 0.04 | 0.618 | 1.365 |
| GB10 alone | 12.25 ± 0.00 | 0.453 | 1.000 |

Flash-Next, tg32, tok/s: Mac **41.3**, GB10 29.4, split 50/50 **27.1** — here the split is slower
than *both* machines, not merely slower than the faster one.

**The careful statement: none of the three splits we measured beat the faster single machine.**
On the dense model the split still beats the GB10 (1.78× at 15/85), so "slower than either machine"
would be wrong there; on Flash-Next it happens to be true. Three configurations at one context
length each is the whole basis for this — we did not sweep the split ratio, the context length or
the batch size, so read it as "these splits, on these two models" rather than as a property of
layer split.

*Our explanation, which these runs do not verify:* a layer split is pipeline-shaped, so each token
walks the layers in order and one box is idle while the other computes; prefill hides this because
a long prompt gives both boxes a large batch of independent work. **We measured the slowdown, not
the idle time.** Nothing here instruments where the wall-clock actually goes, so the waiting
account remains a hypothesis. Separating it would need per-stage timing or a profile, and that run
has not happened.

### Capacity: the case where the ratio does not exist

**GLM-5.3-Flash UD-Q4_K_XL, 185.98 GiB, 320.76 B params.** Reported device budgets are 107.5 GiB
on the Mac and 121.6 GiB on the GB10, so this file **fits neither box alone**. There is no
single-machine baseline, and therefore no speedup to quote:

| test | mean ± s.d. (n=3) | CV |
|---|---:|---:|
| pp512 | 315.42 ± 2.94 | 0.9 % |
| pp2048 | 424.52 ± 5.53 | 1.3 % |
| tg32 | 14.03 ± 0.46 | 3.3 % |

Three passes of the identical command, `-ts 50/50`, `-r 2`. The ± is an across-pass sample s.d.,
matching the other tables here. *Caveat the ± hides:* tg32 rises monotonically across the three
passes (13.69 → 13.84 → 14.55). Three points cannot separate drift from noise, but that is the
shape drift makes, so treat the generation figure as softer than its CV suggests; prefill shows no
such ordering. `-ts 40/60` fails outright with a Metal OOM.

**For this file on these two machines the cable does not buy speed, it buys the model running at
all.** That is a different claim from the prefill speedup above. It is also specific to this pair:
the quant exceeds both device budgets here, which says nothing about hardware we did not test.

### What we could not test

Tensor parallel shares each token's computation across devices instead of handing whole layers to
one box, which is **one route** to faster generation. **We could not test it on this pair: the
stack we used, llama.cpp over RPC, refuses it** — `-sm row` reports "device RPC0 does not support
split buffers". That is a statement about llama.cpp's RPC backend, not about every engine.

**What others report between two DGX Sparks**, given as their configurations rather than as a
result of ours. We did not run these and cannot vouch for them:

- An [NVIDIA developer-forum study](https://forums.developer.nvidia.com/t/comprehensive-qwen3-8-27b-study-on-dgx-sparks-quantization-speculative-decoding-and-tp-dp-scaling/381102)
  on Qwen3.8-27B holds one setup fixed and changes only the parallelism: "Moving the same InferAct
  + MTP setup from TP=1 to TP=2 raises C1 TPS from 18.5 to 23.4." That is **≈1.27× at single
  concurrency**, with speculative decoding on in both arms so the change is attributable to TP.
- [Flowtivity](https://flowtivity.ai/blog/deepseek-v4-flash-1m-context-dual-dgx-spark/) report
  41 tok/s on two Sparks for DeepSeek V4 Flash (284 B MoE) at FP8 dense + MXFP4 experts, vLLM
  0.21.1rc1.dev339, tensor parallelism 2, MTP with 2 speculative tokens, max 6 concurrent
  sequences, against "12-15 tok/s" for one Spark. **That pair is not a controlled comparison**:
  the single-Spark figure is a different quantisation (IQ2_XXS) and the dual-Spark figure adds
  speculative decoding, so the ~3× gap is not an isolated tensor-parallel gain and should not be
  quoted as one.

Take the ≈1.27× as the figure with a controlled comparison behind it. An earlier draft of this
section also cited a 0.85-1.01× pipeline-parallel range on matched Sparks; we could not locate a
primary source for it on re-checking and have removed it.

Mac-to-Spark is a different matter again: **the stack we tested does not support it**, and we are
not aware of one that does, which is a weaker claim than saying none exists.
[Ash Hart's MCDMA](https://github.com/ashhart/MCDMA) is building the RDMA transport such a thing
would need and has demonstrated a prefill/decode hand-off over it, but its README lists tensor
parallelism among planned experiments rather than finished ones.


**Moved:** the open question on whether an ADT-Link Gen4 adapter can replace the Helios 5S, which sat here, is now in [interconnect.md](interconnect.md#open-question-can-an-adt-link-gen4-adapter-replace-the-helios-5s).

One thing this section does **not** establish: that the 10 GbE link is the bottleneck. We never
measured inference wire utilisation, only a bulk file-copy rate, so nothing here justifies buying
a faster interconnect to fix a limit we have not demonstrated. Engine support is the part we can
point at concretely.

**Figure note:** the CAPACITY panel of the image above quotes the original single GLM run
(312 / 424 / 13.7) because it was drawn before the repeat. The n=3 table in this section
supersedes it.

Every number in this section is reproducible from
[`benchmarks/split-2026-09-17/`](../benchmarks/split-2026-09-17/): the per-pass datasets
(`interleaved3.jsonl`, `dense.jsonl`, `gen.jsonl`, `fnsplit.jsonl`, `glm-split.jsonl`), the raw
llama-bench logs, the runner scripts, and [`DATASETS.md`](../benchmarks/split-2026-09-17/DATASETS.md)
explaining what each file is and which conditions differ between them.
[`runners/chart_dark.py`](../benchmarks/split-2026-09-17/runners/chart_dark.py) draws the figure above
from those datasets. Two files are kept deliberately even though they are superseded: an aborted
fixed-order run, and the sweep JSON behind an outlier we could not reproduce (see
[`PROVENANCE.md`](../benchmarks/split-2026-09-17/PROVENANCE.md)).

Builds: Mac `1d0c76f3c` (Metal), GB10 `434ddbbc0` (CUDA 13), ggml 0.23.0 — the two ends are
different commits, which is stated here because it is a limitation of every cross-machine row.
