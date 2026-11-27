# TensorFold and Inkling 2 Mini on the RTX 5090 dock: 27 November 2026

Second day with the card. This page continues
[measurements-2026-11-26-rtx5090-egpu.md](measurements-2026-11-26-rtx5090-egpu.md). On the first evening one
TensorFold kernel ran on the card. On this day TensorFold's CUDA server ran a whole model on it, from a Mac,
and we measured it with and without other people's speed work. Everything here is one day, one card, one cable.
The list of what is not done is at the end.

## Whose work this is

Most of what makes the numbers below possible is not ours.

- **Ash Hart: [TensorFold](https://github.com/ashhart/TensorFold)**, the inference engine. We reported a first
  result on his [RFC #4186](https://github.com/ashhart/TensorFold/issues/4186). The maintainer said yes
  in principle to a macOS CUDA target behind an explicit option, off by default, and wants the kernel contract
  and a comparison of token hashes against Linux first. Neither exists yet (see [Not done](#not-done)).
- **Mia (MiaAI-Lab):
  [Inkling-2-Mini-RTX-5090-TensorFold](https://github.com/MiaAI-Lab/Inkling-2-Mini-RTX-5090-TensorFold)**,
  Apache-2.0. 28 patches on `jschmied/TensorFold@dc2ca8a1785df6ee5c8489ee7cd18e29cf7b4568`. The speed patches
  are hers. Sections D to H use her recipe.
- **incoai: the draft model `incoai/Inkling-2-Mini-DFlash2`**, revision
  `4f9088ea66c21f2b99ed657d1941c306b03e84d2`, Apache-2.0. Every DFlash2 row below uses it.
- **Davinchy: [macuda](https://github.com/Davinchy/macuda)**, commit `e797e31`, the user-space NVIDIA driver
  for macOS. Every run on this page goes through it.
- **Thinking Machines: the model `thinkingmachines/Inkling-2-Mini`.**
- **satellitedown: ingine.** We did not run it. Its figures appear below only as its README states them.

What is ours: the measurements, the build of TensorFold's CUDA server for macOS, and a translation library
between TensorFold and macuda's driver, described next.

## Setup

- **Host:** MacBook Pro, Apple M5 Max, 128 GB, macOS 27.0.1.
- **Card:** GeForce RTX 5090, 32 GB, in a Minisforum DEG2 dock on Thunderbolt 5. The link reported 80 Gb/s.
- **Driver:** macuda `e797e31`.
- **Translation library (ours, not released yet):** a single C file that exports the 77 CUDA driver API
  functions TensorFold resolves, built as a Mach-O library under the file name `libcuda.so.1` on top of
  macuda's `libtinynv`. 73 of the 77 do real work, including CUDA graphs, which it handles by recording the
  graph and replaying it. 4 refuse by name: host memory register, host memory unregister, host memory device
  pointer, and event elapsed time. On 26 November the count was 58 working and 19 refusing. It is not
  published. Until it is, nobody else can repeat the Mac side of this page.
- **TensorFold on macOS:** the CUDA server builds for macOS with a one-line change, the operating system gate
  for `-Dgpu` in `zig/build/product.zig`. Build: `zig build -Dgpu=sm_120 -Dfatbins=DIR` with Zig 0.17.0. Run
  with `TENSORFOLD_DEVICE_CHECK=0`.
- **Kernel images:** compiled on an ARM64 Linux machine with nvcc, first nvcc 13.0.88, later nvcc 13.3.73 from
  the `nvidia/cuda:13.3.1-devel` image. Each section says which.
- **Local workaround, cluster launch:** the Mac driver has no thread-block cluster launch. One line in
  `zig/src/cuda/qmmf.zig` forces the path used before sm_90 (`.clusters = capability_major >= 1000`). This is
  a local workaround, not an upstream change. Sections C, D and E ran with it. Section B did not need it.
- **With a drafter**, `TENSORFOLD_MEMORY_RESERVE_GIB=3` was needed.

## How the runs were made

Unless a section says otherwise: 27 November 2026, one stream, greedy decoding, Inkling 2 Mini, text only, the
whole model on the 5090, 200 generated tokens. The first request is a warm-up and is not counted. Then three
timed runs. A range such as "74.5 to 75.2" is the lowest and highest of those runs. A single figure is given
where only one was recorded.

"Identical" means byte-identical text between the runs being compared. It is a comparison of two runs on this
card through this driver. It is not a comparison against Linux.

Three prompts were used with TensorFold: a 36-token prose prompt, a code prompt, and a document reproduction
prompt, where the model is asked to write out a document it was given.

## A. llama.cpp through macuda, and splitting with the M5 Max CPU

Not TensorFold. Llama 3.1 8B Q8, llama.cpp's CUDA backend through macuda, 256-token prompt, 128 generated
tokens. With all 33 layers on the card: 158.0 to 158.6 tok/s, the same as on 26 November.

Moving layers from the card to the M5 Max CPU:

| Layers on the card, of 33 | Generation (tok/s) | Prompt (tok/s) |
|---|---|---|
| 33 | 158.4 | 9,329 |
| 32 | 131.3 | 3,968 |
| 24 | 68.7 | 713 |
| 16 | 44.5 | not recorded here |
| 8 | 27.4 | not recorded here |
| 0 | 21.9 | not recorded here |

One layer off the card costs 17% of generation speed and more than half of prompt speed. On this route every
layer that leaves the card makes it slower.

## B. Upstream TensorFold, no drafter

36-token prose prompt.

| TensorFold | Generation (tok/s) |
|---|---|
| 1.3.5 | 44.5 to 44.7 |
| 1.3.6 | 74.5 to 75.2 |
| main `ac4cb4f53` | 74.5 to 75.2 |

The text was the same from all three. On 1.3.5 we counted 1,446 kernel launches per token. For scale, the M5
Max in the same MacBook through Metal does 18 tok/s on that prompt.

## C. Upstream main with the drafter of pull request #5197

Upstream main plus the drafter of [ashhart/TensorFold#5197](https://github.com/ashhart/TensorFold/pull/5197),
with the cluster-launch workaround from [Setup](#setup).

| Prompt | With drafter (tok/s) | Plain (tok/s) | Note |
|---|---|---|---|
| Prose | 94.4 | 74.5 to 75.2 (section B) | 87 of 138 drafted tokens accepted |
| Code | 103.4 | 69.3 | |
| Document reproduction, copy drafts | 216.4 | 65.3 | |

The drafted text was identical to the undrafted text each time.

## D. Mia's recipe, 16-bit settings

Her 28 patches, with her 16-bit settings: bf16 KV cache, and `TF_INKLING2_ATTN_FP8`, `TF_INKLING2_PREFILL_TC`
and `TF_INKLING2_ATTN_FP8_TC` unset.

First with kernels we built with nvcc 13.0.88 and without `-Dnative-fp-decode`, which is not how her recipe
builds them (see [section F](#f-a-build-trap-our-mistake)). Then rebuilt her way, with nvcc 13.3.73 and
`-Dnative-fp-decode=true`.

| Drafter | Prompt | nvcc 13.0.88, no native-fp-decode (tok/s) | nvcc 13.3.73, native-fp-decode (tok/s) |
|---|---|---|---|
| none | prose | 120.8 to 121.2 | 122.6 |
| DSpark | prose | 161.6 to 162.2 | not recorded here |
| DFlash2 | prose | 208.1 to 209.6 | 211.7 to 214.7 |
| DFlash2 | code | 345.6 to 346.1 | 352.3 to 352.6 |
| DFlash2 | document reproduction | 380.7 to 381.1 | 386.6 to 387.7 |

DSpark is the drafting head that ships inside the model checkpoint. DFlash2 is incoai's separate draft model.

Time to first token on the rebuilt kernels: 0.03 to 0.06 s. The drafted text was identical to the undrafted
text in every case.

These are 16-bit figures. Her default is the 8-bit configuration in the next section.

## E. Mia's default 8-bit settings, and prompt reading

Her defaults: `--kv-cache fp8` and the three switches from section D turned on. Rebuilt kernels (nvcc 13.3.73,
`-Dnative-fp-decode=true`), DFlash2 drafter.

| Prompt | 8-bit (tok/s) | 16-bit, from section D (tok/s) |
|---|---|---|
| Prose | 199.5 to 206.7 | 211.7 to 214.7 |
| Code | 293.5 to 293.8 | 352.3 to 352.6 |
| Document reproduction | 164.3 to 169.5 | 386.6 to 387.7 |

For generating, 8-bit was slower than 16-bit on this route, by a wide margin on document reproduction. We do
not know the cause. We have not measured it on Linux, so we cannot say whether it belongs to the recipe, to the
Mac driver route, or to something we did.

Prompt reading is where the 8-bit settings were faster. A fresh prompt on each run, so nothing is served from
cache. The figure is prompt tokens divided by time to first token.

| Prompt length (tokens) | Settings | Prompt reading (tok/s) | Time to first token |
|---|---|---|---|
| 5,246 | 8-bit | 13,673 to 13,734 | |
| 20,046 | 8-bit | 14,205 to 14,237 | 1.41 s |
| 100,046 | 8-bit | 11,602 to 11,617 | 8.6 s |
| 5,246 | 16-bit, rebuilt kernels | 10,660 to 10,687 | |
| 5,246 | 16-bit, nvcc 13.0.88 kernels | 10,259 to 10,314 | |

## F. A build trap, our mistake

With kernels built by nvcc 13.0 and without `-Dnative-fp-decode`, the `fp8_experts` kernel is a trap stub.
Setting `TF_INKLING2_PREFILL_TC=1` on that build stops the card, and the firmware reports an SM error. Built
with nvcc 13.2 or later and with the option, it runs.

This was our mistake, not a fault in the recipe. It is recorded because a too-old nvcc produces a build that
looks complete and then stops the card.

## G. Several streams

On this Mac route, more streams did not add throughput.

| Streams | Configuration | Total (tok/s) | One stream, same configuration (tok/s) |
|---|---|---|---|
| 11 | 16-bit, no drafter | 118 | 121 |
| 20 | 8-bit cache and attention, no drafter | 124 | not recorded here |
| 8 | DFlash2 | 170 | 209 at that time |

64 streams do not fit in 32 GB. A stream took about 0.9 GiB at 16-bit with the smallest window, 3,072 tokens,
and about 0.5 GiB at 8-bit.

On Linux the same patched build does batch across streams. On an NVIDIA GB10, which is not a 5090, one stream
ran at 30.1 tok/s and eight streams at 198.5 tok/s in total. So the flat result above points to the Mac
driver route, not to the patches. It has not been measured on a 5090 under Linux.

## H. Exactness, as far as tested (preliminary)

Not on the 5090. On the GB10 under Linux, raw prompts through `/v1/completions`, 200 greedy tokens:

- The patched build gave the same tokens as its base `dc2ca8a1` at seven prompt lengths from 12 to 4,962
  tokens.
- That base and upstream main `ac4cb4f53` agree on short prompts (12 and 484 tokens) and differ on long ones
  (4,962 tokens). Where they part depends on the text, around 490 to 500 tokens.

This is preliminary. Which of the 28 patches were in the tested tree is being re-checked. Until that is done,
read the first point as "a patched tree matched its base", not as a statement about the full recipe.

On the 5090 itself, the only exactness statements are the ones in sections B to E: text from one run compared
with text from another run on the same card.

## I. Other people's figures, not measured by us

These are claims from the projects' own READMEs. We did not reproduce any of them, and they were not made on
our hardware route.

**ingine**, from its README:

| What | Stated figure |
|---|---|
| One stream (chat, code, coding-agent replay) | 487 to 561 tok/s |
| Prompt reading at 20K tokens | 10,900 tok/s |
| Prompt reading at 64K tokens | 9,600 tok/s |
| Prompt reading at 131K tokens | 8,100 tok/s |
| Batch engine, 64 streams | about 4,200 tok/s in total |

ingine is not an exact-output engine and serves a modified model, so its figures and the TensorFold figures on
this page do not measure the same thing.

**Mia's README** states 150 to 230 tok/s for one stream on an RTX PRO 6000, and says the recipe had not yet
been measured on a 5090. Sections D and E are, as far as we know, a first 5090 measurement of it, made through
a Mac driver route she did not target.

## Not done

- No kernel-contract run. The maintainer wants it first.
- No comparison against a Linux sm_120 run of the same prompts on this card. Nothing on this page shows that
  the Mac route produces the tokens Linux produces.
- No measurement of ingine by us.
- Sleep, and unplugging and reconnecting the dock, are untested.
- Several streams on a 5090 under Linux are untested.
- Only Inkling 2 Mini has run end to end through TensorFold on this route. Qwen3.5-2B fails on a cluster
  launch.
- The translation library is not published.
- One day, one card, one cable. Nothing has been repeated on another day.
