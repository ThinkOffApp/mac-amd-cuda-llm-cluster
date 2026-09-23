# NVIDIA GB10 vs Apple Metal

The same GGUF files on the MacBook Pro M5 Max (Metal) and an ASUS Ascent GX10 (GB10, CUDA), 16-17 Sep 2026: dense and sparse models, the context sweep, cache sizes and the 10 GbE file-copy rate.

> Moved verbatim from the [README](../README.md) on 23 Sep 2026 to keep the README short.
> Nothing was cut. Where the text says "above", "below" or "this README", it refers to
> the README as it was before the move; the [README's last section](../README.md#details-and-all-numbers)
> lists where each part now lives.

## A third backend: NVIDIA GB10 vs Apple Metal (16-17 Sep 2026)

An ASUS Ascent GX10 (**NVIDIA GB10**, compute capability 12.1, 124,544 MiB, CUDA 13)
joined the bench over 10 GbE. The same GGUF file was copied to both machines and run with the same `llama-bench` flags,
so there are no quant or model differences between the columns. Two things do still differ
besides the silicon: the backend (Metal vs CUDA) and the llama.cpp build, which is six days
apart (Mac `1d0c76f3c`, GB10 `434ddbbc0`, same ggml 0.23.0). Read the columns as
"this file on this machine's stack", not as a pure silicon comparison.

**Dense — Qwen3.8-27B UD-Q4_K_XL, 16.34 GiB, 27.32 B params**

| test | Mac M5 Max (Metal) | GB10 (CUDA) | winner |
|---|---:|---:|---|
| pp512 | 715.06 | **849.32** | GB10 ×1.19 |
| pp2048 | 648.40 | **858.37** | GB10 ×1.32 |
| tg64 | **25.10** | 12.84 | Mac ×1.95 |

**Sparse — Qwen3.6-35B-A3B UD-Q3_K_S, 14.29 GiB, 34.66 B total / 3 B active**

| test | Mac M5 Max (Metal) | GB10 (CUDA) | winner |
|---|---:|---:|---|
| pp512 | **3240.20** | 2439.58 | Mac ×1.33 |
| pp2048 | **3080.27** | 2455.70 | Mac ×1.25 |
| tg64 | **97.67** | 68.37 | Mac ×1.43 |

**On these two models, the GB10's prefill lead appears on the dense one and reverses on the
sparse one.** On the dense 27B the GB10 prefills 1.32× faster while the Mac generates nearly
twice as fast — the familiar compute-versus-bandwidth split, and the reason a prefill-there /
generate-here arrangement looks attractive. On the sparse 35B-A3B, a *larger* model by total
parameters, the Mac wins both halves.

Two configurations are not a law. What we have is one dense pair and one sparse pair, on one
quant each, on two llama.cpp builds. The obvious *hypothesis* is that with 3 B of 34.66 B
parameters active, prefill stops being dominated by dense matmul and becomes routing and
gather work, which would favour unified memory — but nothing here measures that mechanism,
and we have not varied sparsity, quant or build independently.

Why it still matters for this repo: the models we are aiming at in the 100 GB class are all
MoE — GLM-5.3-Flash, DeepSeek V4.1, Qwen3.8-Flash-Next (512 experts, 10 active). If the
reversal holds for them, the GB10 is the slower box at both halves for exactly the workloads
this repo exists to run. That is the next measurement, not a conclusion.

**The reversal does hold at the 100 GB class (measured 17 Sep 02:47).** The open question above is
now answered for the size this repo actually targets. `Qwen3.8-Flash-Next UD-IQ4_XS`, 87.24 GiB on disk,
**176.94 B parameters with 3 B active**, four passes with both machines measured back to back and the
order swapped each pass:

| test | Mac M5 Max (Metal) | GB10 (CUDA) | Mac ÷ GB10 |
|---|---:|---:|---|
| pp2048 | **1062.5 ± 1.7** | 859.6 ± 2.3 | **1.236 ± 0.004** |
| tg64 | **41.68 ± 0.44** | 29.74 ± 0.05 | **1.401 ± 0.015** |

(± is one sample s.d. across the four passes, n=4, not a confidence interval.)

So on a 177 B sparse model the laptop is 24 % faster at prefill and 40 % faster at generation than the
GB10. Two things fall out of the absolute numbers that are worth more than the ratio:

- **41.7 tok/s of generation on a 177 B model**, on a laptop, because only 3 B parameters are active —
  faster than the same machine manages on the *dense* 27 B (25 tok/s).
- **Both machines were then measured over the full 512 → 4096 range, and the result reversed an earlier
  claim of ours.** Means of 3 order-rotated passes:

  | | 512 | 1024 | 2048 | 4096 | change |
  |---|---:|---:|---:|---:|---:|
  | Mac, Flash-Next | 1076.1 | 1060.9 | 1025.3 | 971.3 | **−9.7 %** |
  | GB10, Flash-Next | 828.3 | 847.2 | 840.2 | 826.6 | **−0.2 %** |
  | Mac, dense 27 B | 713.0 | 654.7 | 610.8 | 552.2 | **−22.6 %** |

  An earlier revision of this section said the Mac's prefill was "flat" on Flash-Next. That was drawn from
  a single run sampled only to 2048, and it was wrong: extended to 4096 with three passes, the Mac declines
  about 10 %. What actually holds across both models is the *original* observation — **the Mac's prefill
  decays with context and the GB10's does not** — with the magnitude depending on the model (−22.6 % dense,
  −9.7 % sparse) rather than the direction. We also previously said this model showed no run-to-run drift;
  it shows less, not none (one of the three passes came in ~7 % low at 2048 and 4096).

*The control we have not run.* Both columns are llama.cpp, so nothing here separates "the
GB10 loses on this workload" from "llama.cpp's CUDA path loses on this workload" — its
mixture-of-experts path may simply be less mature than its Metal one. The separator would be
the same model on a native NVIDIA stack (TensorRT-LLM, or a modelopt-aware vLLM), and **that
run has not happened.** Until it does, read every sparse row here as a statement about this
stack: "llama.cpp on this hardware", not "this hardware".

**Approximate effective weight-read rate** (weight bytes × tg64 on the dense model):
≈440 GB/s on the M5 Max, ≈225 GB/s on the GB10. This is not measured DRAM bandwidth. It
assumes a decode reads each weight exactly once per token and counts nothing else — no
activations, no KV traffic, no cache hits or repeated reads — so treat it as a rough
workload-normalized proxy for comparing the two machines, not as a bound on DRAM bandwidth
in either direction.

**Cache sizes, read from llama.cpp's own allocator rather than computed** (Qwen3.8-27B,
`-c 4096`): KV cache 256.00 MiB over 16 full-attention layers = exactly 64 KiB/token, plus a
`llama_memory_recurrent` block of 149.62 MiB that is **fixed in sequence length** (R f32 5.62,
S f32 144.00) across all 64 blocks. The model is 16 full-attention + 48 linear-attention
layers: the 48 linear layers use a fixed-size recurrent state, while total state still grows
with the 16 KV layers. At the `-c 4096` shown, the variable part is already the larger of the
two (256.00 MiB KV against 149.62 MiB recurrent); they cross at roughly 2,400 tokens.

**Link**: a 17,559,178,144-byte model file copied Mac → GB10 over 10 GbE in 15 s =
1.171 GB/s (1116 MiB/s) = **9.36 Gbit/s**; a 15.36 GB file measured 8.77 Gbit/s. Plain
`cat | ssh`, no compression. Decimal GB and binary MiB throughout, which is worth stating
because mixing them turns the same measurement into 8.93 Gbit/s.


## How the GB10 came into this repo

From the README introduction.

All three backends now appear here, which is why the repo is no longer called
`mac-amd-llm-cluster`. The GB10 first appears below as a straight two-way comparison on
identical files, and then split across the cable with the Mac: when that split is worth
doing, when it is not, and the 186 GiB quant that exceeds either device budget and ran split
across both.
