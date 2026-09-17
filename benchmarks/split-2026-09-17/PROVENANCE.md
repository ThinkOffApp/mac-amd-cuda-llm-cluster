# Qwen3.8-27B layer-split runs, 17 Sep 2026 01:24-01:25 CEST

Recorded because a second, EARLIER split of the same model family exists from the same day
(16 Sep 18:09Z) with different numbers. Both are valid; they used different configurations.
This file exists so neither overwrites the other.

## This run (01:24-01:25 CEST, 16 Sep 23:24Z)

- **Model**: `/Volumes/t705/ModelArchive/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_XL.gguf`
  17,548,181,504 bytes, 27,320,697,856 params, reported by llama-bench as 16.34 GiB Q4_K - Medium
- **Mac binary**: `~/llama-glm5/build/bin/llama-bench`, build_commit **1d0c76f3c**, build_number 753, Metal
- **Spark**: `~/llama.cpp/build-rpc/bin/ggml-rpc-server -H 10.10.10.2 -p 50052 -d CUDA0`,
  build **434ddbbc0** (10884), CUDA 13, NVIDIA GB10. Server was already up (started 20:12 CEST, uptime
  05:13 at bench time), i.e. NOT restarted between runs.
- **Link**: direct 10 GbE, Mac `en12` 10.10.10.1 <-> Spark 10.10.10.2. ICMP RTT 0.976 ms avg (3 pings).
- **llama-bench parameters** (defaults confirmed from `-o json`, not assumed):
  `n_batch 2048, n_ubatch 512, n_threads 6, type_k f16, type_v f16, flash_attn -1,
   n_gpu_layers -1, split_mode layer, main_gpu 0, use_mmap default`
- **`-ts` device order**: the RPC device is listed FIRST. `-ts 50/50` = 50 % Spark / 50 % Mac;
  `-ts 15/85` = 15 % **Spark** / 85 % **Mac**.

Exact commands:

    # solo Mac baseline (01:0x)
    ~/llama-glm5/build/bin/llama-bench -m <Q4_K_XL> -p 512,2048 -n 64 -r 2

    # solo Spark baseline (over ssh)
    LD_LIBRARY_PATH=$HOME/llama.cpp/build-rpc/bin $HOME/llama.cpp/build-rpc/bin/llama-bench \
      -m ~/models/qwen38-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf -p 512,2048 -n 64 -r 2 -ngl 99

    # splits
    ~/llama-glm5/build/bin/llama-bench -m <Q4_K_XL> --rpc 10.10.10.2:50052 -ts 50/50 -p 512,2048 -n 64 -r 2
    ~/llama-glm5/build/bin/llama-bench -m <Q4_K_XL> --rpc 10.10.10.2:50052 -ts 15/85 -p 512,2048 -n 64 -r 2

Results (t/s, mean +- stddev over r=2):

| test   | Mac solo | Spark solo | split 50/50 | split 15/85 |
|--------|---------:|-----------:|------------:|------------:|
| pp512  |   715.06 |     849.32 |      756.75 |      720.21 |
| pp2048 |   648.40 |     858.37 |     1156.35 |      745.53 |
| tg64   |    25.10 |      12.84 |       16.96 |       22.42 |

## The earlier run, same day (16 Sep 18:09Z = 20:09 CEST)

- **Model**: `~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf`, **29 GB** — a DIFFERENT quant
- **Both ends**: llama.cpp **434ddbbc0** (`~/llama.cpp/build-rpc` on the Mac, not llama-glm5)
- **Flags**: `-p 512 -n 128 -r 3`, **no `-ts`**, i.e. llama.cpp's DEFAULT placement
- Results: Mac only **705.5 +- 1.4 / 15.0 +- 0.26**; split **612 +- 65 / 9.67 +- 0.02**
  (prompt -13 %, generate -35 %)

## Reconciliation — the two runs do not contradict each other

1. **Different quant.** Q8_K_XL is ~1.8x the bytes of Q4_K_XL, and decode is bandwidth-bound, so the
   Q8 solo tg of 15.0 against the Q4 solo tg of 25.10 is the expected direction and rough magnitude.
   Prefill is far less sensitive to quant, and indeed the two solo pp512 figures nearly coincide:
   705.5 (Q8) vs 715.06 (Q4).
2. **Different placement, and this is the whole prefill difference.** The earlier run used llama.cpp's
   DEFAULT split; this one set `-ts 50/50` explicitly. Default gave pp512 **-13 %**; explicit 50/50 gave
   **+6 %**. That is the same effect the m5-squared table already documents for the Strix pair
   ("the default placement is the slow one"), now reproduced on the Spark pair.
3. **Decode agrees across both.** -35 % (Q8, default) and -32 % (Q4, 50/50). Decode loses on a layer
   split regardless of quant or placement.
4. **pp2048 has no earlier counterpart.** The 18:09Z run only measured `-p 512`. The 1156.35 figure is new
   and unreplicated by the earlier configuration.

## What these numbers do NOT establish

- **They do not decompose time into network waiting versus GPU work.** The pipeline-filling explanation for
  the pp2048 gain is a HYPOTHESIS. Proving it needs a synchronized trace on both ends, which has not been
  run. Do not present the mechanism as measured.
- `-r 2` is two repetitions; the 18:09Z run used three. Small samples, and the earlier split's prompt
  figure carried a +-65 spread.
- Only one model, one architecture, one link.
- Tensor parallel was separately shown impossible on 434ddbbc0: `-sm row` fails with
  "device RPC0 does not support split buffers". These are layer splits only.

---

# ANOMALY: the split's pp2048 depends on the benchmark composition (01:31-01:39)

Three measurements of the SAME placement, model and binary, differing only in which other prompt
sizes shared the `llama-bench` invocation:

| when  | command (`-ts 50/50`, `--rpc 10.10.10.2:50052`)          | reps | pp2048  |
|-------|----------------------------------------------------------|-----:|--------:|
| 01:24 | `-p 512,2048 -n 64`                                        |  r=2 | 1156.35 |
| 01:38 | `-p 512,2048 -n 64`  (exact rerun)                         |  r=2 | 1089.55 |
| 01:31 | `-p 128,256,512,1024,2048,4096 -n 64`                      |  r=3 |  753.20 |

The two short-ladder runs agree within 6 %. The long-ladder run is ~30 % lower. **Both are real
measurements; neither is discarded.** Mac solo over the same three runs: 648.40 / 595.01 / 610.80 at
pp2048, an 8 % spread, i.e. the solo path does NOT show the same effect, so this is not simply "the
machine got slower".

Full sweep (r=3, `-p 128,256,512,1024,2048,4096`), prefill t/s:

| config    |   128 |   256 |   512 |  1024 |  2048 |  4096 |  tg64 |
|-----------|------:|------:|------:|------:|------:|------:|------:|
| Mac solo  | 517.9 | 685.3 | 713.0 | 654.7 | 610.8 | 552.2 | 21.46 |
| GB10 solo | 751.9 | 835.2 | 839.3 | 842.2 | 841.2 | 841.6 | 12.50 |
| 50/50     | 590.8 | 683.2 | 656.7 | 706.1 | 753.2 | 750.3 | 16.01 |
| 25/75     | 553.9 | 695.5 | 533.1 | 532.6 | 535.4 | 548.6 | 17.59 |
| 15/85     | 528.5 | 641.8 | 498.4 | 474.2 | 512.4 | 490.2 | 17.12 |

Caveat on all of the above: two large downloads (Flash-Next to the T705, GLM Q6 to the Spark) were
running throughout, so both machines were under load.

## What survives regardless of the anomaly

**The GB10 is flat with context and the Mac is not.** 835-842 t/s from 256 to 4096 on the GB10, against
the Mac peaking at 713 (512) and falling to 552 (4096). Reproduced in every run tonight.

## NOT publishable until resolved

- Any single number for split pp2048, hence the "split beats both boxes" claim.
- The **PROVISIONAL** routing rule G ~ 0.035 x P (73 generated tokens at a 2048 prompt): it is computed
  from the disputed 1156 row, and separate steady-state pp and tg rates are a SCREENING estimate, not
  request latency - a real request also pays load, placement and switch costs that appear in neither.

## The experiment that would settle the policy (codexmb's spec, not yet run)

End-to-end prompt+generation timings at several (P, G) pairs, including placement/load/switch cost, with
matched solo and split repetitions INTERLEAVED so drift is visible inside the data rather than between
runs. Plus the isolation currently running: pp2048 alone, then with 512, then a full ladder, then pp2048
alone again.
