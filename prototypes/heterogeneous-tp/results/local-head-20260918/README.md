# Local Mac head experiment, 18 September 2026

Mac MPS solo only, FP32 GPT-2, source a98bb82. Five trials for each head mode
and prompt length, 64 generated tokens, two warmups before each recorded run.
Head order alternated by trial. Every recorded output matched the HuggingFace
reference. Raw timestamps, config, source hashes and outputs are in the JSONs.

| Prompt | Head | Median TTFT (s) | Median decode (tokens/s) |
|---|---|---:|---:|
| 128 | baseline | 0.02393 | 78.83 |
| 128 | last-root | 0.02086 | 74.45 |
| 512 | baseline | 0.03044 | 100.02 |
| 512 | last-root | 0.02971 | 104.97 |

This does not establish a reliable solo decode improvement. Trial spread is
large: p128 baseline 65.86-99.54, optimized 68.05-90.68; p512 baseline
85.40-108.78, optimized 89.07-112.88 tokens/s. Do not interpret a ratio of these
medians as a confirmed speedup or extrapolate it to a mixed-device pair.
No isolation of all background desktop activity was enforced. Read-only source
inspection and a source checkout took place during this run; no compiler or
other benchmark was launched by this task while measurements were running.

Solo does not exercise removing the duplicate head on rank 1 or changing when
the peer becomes ready. The physical-pair experiment remains necessary. The
optimization remains opt-in; it has not replaced the baseline.

For each prompt 128 and 512, repeat five trials in alternating head order:

```sh
GPT2_DIR=/path/to/pinned/gpt2 python bench.py --mode solo --device mps \
  --prompt-tokens 128 --new-tokens 64 --runs 1 --warmups 2 \
  --generation-head baseline --out /path/to/unique-trial.json
```

Use `last-root` for the other mode. `summary.json` includes individual sample
values and start/end timestamps. These are unprofiled runs.
