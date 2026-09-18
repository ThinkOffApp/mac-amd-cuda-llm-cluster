# tools

## `serve_bench.py`

Benchmarks an OpenAI-compatible serving endpoint (vLLM) **without counting stream
chunks as tokens**.

Counts come from the server: it diffs `vllm:prompt_tokens_total` and
`vllm:generation_tokens_total` across each request. It also measures the
chunks-per-token ratio explicitly and shouts if it is not ~1.

### Why this file exists

On 18 September 2026 a benchmark here counted SSE chunks with non-empty text and
called them tokens. The real ratio was **3.76 tokens per chunk**, so every decode
figure was low by that factor. Two conclusions were withdrawn: a "layer split is
4x faster than tensor parallel" comparison, and a predicted network throughput
ceiling that the corrected measurement exceeded by 3-5x. A downstream step-rate
table built on the same figure had to be withdrawn as well.

A chunk is a transport artifact. The server decides how many tokens to pack into
an SSE frame, the ratio is neither 1 nor stable, and nothing in the stream tells
you. Take the count from the server.

### Known limitation, stated rather than hidden

`TTFT` is **not** a clean prefill/decode boundary. When chunks are fat the first
one already carries several decoded tokens, so `prompt_tokens / TTFT` is an upper
bound on prefill time and therefore a **lower bound** on the prefill rate, not an
isolated measurement. Per-token timestamps are needed to split the phases
properly. Until that exists, do not publish a prefill/decode split from this tool.

```
python3 tools/serve_bench.py
```

## `autosplit.py`

Finds the layer-split ratio that is actually fastest, by measuring it.

```
python3 tools/autosplit.py --model /path/model.gguf --rpc 10.10.10.2:50052 \
    --client ~/llama.cpp/build-rpc/bin/llama-bench --prompt 2048
```

### Why it exists

`-ts` is a ratio you type, and llama.cpp's default is proportional to **device
memory**. In a pipeline the slowest stage sets the rate, so an even split hands
the same work to unequal machines. Measured on a MacBook Pro (Metal) + DGX Spark
(CUDA) pair with Qwen3.8-Flash-Next at pp2048:

| -ts (remote/local) | tok/s | vs the faster machine alone |
|---|---|---|
| 60/40 | 789.33 | **0.757x — slower than one machine** |
| 50/50 | 1086.65 | 1.042x |
| 43/57 | 1176.48 | 1.128x |
| **35/65** | **1282.93** | **1.230x** |
| 30/70 | 1217.64 | 1.167x |
| 25/75 | 1169.55 | 1.121x |
| 20/80 | 1115.66 | 1.069x |
| 15/85 | 1085.85 | 1.041x |

**62% spread from the ratio alone**, and the memory-proportional default region
is where the pair is *worse* than not splitting at all.

### Why golden-section rather than gradient descent

Each evaluation is a benchmark run of tens of seconds, the objective is noisy,
it is one-dimensional, and there is no gradient available. Golden-section search
needs no derivative and converges in a handful of evaluations. The measured
curve above is unimodal — a single peak with monotone flanks — which is the
precondition that makes the search valid. On a multi-peaked curve, scan instead.

Validated against that measured curve offline: the search finds 35/65 in 8
evaluations, the same answer the full sweep gives. If every benchmark run fails
it raises rather than returning a ratio.

### Known limit

**One ratio cannot optimise both phases.** Over the same sweep, prefill varied
by 62% and generation by 4.3% — generation is nearly flat and mildly prefers the
opposite end. Optimising for prefill is the right default because generation
barely cares, but do not present the result as optimal for both.

A naive speed-proportional formula (share ∝ measured solo speed) predicted 43/57
and left 9% on the table: the remote side also pays transfer cost, so it should
get less than its raw speed share. Measure, do not model.
