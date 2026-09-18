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
