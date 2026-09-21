# Shared-model GGUF context probe

`gguf_contexts.cpp` tests three independent contexts sharing **one loaded model**.
The serial control calls decode and collects output for each context in turn. The
deferred variant submits all three initially, then collects one context's output
and immediately submits that context's next token. Each context owns its KV cache.
One host thread submits all work, avoiding concurrent access to RPC dispatch state.

This is a feasibility and correctness probe. It does not yet implement multiple
sequences per context, dynamic admission, stage balancing or a production scheduler.
It records host submit/collect intervals, which **do not prove GPU overlap**.
Timing includes prefill, argmax, output copying and event recording; it is diagnostic
wall time, not a production decode benchmark. It has one warmup per schedule and two
subsequent observations per schedule. The measurements are insufficient for a
reliable speedup claim, especially across backend configurations.

## Verified on current models

Engine `bdcbaaf6e7520b68c8c60ff724c67409970d70e1`, with the private patch archived beside
the results. The optional meta profiler was disabled. All tests used one Apple M5 Max
GPU; split cases use a separate Metal RPC process on loopback, still the **same GPU**.
Both solo and split use the identical model and three prompt/reference fixtures from
`engine_correctness.cpp`. All vocabulary logits are compared with the existing
atol=0.1, rtol=0.01 gate; observed error was actually zero throughout.

| Model | GGUF | Choices per run | Configurations | Result |
|---|---|---:|---|---|
| Gemma 4 E4B | Q4_K_M | 46, EOS-aware | Solo / layer split, BLAS present / excluded | Exact IDs and full logits |
| Qwen3.8-27B | UD-Q4_K_XL | 48 | Solo / layer split, BLAS present / excluded | Exact IDs and full logits |

Each configuration runs six schedule trials, including the two warmups: 48 runs total.
Full reports, source hashes, event intervals, compressed engine/server logs, exact
executed runners, checkpoint hashes and private engine patch are under
`results/gguf-contexts-20260918`. Initial-suite source is preserved separately because
the second suite added the explicit BLAS exclusion and capability reporting.

The first runner supplied a backend directory through `GGML_BACKEND_PATH`, which
logged a failed attempt to load the directory as a library. Device loading succeeded
through normal discovery, as confirmed by the model placement logs. The second
runner omits that invalid setting. Neither suite silently fell back to CPU model
placement; logs show layers assigned to MTL0 and RPC0 in split cases.

## Concrete scheduler gate found

At this engine revision, `llama-context.cpp` enables layer pipeline scheduling only
when every non-CPU backend supports async execution and events. The loop also visits
BLAS, an auxiliary ACCEL backend with both capabilities false. This disables the
scheduler's pipeline mode even though Metal and RPC advertise both capabilities.

Setting `GGUF_PROBE_NO_BLAS=1` makes this private driver unregister BLAS **before**
model/context creation. It does not change global configuration or production
services. Both modern models then log `pipeline parallelism enabled` for all three
split contexts, with exact reference correctness preserved. With BLAS present, no
context logs that message. This establishes the configuration gate, **not** a
performance gain or proof that stages overlap. Excluding BLAS can also change CPU
operation placement/performance, so a timing difference cannot be attributed solely
to pipeline activation.

Source inspection also finds synchronization at graph reuse and at scheduler input
copies. These remain possible barriers to overlap; removing a safety synchronization
without a buffer-lifetime/dependency design is not a valid optimization.

## Build and run

Build against the pinned private engine using its `include`, `ggml/include`, and
`vendor` headers; link `llama`, `ggml`, `ggml-base`, and `ggml-rpc` from `build-tp/bin`.
The archived runners record exact model/reference paths and RPC server invocation.
Create the solo reference directory with the existing `engine_correctness` helper
before invoking this driver; its model path, tokenizer IDs and vocabulary must match.

```sh
gguf-contexts MODEL.gguf layer 127.0.0.1:29591 REFERENCE_DIR report.json
GGUF_PROBE_NO_BLAS=1 gguf-contexts MODEL.gguf layer 127.0.0.1:29591 REFERENCE_DIR report-no-blas.json
```

A physical-pair test must use compatible RPC protocol/engine builds and the exact
checkpoint, verify device placement and correctness, measure actual overlap, and
include matched-workload solo and serial controls. These receipts do not settle
heterogeneous pipeline performance.
