# 27B GGUF engine gates, 18 September 2026

Engine: llama.cpp `bdcbaaf6e7520b68c8c60ff724c67409970d70e1`, locally built
with Metal and RPC. Model: Qwen3.8-27B UD-Q4_K_XL, architecture `qwen35`.
The current file is 17,559,178,144 bytes; SHA-256
`3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e`.
This current file identity takes precedence over older README file-size notes.

## Original failure and local experimental patch

Unmodified upstream ran Metal solo but aborted for Metal + loopback CPU RPC
with `-sm tensor -ts 1,1`. Diagnostic logging found a zero-element cache view
`cache_s_l62 (view)`, shape `[786432,0,1,1]`, whose computed address lay beyond
its owning buffer. The server correctly enforced its buffer-bounds assertion.

`rpc-empty-address.patch` canonicalizes the wire address of empty tensors to
their owning buffer's base. No tensor elements exist to read at that address.
It retains the server's size, overflow and bounds assertions. The included
extra logging records any future bounds failure. This is a LOCAL experiment,
not an upstream submission, general RPC fix claim or deployed server change.
The patch applies to the pinned checkout above; its diagnostic predecessor
and the failing server log are preserved for comparison.

## Numerical gate

`engine_correctness.cpp` compares three prompts, up to 16 new tokens each,
against unpartitioned Metal on exactly the same GGUF. For partitioned runs it
feeds reference tokens so each logit comparison uses the same context; all
48 token choices match in these runs. Full vocabulary logits are checked for
finiteness and against a tolerance declared before the run: atol 0.1, rtol 0.01.
Passing sampled token choices is not a general quality guarantee.

| Configuration | Token choices | Maximum absolute logit error | Full gate |
|---|---:|---:|---|
| Metal + loopback CPU RPC, layer split | 48/48 | 0.818263 | FAIL: logit tolerance |
| Metal + loopback CPU RPC, tensor split, patched | 48/48 | 0.635272 | FAIL: logit tolerance |
| Metal + loopback Metal RPC, tensor split, patched | 48/48 | 0.001158 | PASS |

The CPU-backend controls indicate backend numerics need separate treatment;
they do not justify loosening the threshold after seeing the errors. The
Metal/Metal test exercises real quantized model tensor splitting through RPC
on one physical GPU. It does NOT prove Metal/CUDA or Metal/ROCm correctness,
performance across machines, or useful parallel speedup. No elapsed times
from these cold-load correctness probes should be presented as benchmarks.
All temporary loopback RPC processes were stopped or observed to exit.

## Flash-Next

The existing Flash-Next files identify architecture `qwen4exp`. Unmodified
upstream rejects tensor mode with `LLAMA_SPLIT_MODE_TENSOR not implemented for
architecture 'qwen4exp'`. The runtime error and command are preserved. Its
architecture gate has NOT been bypassed. This remains outstanding engineering,
not a measured large-model tensor-parallel performance result. Header hashes
for its three files are in `model-files.json`; those are not full weight hashes.

## Reproduction

Build the pinned engine with `-DGGML_RPC=ON` and targets `llama-completion` and
`ggml-rpc-server`; apply the local patch for the tensor cases. Compile the helper
against that checkout's `include`, `ggml/include`, and `vendor` headers and
`llama`, `ggml`, `ggml-base`, and `ggml-rpc` libraries. Set the dynamic library
search path and `GGML_BACKEND_PATH` to the build's `bin` directory.

```sh
engine_correctness /path/to/model.gguf solo 127.0.0.1:29591 /out/solo /out/solo
# In a separate process on the SAME Mac, bind only loopback:
ggml-rpc-server -d MTL0 -H 127.0.0.1 -p 29591 -t 4
engine_correctness /path/to/model.gguf tensor 127.0.0.1:29591 /out/tensor /out/solo
```

Use `-d CPU` for the CPU RPC controls and `layer` for their layer-split run.
Reference logits are native float32 temporary files alongside each prompt's
JSON; full logits remain local rather than bloating the repository. Reports
include per-prompt output IDs, declared thresholds and measured errors. Runner
receipts record source/patch hashes and commands. Hardware-pair checks and
matched unprofiled performance trials are still required.
