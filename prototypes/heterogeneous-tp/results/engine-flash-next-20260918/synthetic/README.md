# Flash-Next tensor-mode investigation

Engine: llama.cpp bdcbaaf6e7520b68c8c60ff724c67409970d70e1.
Local experimental worktree: work/llama-flash-tp, build-probe.

Upstream disables tensor mode for qwen4exp. The diagnostic build opts in only with LOCAL_QWEN4EXP_TP_PROBE; this is not a production support declaration.

The original synthetic architecture test aborts because model.input_embed (reshaped), a CPU-buffer RESHAPE of CPU GET_ROWS, reaches the meta-buffer accessor. The accessor correctly rejects it. tensor-diagnostic.log preserves the failure and source details.

The experimental patch extends the existing CPU-view exception to explicit VIEW/RESHAPE/PERMUTE/TRANSPOSE nodes with a non-meta host buffer and host-backed view source. It applies the same predicate in graph mapping and reduction scans. Buffer assertions remain active. The test-only addition runs two partitions using the same Metal device.

Results: qwen4exp seeds 1, 42 and 1234 pass the existing NMSE gate, including Meta-local-2; qwen35 seed 1234 also passes. Seed 1234 qwen4exp NMSE is 2.14e-7. The roundtrip serialization test for Meta is skipped by the existing test and has NOT been validated. Verbose output records CPU fallback for fused hyperconnection pre/post operations.

Scope: small synthetic models, prefill logits, one physical Apple M5 Max. These tests do not establish incremental decode/cache correctness, real Flash-Next quantized weights, RPC interoperability, physical-pair correctness, communication share, or speedup. Those are still required. The original engine worktree and serving services were not modified by this experiment.

Reproduce: configure with GGML_RPC=ON, LLAMA_BUILD_TESTS=ON and CMAKE_CXX_FLAGS=-DLOCAL_QWEN4EXP_TP_PROBE=1; build test-llama-archs; run build-probe/bin/test-llama-archs -a qwen4exp -s 1234 -v 1. See validation.json, host-view-experimental.patch, and logs for additional seeds and control.

## Incremental decode follow-up

The diagnostic test supports LOCAL_TP_INCREMENTAL=1: 64 prompt tokens followed by 64 single-token calls, retaining cache state. At seed 1234, qwen4exp passes all backend comparisons, including Meta-local-2 (NMSE 1.05e-7 against the CPU incremental reference). This is teacher-forced synthetic input, not greedy generation or a physical-pair run. See incremental-test.json/log and incremental-experimental.patch. The earlier limitation on incremental coverage now applies to actual weights and RPC, rather than this synthetic sequence.
