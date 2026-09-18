# Actual Flash-Next local tensor experiment

Pinned engine and experimental patch hashes are in provenance.json. This uses the qwen4exp architecture opt-in and CPU-view fix from the separate local diagnostic worktree. It does not modify a serving installation.

Solo reference completed three prompts and 48 token choices. The tensor-local mode uses two Metal backend instances on one physical M5 Max. It matched 48/48 argmax token choices and produced finite full-vocabulary logits, but FAILED the existing atol=0.1, rtol=0.01 logit gate on prompts 0 and 1. Maximum absolute errors were 1.470399 and 0.761870; prompt 2 passed with 0.0007474. The tolerance was not relaxed.

Thus actual Flash-Next tensor-mode correctness is not yet established. This is stronger evidence than the passing tiny synthetic architecture tests, and their pass must not be treated as full-model support. Next isolate single-partition tensor-mode behavior before attributing the discrepancy to multi-partition arithmetic or cache behavior.

No performance claim: run.json wall times include model loading. CPU-resident embedding/PLE weights and CPU operation fallbacks remain part of the execution. No physical pair or RPC was tested here. Full SHA-256 hashes of all three split files are recorded in model-sha256.json. Each file was checked for unchanged size and modification time across hashing; cross-machine tests must match these hashes.

## Single-partition control

The actual-weight tensor-single control completed successfully: all 48 choices matched and every compared logit was exactly equal to the solo reference (maximum error 0.0 for all three prompts). It uses the same experimental engine and weights, with one Metal device in tensor mode. This narrows the observed failure to the multiple-partition execution path rather than the general tensor-mode graph alone; it does not yet identify the faulty operation. See tensor-single/report.json and run.json.

## Repeated two-partition run with per-step diagnostics

The failure reproduces with identical maximum errors. All three prefill outputs pass. Prompt 0 first exceeds tolerance at zero-based generated step 3; prompt 1 at step 11; prompt 2 passes every step. All token choices still match. Detailed counts of vocabulary logits outside tolerance are in tensor-local-steps/report.json. This localizes the next investigation to an incremental operation or data-dependent numerical/routing difference; it does not yet prove a cache bug. The test uses teacher-forced identical contexts, so divergent generated tokens cannot explain the logit discrepancy.

## Routing trace

Read-only evaluation callbacks recorded 672 matching tensor keys across the single- and two-partition controls: expert probabilities/selections and sparse-attention selections for prompt 0, generated steps 0 through 3. Tracing reproduced the original per-prompt maximum errors exactly, while the traced single-partition control retained zero error.

The first differing expert set occurs at zero-based step 3, layer 29. The first nine experts agree. For the last slot, solo/single selects expert 34 and two partitions select expert 259. Their single-partition probabilities are 0.011779369786381721 and 0.011779310181736946, a gap of about 5.96e-8. With two partitions they are 0.011779332533478737 and 0.011779405176639557, reversing the order. Maximum probability difference in that layer is 6.85e-7. Earlier layer probability differences are small, with no expert-set differences. Later layers develop additional selection differences.

This supports routing sensitivity to small numerical differences as the first observed discontinuity, rather than establishing a gross partition/cache error. Causality still needs a diagnostic intervention or deeper operation-level comparison. It does NOT waive the failed strict logit gate, establish general model quality, or justify changing production routing to match this one reference. No expert choice was forced by this trace. See trace/comparison.json, first-routing-divergence.json, compare.py, and the raw routing.jsonl files.

## Diagnostic causal intervention (not a model fix)

A separate executable replaces ONLY the tenth expert at prompt 0, generated step 3, layer 29: 259 becomes reference expert 34. The callback asserts the expected shape and original value, writes the replacement, and reads it back. This is explicitly reference-dependent diagnostic behavior and is not part of engine_correctness.cpp or the engine patch.

At the affected step, maximum logit error falls from 1.209365 to 0.000764847, with zero logits outside tolerance. Steps 4 through 9 also pass; a later failure appears at step 10 (0.836588). The overall run still exits 1 and remains invalid. This intervention supports the near-tied routing choice as the cause of the first divergence, rather than merely a correlated downstream symptom. It does not repair later routing differences, prove all failures share this cause, or justify using forced experts for correctness/performance claims.
