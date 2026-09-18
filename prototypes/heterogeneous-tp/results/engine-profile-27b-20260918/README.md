# 27B engine collective profile

Model: Qwen3.8-27B UD-Q4_K_XL GGUF, full SHA-256 in profile-provenance.json. Engine: llama.cpp bdcbaaf6 with the existing empty-view RPC fix and an opt-in local meta-backend profiler. Machine: one Apple M5 Max, two Metal backend instances with the second behind localhost RPC. This is not a physical Mac/Spark or Mac/AMD result.

Three short prompts, up to 16 choices each, ctx256, batch/ubatch128, KV f16. Profile ON and OFF both pass 48/48 reference choices and the preset full-logit tolerance; worst absolute logit error 0.001158. The reference is the same GGUF on unsplit Metal.

Instrumented decode (45 calls): compute 2.666 s, collective 5.401 s, preparation 0.0116 s, initial wait 0.0004 s, unattributed 0.1007 s, whole decode calls 8.180 s. Collective time is 66.03% of this instrumented wall time. There are 128 collectives per call, 5760 total. Summed participant input tensor bytes are 235,929,600; these are NOT network bytes. Three prefill calls have a 52.05% collective share but use short prompts and include cold execution effects.

GGML_META_PROFILE=1 adds synchronization around compute and collective phases. Collective time includes transfer, RPC handling, reduction work and waits; it is not pure wire time. Compute includes graph dispatch and waiting for both participants, measured as elapsed wall time rather than summed device times. Backend preparation and initial synchronization are separate. Parent decode-call markers include CPU/scheduler work outside the meta backend. Weight loading is outside these markers. summarize.py verifies accounting bounds and correspondence with actual token counts.

The OFF control emits no META_PROFILE records and has 8.275 s across 45 decode calls, compared with 8.180 s ON (ratio 0.9884). This single ordered pair does not prove negative overhead or a speed improvement; it is only a consistency check. Profiled timings are not accepted as uninstrumented performance benchmarks. Full model loading time is not used in these phase totals.

Source: engine-profile.patch plus profile_correctness.cpp. The patch preserves existing graph execution with profiling disabled. The local Python runners record commands and ensure the RPC server is terminated afterward. They currently contain workspace-specific paths; adapt those paths for another host and verify the model hash. Archived OFF receipts are under unprofiled/tensor. No remote serving service was stopped.
