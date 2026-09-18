# Static GPT-2 batching

`batch_tp.py` extends ClaudeMM's batch math from commit 271a9f9 with mandatory correctness gates and matched solo execution. It uses the pinned GPT-2 checkpoint (124M, FP32), distinct uniform-length prompts, independent per-layer/per-sequence KV caches, and the last-position/rank-0 generation head. The checkpoint SHA-256 is checked before loading. Default batch is 1.

This amortizes one tensor reduction across multiple sequences. It does not implement asynchronous overlap or a continuous request scheduler. Finished EOS lanes remain in the batch while others finish; their later work is not counted as generated output. This is an explicit static-batch cost.

Every sequence has an independently generated, EOS-aware HuggingFace reference. Both ranks must pass the initial check and warmups. Each timed output is checked again outside the timed interval; any mismatch withholds ALL timing records and exits both ranks nonzero. A barrier precedes each timed run. Solo runs the full model on its device without pointless CPU staging of reduction inputs.

Reports include prompt/reference/produced token IDs, model and source hashes, device/host, threads, actual token counts, reduction input bytes, each sequence's TTFT/latency/decode rate, and batch wall throughput. The aggregate decode interval spans the first available token through the last; it is not a claim that all lanes stayed active throughout. Compare end-to-end throughput alongside per-sequence latency. Never multiply median per-stream rates by the requested batch size.

Example solo:

```sh
GPT2_DIR=/path/to/pinned/gpt2 python batch_tp.py --mode solo --device mps --batch 4 --prompt-tokens 128 --new-tokens 64 --warmups 2 --runs 5
```

For a pair, use `--mode tp --transport tcp` on both machines with matching prompt/batch/run arguments and `WORLD_SIZE=2`, `RANK=0/1`, `MASTER_ADDR` and `MASTER_PORT`. Use mps on the Mac and rocm/cuda on the other host. Keep model/source hashes identical.

Hardware comparison: batch 1/2/4/8/16, prompt 128/512, 64 new tokens, two warmups, five runs. Interleave both solo arms and the pair at EACH matching batch and compare against the faster member of that same pair. Preserve both-rank outputs and service restoration evidence. Local CPU/MPS correctness checks do not establish a physical-pair speedup. Serving-service interruption still requires the coordinated window.

Validation: `GPT2_DIR=... python validate_batch_edges.py --out /path/to/results` tests early EOS, one-token rate accounting, and a mutation applied only after the initial correctness gate and warmup. The mutation must make both ranks exit 1 and publish no timing rows. Test-only EOS forcing and corruption are generated into temporary scripts, never production paths.
