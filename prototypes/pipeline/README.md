# Rolling microbatch pipeline

`rolling.py` implements a bounded, whole-layer, two-stage scheduler derived from
ClaudeMM's `51103d9` prototype. It advances each chunk immediately after that
chunk's reply arrives, without waiting for every other chunk at the token boundary.
Each chunk owns its KV state and position. Messages identify run, chunk and step.
The number of outstanding chunks is bounded by `--chunks`.

Three controls share the same model arithmetic and transport:

- `alternating`: send one chunk, wait for its reply, then send the next.
- `barrier`: submit all chunks, drain all replies, then advance the global step.
- `rolling`: prime all chunks, then submit a chunk's next step as soon as its reply returns.

This prototype deliberately supports only the pinned GPT-2 124M checkpoint. It is
an algorithm/correctness gate, **not** the requested Gemma 4 / 27B performance result.
It requires an independent, cached Hugging Face FP32 reference for every sequence,
checks initial, warmup and timed outputs, and withholds all rates if any check fails.
Both ranks must agree on checkpoint hash and workload configuration. Distinct prompts
support batch sizes up to 64. Reported token counts stop at EOS; finished lanes remain
in the static computation budget, so this is not continuous admission/compaction.

`host_event_order` establishes submission order only. It does not establish GPU
utilization or actual cross-device overlap. No GPU busy fraction is reported.
Rank-0 wall timing includes computation, transfers and scheduling. Decode counts
exclude the prefill-produced first token. Zero decode tokens produce no decode rate.

## Run

Set `GPT2_DIR` to the pinned checkpoint directory and launch one process per host:

```sh
RANK=0 MASTER_ADDR=HOST_A MASTER_PORT=29901 GPT2_DIR=/model/gpt2 \
  python rolling.py --device mps --schedule rolling --batch 16 --chunks 2 \
  --prompt-tokens 128 --new-tokens 64 --warmups 2 --runs 5 --out rolling.json
RANK=1 MASTER_ADDR=HOST_A MASTER_PORT=29901 GPT2_DIR=/model/gpt2 \
  python rolling.py --device rocm --schedule rolling --batch 16 --chunks 2 \
  --prompt-tokens 128 --new-tokens 64 --warmups 2 --runs 5 --out unused-rank1.json
```

All options except device and output must match across ranks. TCP is intended for a
trusted private network and is unauthenticated. The same transport framing caveats
apply as to the originating prototype.

Sweep **fixed chunk size** with more chunks in flight: e.g. chunk size 8 and
(B,C)=(8,1),(16,2),(32,4),(64,8). For each total B, also run barrier and alternating
controls with identical chunk partition, plus a C=1 whole-batch control at that B.
Use measured stage times to choose layer allocation; equal layer counts are not
necessarily balanced on heterogeneous devices. Interleave repeated trials and
include the faster member's solo control at the same total workload/concurrency.
More sequences may amortize bubbles but can add queueing latency; no 2x claim follows
from doubling B. A rolling schedule still has startup/drain bubbles and can be
limited by transfers, GPU memory bandwidth, launches or an imbalanced stage.

## Validation receipts

`results/rolling-20260918` contains raw local reports and the exact executed runners:

- CPU/CPU alternating, barrier and rolling: all initial/warmup/timed outputs match HF.
- Event-order check: rolling sends chunk 0's next step before receiving chunk 1's
  current reply; barrier and alternating wait for that reply.
- Injected corruption only in timed runs: both ranks exit 1; all rates withheld.
- Injected early EOS: output counts 2/4/4/4; other sequences keep generating correctly.
- One-token request: zero decode count and null decode rate.
- Batch 16: distinct prompts and independent reference check pass.
- Metal/CPU rolling: independent reference check passes.

These are same-host functional checks, **not physical-pair performance measurements**.
The negative fixtures modify temporary copies only. To repeat locally:

```sh
python validate_schedules.py --model-dir /model/gpt2 --out /tmp/rolling-schedules
python validate_edges.py --model-dir /model/gpt2 --out /tmp/rolling-edges
```

The edge runner includes a Metal case and therefore requires a Metal-capable host.
The modern GGUF-model diagnostic is described in [GGUF_CONTEXTS.md](GGUF_CONTEXTS.md).
Physical-pair overlap and performance remain to be measured.

A working same-host prefill/decode state-transfer recipe for Gemma 4 and 27B,
including failed and passing full-logit controls, is in [SLOT_HANDOFF.md](SLOT_HANDOFF.md).
