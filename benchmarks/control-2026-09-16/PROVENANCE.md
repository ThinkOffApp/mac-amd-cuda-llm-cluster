# The 16 Sep 2026 control run, and two dead ends (raw files)

> **TRANSPORT: TCP over the 10 GbE cable. Not RDMA.** See [`TRANSPORT.md`](../../TRANSPORT.md).

This directory holds the raw `llama-bench` JSON and the complete stderr behind the
**16 Sep 18:09Z control** that [`../split-2026-09-17/PROVENANCE.md`](../split-2026-09-17/PROVENANCE.md)
already reports in prose. The numbers there were correct; the artifact behind them was not
committed. It is here now so the figures are checkable rather than quoted.

## CONTROL — MEASURED by us

**This is a CONTROL, not a result.** Its job is to establish what the default layer
placement costs, so that the explicit-`-ts` results elsewhere have something to be read
against. It is not the memory-spread outcome anyone is looking for.

- **Model**: `Qwen3.8-27B-UD-Q8_K_XL.gguf`, 31,457,991,680 bytes on disk;
  `model_size` 31,446,994,944 (29.29 GiB), 27.32 B params
- **Build**: `434ddbbc0` (build 846), **both ends**, Metal client / CUDA RPC server
- **Flags**: `-p 512 -n 128 -r 3`, **no `-ts`** — llama.cpp's DEFAULT placement
- **RPC device**: `10.10.10.2:50052`

| config | pp512 tok/s | tg128 tok/s | file |
|---|---:|---:|---|
| Mac alone | **705.48 ± 1.44** | **15.00 ± 0.26** | `local-20260916T180349Z.json` |
| Mac + Spark, default split | **612.37 ± 65.10** | **9.67 ± 0.02** | `split-20260916T180541Z.json` |
| | −13.2 % | −35.5 % | |

± is llama-bench's own stddev across `r=3` within the invocation.

**The ±65.10 on the split prompt figure is 10.6 % of the mean** and is the single loosest
error bar in this repo. Three repetitions inside one invocation cannot separate that from
drift. Treat the −13 % as directional.

**A naming trap worth one line**, and the same one the top-level README already warns about:
this file is named `UD-Q8_K_XL` and `llama-bench` reports its type as **`Q4_K - Medium`**.
The name is not the contents. Read `model_type` in the JSON, not the filename.

## DEAD END 1 — a 177 GB GGUF that upstream llama-bench will not load

**Record this so nobody spends an evening on it twice.**

`/Users/petrus/ds4/gguf/Qwen3.8-Flash-Next-Q4.gguf`, **177,280,286,720 bytes** (177.28 GB
decimal / 165.1 GiB). Read straight out of the file's own GGUF header for this commit, so
it is verifiable and not recalled:

```
magic GGUF   version 3   n_tensors 1256   n_kv 58
general.architecture = 'qwen4exp'
general.name         = 'Qwen3.8-Flash-Next Q40Routed (converted fast-pack)'
```

Upstream `llama-bench` at `434ddbbc0` fails on it. The captured stderr
(`split-20260916T180805Z.err`) contains, in full, one line:

```
llama_bench: error: failed to load model '/Users/petrus/ds4/gguf/Qwen3.8-Flash-Next-Q4.gguf'
```

**The tooling trap is that this is the *whole* message.** `llama-bench` swallows the
underlying loader error at default verbosity — the same defect the README already documents
for Metal OOM (`res = -3`, upstream ggml-org/llama.cpp#28107). The underlying failure was
reported to us as `gguf_init_from_reader: failed to read tensor data`, visible only with
`-v`. **We do not hold a `-v` capture of it**, so that inner message is recorded here as
reported, not as something this directory proves. What this directory proves is the
outer symptom and the file's identity.

**The sharp version of the finding, which is narrower and more useful than "qwen4exp
fails":** the `qwen4exp` architecture loads fine. The 87.24 GiB `Qwen3.8-Flash-Next-UD-IQ4_XS`
is the *same* architecture and is the model behind every split number in
[`../split-2026-09-18/`](../split-2026-09-18/) and [`../three-box-2026-09-17/`](../three-box-2026-09-17/).
It is **this "converted fast-pack" repack specifically** that upstream will not read.

## DEAD END 2 — `-sm row` (tensor parallel) over RPC

`splitrow-20260916T180859Z.err` is the attempt, and it fails the same swallowed way:

```
llama_bench: error: failed to load model '…/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q8_K_XL.gguf'
```

The informative message (`device RPC0 does not support split buffers`) is the one already
recorded in [`../split-2026-09-17/PROVENANCE.md`](../split-2026-09-17/PROVENANCE.md); it
does not appear in this capture. **Layer split is the only split llama.cpp's RPC backend
offers.** That is a statement about this engine, not about every engine.

## DEAD END 3 — the RPC server was simply not running

`split-20260916T180349Z.err` is kept because it is a *different* failure that looks similar
from a distance, and the paired `.json` is **0 bytes**:

```
ggml/src/ggml-rpc/ggml-rpc.cpp:547: Failed to connect to 10.10.10.2:50052
```

Empty and near-empty result files are kept rather than deleted, matching the convention in
[`../roce-2026-09-17/raw/`](../roce-2026-09-17/raw/). A zero-byte JSON is a record of an
attempt, not an absence of one.
