# Measurements added 16-18 September 2026

The cable itself, the same-commit Mac + Spark rerun, the `-ts` ratio sweep, three boxes, the KV hand-off, the two-Spark GLM serve, and dead ends. All TCP over 10 GbE, none RDMA.

> Moved verbatim from the [README](../README.md) on 23 Sep 2026 to keep the README short.
> Nothing was cut. Where the text says "above", "below" or "this README", it refers to
> the README as it was before the move; the [README's last section](../README.md#details-and-all-numbers)
> lists where each part now lives.

## Measurements added 16-18 September 2026

> **TRANSPORT for this whole section: TCP over the direct 10 GbE cable. None of it is
> RDMA**, and the Mac end cannot be — it has no RDMA-capable NIC.
> [`TRANSPORT.md`](../TRANSPORT.md) records this per run.

Five directories of raw material, all **MEASURED by us** unless a row says otherwise.
Everything below is reproducible from the committed `llama-bench -o json`, which carries
`build_commit`, `model_filename`, `model_n_params`, `n_batch`, `n_ubatch`, `tensor_split`
and per-repetition `samples_ts`.

| directory | what | date |
|---|---|---|
| [`benchmarks/link-2026-09-16/`](../benchmarks/link-2026-09-16/) | the cable itself: iperf3 both ways, TCP ping-pong at three sizes | 16 Sep |
| [`benchmarks/control-2026-09-16/`](../benchmarks/control-2026-09-16/) | raw files behind the default-placement CONTROL, plus three dead ends | 16 Sep |
| [`benchmarks/three-box-2026-09-17/`](../benchmarks/three-box-2026-09-17/) | Mac / 1 Spark / 2 Sparks / all three, one model, one commit | 17 Sep |
| [`benchmarks/split-2026-09-18/`](../benchmarks/split-2026-09-18/) | the cleanest Mac+Spark split set, plus the `-ts` ratio sweep | 18 Sep |
| [`benchmarks/handoff-2026-09-18/`](../benchmarks/handoff-2026-09-18/) | prefill on the Spark, **KV cache moved**, decode on the Mac | 18 Sep |

### The link, and the number an RDMA transport has to beat

Bandwidth is not the problem on this cable. `iperf3`, 4 streams, 10 s, both directions:
**9.42 Gbit/s sending / 9.41 receiving, zero retransmits, symmetric.** Latency is the
problem. TCP ping-pong, `TCP_NODELAY`, 3000 samples per size, round-trip microseconds:

| payload | min | p10 | median | p99 |
|---|---:|---:|---:|---:|
| 64 B | 100.0 | 195.2 | *see caveat* | 1047.9 |
| **4 KiB** | 102.8 | 222.1 | **244.0** | 1143.4 |
| 64 KiB | 234.9 | 354.2 | 810.8 | 1484.0 |

**Caveat on the 64 B row, which is why it has no median here.** That series is bimodal and
drifts *within the run*: its first 200 samples median 213 µs, the full 3000 median 403 µs,
and 500-sample windows read 407 / 407 / 412 / 574 / 393 / 319. It is also *slower* than
4 KiB at the median, which is backwards for a wire and points at host-side scheduling. The
4 KiB and 64 KiB rows are stable to within 5 % across the run and are safe to quote.

Against that, **NOT OURS**: Ash Hart's MCDMA 4 KiB QD1 medians, Mac→Spark **7.6 µs WRITE /
6.0 µs READ**. A **16-32×** gap depending on a measurement boundary neither side has stated.
Our *best single sample in 3000* is still 13× his median. Bandwidth at line rate, latency
an order of magnitude and a half away, is the whole shape of the problem.

### Mac + Spark split, on one commit, on a quiet machine

The 17 Sep Flash-Next split rows above were taken with mismatched builds under concurrent
downloads. This repeats them properly: `434ddbbc0` **both ends**, quiet machine,
`Qwen3.8-Flash-Next-UD-IQ4_XS` (87.24 GiB, 176.9 B params / 3 B active), which fits either
box alone so both solo arms are real. Two sets, n=3 and n=2, each repetition a separate
invocation:

| | pp512 | pp2048 | pp4096 | tg128 |
|---|---:|---:|---:|---:|
| Mac alone | **1064.03 ± 8.59** | 1043.26 ± 10.10 | 964.71 ± 3.92 | **37.88 ± 0.87** |
| Spark, all layers, over RPC | 809.80 ± 0.68 | 828.03 ± 3.10 | 819.94 ± 5.17 | 26.98 ± 0.09 |
| split 50/50 | 842.75 ± 0.59 | **1089.97 ± 6.79** | **1143.03 ± 0.47** | 26.95 ± 0.29 |
| **split ÷ Mac** | 0.792 | **1.045** | **1.185** | 0.711 |

**This reproduces the 17 Sep result under better conditions** (0.722 / 1.003 / 1.128 then,
0.792 / 1.045 / 1.185 now) — same direction, slightly better, and now with both ends on one
commit. Generation is 0.71 of the Mac and a **tie with the slower box**, so on this model
the split costs generation outright.

*Two limitations this set does not share with the 17 Sep data:* the arms ran in **fixed
order**, not rotated, so drift within a round is absorbed rather than exposed; and the
"Spark" arm is the Spark holding every layer **with the Mac as RPC client**, which still
pays a round trip per token and is not the native GB10 solo measurement used elsewhere.

### The `-ts` ratio sweep the 17 Sep section said we had not done

Flash-Next, pp2048, **n=1 per point** — a screening sweep, not a measured optimum:

| `-ts` spark/mac | 60/40 | 50/50 | 43/57 | **35/65** | 30/70 | 25/75 | 20/80 | 15/85 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pp2048 | 789 | 1087 | 1176 | **1283** | 1218 | 1170 | 1116 | 1086 |

**Prefill has an interior optimum near `-ts 35/65`** — 1283 tok/s, **1.23× the Mac alone**
and 18 % above the 50/50 used everywhere else in this repo. The runner's *predicted*
optimum of 43/57 was wrong; the measured peak is Mac-heavier. The generation row from the
same sweep is **not** reportable across its full range: the eight points are two batches
three minutes apart and `tg64` steps 3.82 at the batch boundary against 0.68 for the
largest within-batch step. Details and the full table in
[`benchmarks/split-2026-09-18/`](../benchmarks/split-2026-09-18/).

### Three boxes: a second Spark helps prefill, a third box helps nothing

One model, one commit, one client, **only the layer allocation changes** (17 Sep,
r=3, tok/s):

| | pp512 | pp4096 | tg128 |
|---|---:|---:|---:|
| Mac alone | 1010.60 ± 3.34 | **1015.10 ± 11.54** | **41.41 ± 0.12** |
| Spark 1 (`-ts 1/0`) | 818.41 ± 18.38 | 824.02 ± 1.06 | 27.33 ± 0.06 |
| both Sparks (`-ts 1/1/0`) | 686.07 ± 15.05 | **927.53 ± 2.87** | 23.49 ± 0.16 |
| all three (`-ts 1/1/1`) | 626.28 ± 22.91 | 652.45 ± 89.28 | 20.72 ± 0.56 |

A second Spark buys **+12.6 % prefill at 4096** and costs 16 % at 512. Adding the Mac as a
third weight-holder is worse on every column and carries a 13.7 % error bar. **No multi-box
row beats the Mac alone**, and generation falls monotonically with each box added:
41.41 → 27.33 → 23.49 → 20.72.

**The topology point that makes row 3 easy to misread:** llama.cpp RPC is a star. The two
Sparks never talk to each other — every activation goes **Spark → Mac → Spark**. The
200 GbE RoCE fabric carries none of this, even though the addresses used sit on that
subnet. The 185 Gbit/s `ib_write_bw` figure elsewhere in this README is a synthetic
result between idle boxes and is unrelated to these rows.

*Drift that invalidates part of that table:* the Mac baseline was measured twice at pp4096,
28 minutes apart, same command — 881.23 ± 28.66 then 1015.10 ± 11.54, a **15 %** difference.
Both are committed, neither is discarded, and the later one is the time-matched baseline for
rows 2-4. Within-invocation repetitions did not reveal this; only re-running did.

### Prefill on one box, decode on the other: a KV hand-off over sockets

The most directly comparable thing in this repo to an RDMA transport project, because the
moved bytes and the transfer time are both explicit. Prefill N-1 tokens on the Spark, `scp`
the saved KV slot to the Mac, restore, generate there. **Output verified token-for-token
against a Mac-native run at all three lengths**, and the state file's SHA-256 matched
across the wire.

| prompt tokens | KV state | copy | Spark prefill | Mac alone | split TTFT | **÷ Mac alone** |
|---:|---:|---:|---:|---:|---:|---:|
| 2111 | 295.3 MB | 0.672 s | 2.662 s | 3.001 s | 3.568 s | 0.841 |
| 3072 | 358.2 MB | 0.694 s | 3.736 s | 4.227 s | 4.686 s | 0.902 |
| 8192 | 693.9 MB | 1.088 s | 9.915 s | 12.921 s | 11.471 s | **1.126** |

**Break-even is between 3072 and 8192 tokens.** KV state grows ≈ 65.5 KB per token on this
model — that is the quantity a faster transport would move.

**IDEA, arithmetic on measured inputs, not a measurement:** at 8192 the copy is 1.088 s of
a 1.556 s hand-off overhead, **70 % of it**. A free transfer would move the ratio from 1.126
to ≈ 1.24 and pull break-even below 3072. That is the size of the prize, not a prediction.

**Do not read 638 MB/s as a link ceiling.** That is 5.10 Gbit/s against 9.42 Gbit/s on the
same cable under `iperf3`. The copy is `scp`, so SSH crypto and single-stream are plausible
limiters, and **we did not isolate which** — the gap is unexplained, not attributed.

### The two-Spark GLM serve: read the transport per run

**Serve MEASURED by codexmb. The bounded measurement below was produced by the 2026-09-20
Codex task and independently inspected by codexmb; its raw artifact is committed byte-exact
at [`benchmarks/glm-2spark-2026-09-20/`](../benchmarks/glm-2spark-2026-09-20/).**
MiaAI-Lab GLM-5.3-Flash-EXL3-2x-DGX-Sparks at checkout
`6961fa0706f3c0b25775bf42a575471972582bac`, image
`ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor`, both ASUS Ascent GX10
boxes serving GLM-5.3-Flash-EXL3. DFlash 7, MTP 2, context 8192, max seq 2. One bounded
streaming measurement: **24 prompt + 384 completion tokens**.

> **That run used `NCCL_NET=Socket` with `NCCL_IB_DISABLE=1`, after `ibv_reg_mr_iova2`
> returned ENOMEM. It is explicitly NOT an RDMA success** — it is what the stack fell back
> to when RDMA registration failed.

**The 19.2 tok/s figure, and what it actually measures.** It is **fully documented**, and
the artifact is now committed unmodified at
[`benchmarks/glm-2spark-2026-09-20/measurement.json`](../benchmarks/glm-2spark-2026-09-20/measurement.json)
(445 bytes, sha256 `cf0b7df3…bf11`), copied from
`Documents/Codex/2026-09-20/che/outputs/glm-performance/measurement.json`. It reads:

```
prompt_tokens 24   completion_tokens 384   total_tokens 408
elapsed_seconds 20.032584      first_text_seconds 0.3227
overall_completion_tokens_per_second 19.16877
reasoning_characters 1634      answer_characters 0
first_answer_seconds null      finish_reason "length"
```

**The reason it cannot be quoted as throughput is not missing conditions. It is that the
task produced no answer at all.** All 384 completion tokens were spent inside the reasoning
block, the run hit the token cap, and it emitted **zero answer characters**.

Stated precisely, and this is the label to carry if the figure is quoted anywhere:
**19.16877 is the overall completion-token generation rate for a length-capped,
reasoning-only response with zero final-answer characters. It is not answered-task
performance.**

Those are different quantities, and a reader assembling a benchmark bundle needs the
distinction: this number is usable as "how fast does this stack emit tokens", and is not
usable as "how fast does this stack answer a question". We publish it as the former,
labelled, rather than as the latter.

An earlier revision of this section said the conditions could not be located. **That was
wrong** — they were recorded all along, in the file above.

**And the transport on that pair is not a constant.** A *different* run on the same two
boxes the same day used `NCCL_NET=IB` with `NCCL_IB_DISABLE=0` on GID 6/5 and ran over
RoCEv2. As of 20 Sep 2026 RDMA there is failing again with the same
`ibv_reg_mr_iova2: Cannot allocate memory`, suspected to be a kernel-level registration
regression, and the serve is on TCP. **That suspicion is not a diagnosis.** The per-run
table is in [`TRANSPORT.md`](../TRANSPORT.md); never generalise a transport from the machine
pair to a run.

### Dead ends, recorded so nobody repeats them

- **A 177 GB GGUF upstream will not load.** `Qwen3.8-Flash-Next-Q4.gguf`, 177,280,286,720
  bytes. Read from the file's own header for this commit: `general.architecture = 'qwen4exp'`,
  `general.name = 'Qwen3.8-Flash-Next Q40Routed (converted fast-pack)'`, GGUF v3, 1256
  tensors. `llama-bench` at `434ddbbc0` fails on it, and the **entire** captured error is
  `llama_bench: error: failed to load model '…'` — the loader's real message is swallowed at
  default verbosity and needs `-v`, the same defect the Metal-OOM pitfall below describes.
  **The sharp version:** `qwen4exp` is fine — the 87.24 GiB `UD-IQ4_XS` of the *same
  architecture* is the model behind every split number in this section. It is this
  "converted fast-pack" repack specifically that upstream will not read.
- **A concurrency harness that reported numbers with a dead server.** Three multistream
  attempts, all kept in [`benchmarks/split-2026-09-18/`](../benchmarks/split-2026-09-18/),
  produced no usable comparison. `llama-server` allocates `n_ctx_slot` **per slot**, so
  8 slots at a large `-c` OOMs Metal (2650 `kIOGPUCommandBufferCallbackErrorOutOfMemory`
  lines in one log); and in the third attempt the driver recorded pair figures while that
  arm's server log stops at `load_model`, so **which process produced them is UNKNOWN** and
  they are quoted nowhere.
