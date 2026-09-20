# Prefill on the Spark, decode on the Mac: a KV-cache hand-off over TCP (18 Sep 2026)

> **TRANSPORT: `scp` over SSH over the 10 GbE cable. TCP, not RDMA.** This is the *same*
> hand-off shape that [MCDMA](https://github.com/ashhart/MCDMA) demonstrates over RDMA,
> run over sockets. That makes it a baseline for that work, not a competing result.
> See [`TRANSPORT.md`](../../TRANSPORT.md).

**Why this is the most directly comparable thing in this repo for an RDMA transport
project.** Every other result here is a *layer* split, where the two machines share one
model. This one instead runs the whole prompt on one box, moves the **KV cache state** to
the other box, and finishes generation there. The moved bytes are explicit, the transfer
time is separated from the compute time, and the output is checked token-for-token against
a native run. So the question "what would a faster transport buy?" has an arithmetic
answer here rather than a guess.

## Setup — MEASURED by us

- **Model**: `Qwen3.8-27B-UD-Q4_K_XL.gguf` (dense, 16.34 GiB, 27.32 B params), the same
  file on both machines
- **Build**: `434ddbbc0` (846) **both ends** — Mac Metal, Spark CUDA
- **Servers**: `llama-server` on each. Spark: `-ngl 999 -c 4096 -np 1 -b 512 -ub 512 -t 4
  --host 10.10.10.2 --port 19083 --slot-save-path … --cache-ram 0` (`runners/spark-start.sh`).
  Mac: local, `-c 4096` for the 2111 run and `-c 10240` for the 3072/8192 runs.
- **Method** (`runners/run_n1.py`, `runners/run_n1_lengths.py`): prefill **N-1** tokens on
  the Spark → `/slots/…?action=save` → `scp` the slot file to the Mac → `…?action=restore`
  → feed the final token and generate 16 on the Mac. The control is the **same prompt
  prefilled cold on the Mac and generated there**, in the same script run.
- **Correctness**: the slot file's SHA-256 is compared **before and after the copy** and
  matched in all runs, and the 16 generated token IDs are compared against the Mac-native
  arm. `tokens_match_native` is **true** at all three lengths. Correctness here is
  verified, not assumed.

## Results — MEASURED by us, n=1 per length

| prompt tokens | state bytes | copy s | copy MB/s | Spark prefill s | Mac cold prefill s | overhead s | split TTFT s | **TTFT ÷ Mac alone** | decode tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2111 | 295,300,388 | 0.672 | 439.4 | 2.662 | 3.001 | 0.906 | 3.568 | **0.841** | 25.50 |
| 3072 | 358,241,828 | 0.694 | 515.9 | 3.736 | 4.227 | 0.950 | 4.686 | **0.902** | 25.18 |
| 8192 | 693,929,508 | 1.088 | 637.7 | 9.915 | 12.921 | 1.556 | 11.471 | **1.126** | 25.04 |

`overhead s` = split TTFT − Spark prefill, i.e. everything the hand-off costs on top of
doing the prefill on the faster-at-prefill box: save, copy, restore, and the final token.

**The hand-off crosses break-even between 3072 and 8192 tokens.** Below that the transfer
costs more than the prefill it saves. This is the same shape as the layer-split crossover
elsewhere in this repo, arrived at by a completely different mechanism, which is mild
independent support for the general picture rather than a second measurement of the same thing.

**KV state grows about 65 KB per token** on this model (295.3 MB / 2111 ≈ 140 KB, 693.9 MB /
8192 ≈ 85 KB — not a constant, because a fixed base is included; the *increments* between
the three points give ≈ 65.5 KB/token). That is the quantity an RDMA transport would move.

## What a faster transport would buy here — IDEA, derived arithmetic, NOT measured

At 8192 tokens the copy is **1.088 s of a 1.556 s overhead, i.e. 70 % of it**. If the
transfer were free, split TTFT would fall from 11.471 s to ≈ 10.38 s and the ratio against
the Mac alone would move from **1.126 to ≈ 1.24**. Break-even would also move earlier,
somewhere below 3072 tokens.

**This is arithmetic on measured inputs, not a measurement.** It assumes the rest of the
overhead is unchanged and that a zero-copy path costs nothing on either host, neither of
which is true. It is offered as the size of the prize, not as a prediction.

**And the copy is not wire-limited, so do not read 638 MB/s as a link ceiling.** 637.7 MB/s
is **5.10 Gbit/s** against the **9.42 Gbit/s** the same cable carries under `iperf3`
([`../link-2026-09-16/`](../link-2026-09-16/)). The transfer is `scp`, so SSH encryption and
the single-stream copy are plausible limiters. **We did not isolate which**, so the gap
between 5.1 and 9.4 Gbit/s is UNEXPLAINED rather than attributed. A plain-socket or
RDMA transfer would have to be measured against a plain-socket control, not against this one.

## Limitations

- **n=1 at every length.** No repetitions, no rotation. The three rows are consistent with
  each other and with a smooth trend, which is not the same as being replicated.
- **The 2111-token row used `-c 4096` on the Mac; the 3072 and 8192 rows used `-c 10240`.**
  Not a matched configuration across the three rows.
- `report.json` and `report-n3072.json` / `report-n8192.json` overlap with `report-3k8k.json`,
  which is the combined driver output for the two longer lengths. All are kept.
- **Only one model, one direction (Spark prefill → Mac decode), one link.** The reverse
  direction was not run.
- The KV slot `.bin` files themselves (1.3 GB) are **deliberately not committed**; their
  SHA-256 values are in the reports.

## Files

| path | what |
|---|---|
| `datasets/report.json` | full driver output, 2111 tokens, including both arms' token IDs and timings |
| `datasets/report-n3072.json`, `datasets/report-n8192.json` | the same shape at 3072 and 8192 |
| `datasets/report-3k8k.json` | combined summary rows for the two longer lengths |
| `runners/run_n1.py`, `runners/run_n1_lengths.py` | the drivers, including the SHA-256 check |
| `runners/spark-start.sh` | the exact Spark `llama-server` command |
| `logs/mac-server.log`, `logs/mac-server-3k8k.log`, `logs/spark-server.log` | complete server logs, verbosity 4, carrying the build line and per-slot timings |
| `logs/mac-prefill-curve.txt` | the Mac prefill-versus-length curve used to pick the lengths |
