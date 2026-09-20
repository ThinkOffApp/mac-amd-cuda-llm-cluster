# Mac + Spark layer split, quiet machine, both ends on one commit (18 Sep 2026)

> **TRANSPORT: TCP over the 10 GbE cable, every row. Not RDMA.** The Mac has no RDMA NIC.
> `grep -ci rdma` over every log in this directory returns 0, against the
> [`../rpc-rdma-2026-09-17/`](../rpc-rdma-2026-09-17/) capture where those lines are present.
> See [`TRANSPORT.md`](../../TRANSPORT.md).

This is the cleanest Mac+Spark set in the repo and it supersedes the Flash-Next split rows
in [`../split-2026-09-17/`](../split-2026-09-17/) for that model. Three things improved:
**both ends are on the same llama.cpp commit** (the 17 Sep set had Mac `1d0c76f3c` against
Spark `434ddbbc0`), the machine was quiet (the 17 Sep prefill panel ran under two large
downloads), and the solo arms are real because the model fits either box alone.

- **Model**: `Qwen3.8-Flash-Next-UD-IQ4_XS`, 93,671,559,680 bytes (87.24 GiB),
  **176,943,899,520 params / 3 B active**, arch `qwen4exp`
- **Build**: `434ddbbc0` (846) **both ends**, Metal client + CUDA `ggml-rpc-server`
- **Link**: Mac `en12` 10.10.10.1 ↔ Spark 1 `10.10.10.2:50052`
- **Arms**: `mac` = `-ts 0` (Metal only) · `spark` = `-ts 1.00` (**all layers on the Spark,
  driven from the Mac over RPC** — not a native local Spark run) · `pair` = `-ts 1.00/1.00`
  (50/50). `-ts` lists the RPC device first throughout this repo.

**Read the `spark` arm carefully.** It is the Spark holding every layer with the Mac as
client, so it still pays one round trip per token and is *not* the same measurement as the
native GB10 solo rows elsewhere in this README. It is the right control for the `pair` arm
and the wrong one for a silicon comparison.

## Set A — `datasets/macspark-20260918-115032/`, pp512 + tg128, **3 repetitions**

| arm | pp512 | tg128 |
|---|---:|---:|
| Mac alone | **1056.24 ± 8.41** | **39.23 ± 0.56** |
| Spark via RPC | 806.83 ± 4.24 | 26.99 ± 0.14 |
| pair 50/50 | 835.28 ± 1.74 | 26.20 ± 0.27 |
| **pair ÷ Mac** | **0.791** | **0.668** |
| **pair ÷ Spark** | **1.035** | **0.971** |

## Set B — `datasets/macspark-20260918-120154/`, pp512/2048/4096 + tg128, **2 repetitions**

| arm | pp512 | pp2048 | pp4096 | tg128 |
|---|---:|---:|---:|---:|
| Mac alone | **1064.03 ± 8.59** | 1043.26 ± 10.10 | 964.71 ± 3.92 | **37.88 ± 0.87** |
| Spark via RPC | 809.80 ± 0.68 | 828.03 ± 3.10 | 819.94 ± 5.17 | 26.98 ± 0.09 |
| pair 50/50 | 842.75 ± 0.59 | **1089.97 ± 6.79** | **1143.03 ± 0.47** | 26.95 ± 0.29 |
| **pair ÷ Mac** | 0.792 | **1.045** | **1.185** | 0.711 |
| **pair ÷ Spark** | 1.041 | **1.316** | **1.394** | 0.999 |

± is the sample s.d. **across repetitions** (each repetition is a separate `llama-bench`
invocation with `r=1`), matching the convention in the other tables here. n=3 and n=2 are
small; these are not confidence intervals.

**What these two sets say.**

- **Prefill crosses parity between 512 and 2048 and keeps climbing.** 0.79 → 1.05 → 1.19
  against the Mac. This *reproduces* the 17 Sep Flash-Next result (0.722 / 1.003 / 1.128)
  on a quiet machine with matched builds, and lands slightly higher at every point. Two
  independent runs now agree on the direction and roughly on the magnitude.
- **Generation never wins, and this time it does not even beat the slower box.** 0.67-0.71
  of the Mac, and 0.97-1.00 of the Spark — a statistical tie with the *slower* arm. On this
  model the split costs generation outright.
- **The two sets agree with each other** where they overlap (pp512 0.791 vs 0.792; Mac
  pp512 1056 vs 1064, 0.8 % apart), which is the cheapest available check that neither is a
  fluke.

**Limitation that applies to both sets and is not in the 17 Sep convention.** The arms ran
in **fixed order** — `mac`, `spark`, `pair`, every round — not rotated. The 17 Sep datasets
rotated the order precisely so drift would show up inside the data. These do not, so a
monotonic drift over a round would be silently absorbed into the arm ordering. The two sets
agreeing across a 12-minute gap is reassurance, not a substitute.

## Set C — the `-ts` ratio sweep, `datasets/tssweep-20260918-*` — **n=1 per point**

The 17 Sep section of the README says plainly that we never swept the split ratio. This is
that sweep. Flash-Next, pp2048 and tg64, **one invocation per ratio, no repetition.**

| `-ts` spark/mac | 60/40 | 50/50 | 43/57 | 35/65 ┊ | 30/70 | 25/75 | 20/80 | 15/85 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pp2048 | 789.33 | 1086.65 | 1176.48 | **1282.93** ┊ | 1217.64 | 1169.55 | 1115.66 | 1085.85 |
| tg64 | 24.48 | 28.68 | 29.29 | 29.97 ┊ | 33.79 | 34.09 | 34.70 | **35.24** |

- **Prefill has an interior optimum at `-ts 35/65`**, 1282.93 tok/s — **1.23× the Mac alone**
  at pp2048 (1043.26, Set B) and 19 % above the 50/50 the rest of this repo uses. The
  script's *predicted* optimum of 43/57 was **wrong**; the measured peak is Mac-heavier.
- **Generation rises monotonically toward Mac-heavy** and is still climbing at the edge of
  the sweep — consistent with generation being bandwidth-bound and the Mac being the faster
  box, but it **never reaches the 37.9 tok/s the Mac manages alone.**

**┊ marks a batch boundary, and the `tg64` row must not be read across it.** The eight
points are two separate four-point runs three minutes apart (`…-125139`: 60/40 → 35/65;
`…-125755`: 30/70 → 15/85). `tg64` steps by **+3.82** across that boundary while the largest
step inside either batch is +0.68. That is a between-batch offset, not a trend. **The
prefill row does not show it** (its boundary step is smaller than its neighbours' and in
the direction of the peak), so the prefill optimum survives; the generation row is only
safe to read *within* a batch.

**Also: n=1 per point, and no matched Mac-alone `tg64` baseline was taken** (the 18 Sep solo
arms used `tg128`). Treat this whole set as a **screening sweep that located a direction**,
not as a measured optimum. The next run is 35/65 against 50/50 with repetitions and rotation.

## Set D — the multistream / concurrency attempt: **FAILED, kept as a failure**

`datasets/multistream-20260918-{121945,122029,123521}/`. Three attempts at an N-concurrent-
stream ladder against `llama-server`. **No usable pair-versus-Mac comparison came out of
any of them**, and they are committed for the traps rather than for numbers.

| attempt | what happened |
|---|---|
| `121945` | every stream failed in both arms. Both server logs stop at `load_model`. |
| `122029` | Mac arm allocated **8 slots × `n_ctx_slot` 164864** and logged **2650** `kIOGPUCommandBufferCallbackErrorOutOfMemory` lines; N=4 returned 0.35 tok/s, N=8 failed. Pair arm ran at `n_ctx_slot` **262144** — a different context entirely, so the two arms are not comparable. |
| `123521` | Mac arm is the only clean one (`n_ctx_slot` 2048): 33.45 → 41.46 → 54.43 → **92.23** tok/s aggregate at N = 1/2/4/8. Pair arm N=1 failed, N=8 completed 4 of 8 streams at 3.67 tok/s. |

**And one result in `123521` cannot be attributed at all.** Its `server-pair.log` is 647
bytes and stops at `load_model` — that server never finished loading — yet the driver
recorded pair N=2 and N=4 figures. The previous attempt's pair server (pid 85213) was
plausibly still holding the port. **Which process produced those numbers is UNKNOWN**, so
they are not quoted anywhere and should not be lifted out of the log.

**The two traps worth carrying away**, which cost real time and are the reason this
directory is not empty:

1. **`llama-server` slot context is per slot, not shared.** `--parallel 8` at a large `-c`
   asks for eight full contexts and OOMs Metal. The Mac arm only worked once `n_ctx_slot`
   came down to 2048.
2. **A harness that reports numbers while its server is dead is worse than one that
   crashes.** Check the server's own log for a completed load before believing a row.

The one thing `123521`'s Mac arm does show — **2.8× aggregate throughput from 1 to 8
concurrent streams on a single machine** — is a single unreplicated pass on one arm and is
reported here only as an observation, not as a result.
