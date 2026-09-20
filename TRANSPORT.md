# Which link carried which number

> **Read this before quoting anything from this repo in an RDMA context.**
>
> **Every inference number in this repository that involves the Mac was carried by TCP
> over Ethernet. None of it is RDMA.** The Mac in this bench has no RDMA-capable NIC in
> it at all, so no configuration change could have made it RDMA. If you are comparing
> against an RDMA transport, our Mac numbers are the *baseline you are trying to beat*,
> not a competing RDMA result.

This file exists because this repo is read alongside projects whose whole subject is
RDMA ([MCDMA](https://github.com/ashhart/MCDMA), [MelonDMA](https://github.com/b-ostrov/MelonDMA)),
and a sockets number filed next to an RDMA number becomes an RDMA number within one
retelling. Transport is recorded here **per run**, not per machine pair, because the same
pair of machines was run over different transports on the same day.

## The three links on this bench

| link | endpoints | transport actually used | RDMA? |
|---|---|---|---|
| **10 GbE** | MacBook `en12` 10.10.10.1 ↔ Spark 1 `10.10.10.2` | TCP | **No, and cannot be.** See below |
| **Thunderbolt** | MacBook ↔ Bosgame M5 (Strix Halo), Helsinki | IP over Thunderbolt | **No** |
| **200 GbE** | Spark 1 ↔ Spark 2, one QSFP56 DAC, ConnectX-7 | RoCEv2 *or* TCP, **varies by run** | **Sometimes.** Read the run |

## The Mac link cannot be RDMA, and that is a hardware fact

Checked on the MacBook on 2026-09-18 (read off the machine, not recalled):

```
network adapters   Thunderbolt Ethernet Slot 1 (en12), AX88179B USB,
                   USB 10/100/1000 LAN, Wi-Fi
NO ConnectX, NO RDMA-capable NIC on the Mac side.
```

`en12` is a Thunderbolt-to-10GBASE-T Ethernet adapter. Everything the Mac sends to a
Spark goes over IP on that adapter. This is why the Mac↔Spark link measures in **hundreds
of microseconds** where the Spark↔Spark RoCE fabric measures in **single-digit
microseconds**.

**Positive and negative control for this claim, both in this repo.** llama.cpp's RPC
auto-negotiates RDMA when both ends carry it and says so in its log:

- *Positive* — Spark↔Spark, [`benchmarks/rpc-rdma-2026-09-17/`](benchmarks/rpc-rdma-2026-09-17/)
  prints `RDMA probed: dev=rocep1s0f0 gid=5 RoCEv2` and `RDMA activated: qpn=…`.
- *Negative* — every Mac-client run added in this update
  ([`split-2026-09-18`](benchmarks/split-2026-09-18/), [`three-box-2026-09-17`](benchmarks/three-box-2026-09-17/))
  prints **no such line**, in logs captured at the same verbosity. `grep -ci rdma` over
  those logs returns 0.

So the Mac-side sockets claim rests on a matched pair of observations, not on absence alone.

## The two-Spark GLM serve: transport differs BY RUN

Both of the following are from 2026-09-18 on the **same two boxes** with the **same model**.
They are not the same run and they did not use the same transport. Neither has a raw
artifact in this repo; both are reported from our own working record and are labelled as
such.

| run | reported by | NCCL setting | transport | notes |
|---|---|---|---|---|
| GLM-5.3-Flash-EXL3, DFlash 7, MTP 2, ctx 8192, `max_num_seqs 2` | **codexmb** | `NCCL_NET=Socket`, `NCCL_IB_DISABLE=1` | **TCP — explicitly NOT an RDMA success** | fell back after `ibv_reg_mr_iova2` returned **ENOMEM** |
| GLM-5.3-Flash-EXL3-TR3-4bpw, PP=2/TP=1, speculation OFF, ctx 8192, `max_num_seqs 8` | claudeMB | `NCCL_NET=IB`, `NCCL_IB_DISABLE=0`, GID 6 head / 5 worker | **RoCEv2** | stock vLLM `0.1.dev20051+g487ecf187`, image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-instanttensor` |

**Do not generalise either row to "the Spark pair".** The correct question is always which
run, on which date, with which `NCCL_NET`.

**State as of 2026-09-20:** RDMA on that fabric is failing again with
`ibv_reg_mr_iova2: Cannot allocate memory`, and the GLM serve is on TCP. The suspected
cause is a kernel-level GPUDirect/registration regression on `7.0.0-1019-nvidia`, with the
two boxes differing in whether they boot with `kho=off`. **That is a suspicion, not a
diagnosis** — confirming it needs a kernel log we have not captured. A statement about
this fabric is a snapshot, not a property.

## What each grade of label in this repo means

| label | meaning |
|---|---|
| **MEASURED by us** | run on our machines, raw artifact committed here, date + build commit + repetition count stated |
| **MEASURED by codexmb** | run by codexmb on our hardware and independently verified by them; **no raw artifact in this repo** |
| **NOT OURS** | someone else's published or reported figure, under their conditions, which we did not run |
| **IDEA / untested** | a hypothesis, a prediction, or an arithmetic projection from measured inputs |
| **CONTROL** | a baseline run whose job is to make another number interpretable — not itself the result being sought |
| **UNKNOWN** | conditions we cannot state. Left unknown rather than guessed |
