# Three boxes, one model, only the layer allocation changes (17 Sep 2026)

> **TRANSPORT: TCP throughout. No RDMA anywhere in this series**, including between the two
> Sparks — see the topology note below, which is the part most likely to be misread.
> [`TRANSPORT.md`](../../TRANSPORT.md).

A controlled series: one client, one GGUF, one commit, one set of flags. **The only thing
that changes between rows is which devices hold layers.** Runner: `runners/series.sh`
(`runners/run3box.sh` is the earlier driver kept with it).

| row | `-ts` (RPC devices FIRST) | who holds weights |
|---|---|---|
| 1 `mac` | *(none)* | MacBook only, Metal, no RPC |
| 2 `spark1` | `1/0` | Spark 1 only. Mac orchestrates and samples, **holds no weights** |
| 3 `sparks2` | `1/1/0` | both Sparks. Mac orchestrates only |
| 4 `all3` | `1/1/1` | all three |

- **Model**: `Qwen3.8-Flash-Next-UD-IQ4_XS`, `model_size` 93,671,559,680 (87.24 GiB),
  **176,943,899,520 params, 3 B active**, arch `qwen4exp`. Fits the Mac *and* one Spark
  alone, which is what makes rows 1 and 2 real baselines rather than capacity workarounds.
- **Build**: `434ddbbc0` (846) — the *same* commit on both ends, unlike the 17 Sep split
  set, where the Mac was on `1d0c76f3c`.
- **Flags**: `-p {512|2048|4096} -n 128 -r 3 -o json`. `n_batch 2048, n_ubatch 512,
  n_threads 6, type_k/v f16, flash_attn -1, split_mode layer`.
- **Reps**: `r=3` **inside one invocation**. The ± below is llama-bench's own stddev across
  those three, which is a *within-invocation* spread. It does not capture between-invocation
  drift, and the next section shows that drift is the larger effect here.

## The topology note that decides how to read row 3

llama.cpp RPC is a **star**: the client on the Mac holds a connection to each server, and
the servers do not talk to each other. So in rows 3 and 4 the two Sparks exchange nothing
directly — **every activation goes Spark → Mac → Spark**, over the 10 GbE cable to Spark 1
and over a second IP path to Spark 2 (`192.168.100.11`).

**The 200 GbE RoCE fabric between the Sparks is not carrying any of this**, even though the
addresses used are on that subnet. The 185 Gbit/s figure in
[`../roce-2026-09-17/`](../roce-2026-09-17/) is a synthetic `ib_write_bw` result between
idle boxes and has nothing to do with these rows. This is our reading of llama.cpp's RPC
architecture, corroborated by the absence of any `RDMA probed` / `RDMA activated` line in
the committed `.err` files — `grep -ci rdma datasets/*.err` returns 0 — against the
[`../rpc-rdma-2026-09-17/`](../rpc-rdma-2026-09-17/) capture where those lines *are* present.

## Results — MEASURED by us, tok/s, mean ± within-invocation s.d. (r=3)

| row | pp512 | pp2048 | pp4096 | tg128 |
|---|---:|---:|---:|---:|
| 1 Mac alone (17:30) | 1010.60 ± 3.34 ‡ | 985.09 ± 16.61 ‡ | **1015.10 ± 11.54** | **41.41 ± 0.12** |
| 2 Spark 1 (`1/0`) | 818.41 ± 18.38 | *not run* | 824.02 ± 1.06 | 27.33 ± 0.06 |
| 3 both Sparks (`1/1/0`) | 686.07 ± 15.05 | *not run* | **927.53 ± 2.87** | 23.49 ± 0.16 |
| 4 all three (`1/1/1`) | 626.28 ± 22.91 | *not run* | 652.45 ± **89.28** | 20.72 ± 0.56 |

‡ **These two cells are not time-matched to rows 2-4 and must not be used for ratios.**
See the drift section immediately below.

**Two findings, and one of them is negative:**

1. **A second Spark helps prefill at long context and nothing else.** 824 → **928** tok/s at
   pp4096 going from one Spark to two (+12.6 %), while pp512 *falls* 818 → 686 (−16 %). Same
   shape as everything else in this repo: a long prompt gives both boxes a large batch of
   independent work; a short one just pays the crossings.
2. **Adding the Mac as a third weight-holder makes everything worse.** Row 4 is below row 3
   on every column, and its pp4096 error bar (± 89.28, **13.7 % of the mean**) is by far the
   worst in the series. Three boxes on this stack is not a configuration we would hand anyone.
3. **No multi-box row beats the Mac alone.** 1015 pp4096 and 41.41 tg128 on one laptop
   against 928 / 23.49 for the best pair. Generation degrades monotonically with every box
   added: **41.41 → 27.33 → 23.49 → 20.72**. On a 3-B-active MoE that fits one machine, the
   cable is a cost, not a capability.

## Drift, stated because it invalidates part of the table

The Mac baseline was measured **twice at pp4096, 28 minutes apart, on the same machine with
the same command**:

| when | pp4096 | tg128 | file |
|---|---:|---:|---|
| 17:02:01Z | 881.23 ± 28.66 | 35.11 ± 0.27 | `qwen-1-mac-p4096-20260917T170201Z.json` |
| 17:30:26Z | **1015.10 ± 11.54** | **41.41 ± 0.12** | `qwen-1-mac-p4096-20260917T173026Z.json` |
| | **+15.2 %** | **+17.9 %** | |

Both are committed. **Neither is discarded**, and neither is an outlier inside its own
invocation — the within-run s.d. on both is a few percent. The machine was in different
states, and the 17:02 pair was taken while the box was busier.

**Consequence: use the 17:30 Mac row.** Rows 2-4 ran 17:30-17:38, adjacent to it. The
17:02 row (881 / 35.11) and the pp512 / pp2048 Mac cells (16:58-17:02) are from the earlier,
slower window and are **not comparable to rows 2-4**. This is exactly why the repetition
count inside an invocation is the wrong thing to trust on its own.

`qwen-4-all3-p512` was also run twice (17:25:54Z → 626.28, 17:28:10Z → 638.63, 2 % apart),
which is the normal spread and is reassuring about the rows that *are* time-matched.

## Files

`datasets/qwen-{row}-{tag}-p{512|2048|4096}-{UTC}.json` is the `llama-bench -o json`
output, carrying `build_commit`, `model_filename`, `model_n_params`, `n_batch`, `n_ubatch`,
`tensor_split`, `test_time` and the per-repetition `samples_ts`. The matching `.err` is the
**complete** stderr for that invocation.

**The `.err` files carry no allocation receipt.** `series.sh` says so out loud rather than
printing an empty header that reads as a pass: `llama-bench` only emits `load_tensors`
buffer-size lines with `-v`, which these runs did not pass. **Placement in this series is
asserted by `-ts`, not verified from the allocator.**
