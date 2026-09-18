# Physical rolling-pipeline receipts — Mini(Metal) ↔ M5(ROCm), 2026-09-18

Requested by @codexmb for lead review. Everything here is the run as executed, including
the calibration I discarded and why.

## Identity
```
rolling.py source_sha256   273c9ffb3b74b9e0…   (codexmb c21bee9, unmodified)
gpt2 weight_sha256         248dfc3911869ec4…   pinned 124M checkpoint
rank 0  Mac mini M4, macOS, --device mps      192.168.50.241
rank 1  Bosgame M5, Linux, --device rocm      192.168.50.127
transport  TCP over home gigabit LAN
```

## Naming: "b32 c4" means --batch 32 --chunks 4

`--batch` is the TOTAL sequences; rolling.py computes `b = args.batch // args.chunks`
(line 254) and rejects any batch not divisible by chunks (line 187). So b32 c4 is
**32 total sequences in 4 chunks of 8**, and the C=1 control is **32 total in one chunk**.
Total work is therefore identical across every row of the comparison — that equality is
the whole point of the control.

## Trial order
```
forward   c1-whole-b32 → alt-b32-c4 → barrier-b32-c4 → rolling-b32-c4
reversed  r-rolling → r-barrier → r-alt → r-c1-whole
```
Run reversed because the M5's load fell across the forward pass (7.36 → 4.10), which
favoured whichever schedule ran last. Reversing changed the rolling result materially
and left the others stable — see below.

## Result, pooled over BOTH orderings (n=6 per schedule)
```
  schedule              median      min      max
  C=1 whole batch 32    1469.2   1298.6   1583.2   <- fastest
  rolling   b32 c4       987.6    351.3    995.1
  barrier   b32 c4       823.8    806.9    844.2
  alternating b32 c4     470.0    429.3    512.8
```
Every row reported `valid=true`: codexmb's independent Hugging Face reference passed on
all 9 configurations. This is its first run on real heterogeneous hardware, not loopback.

**Not established: rolling vs barrier.** Forward, rolling was 995.1 with a 2 tok/s spread;
reversed it was 687 with a 351–982 spread, while barrier was stable in both. An earlier
claim of mine that rolling beat barrier by 19% is withdrawn.

**Robust: the C=1 control wins in both orderings** (1545 forward, 1465 reversed).

## Calibration (alternating, batch 16, chunks 1)
```
  Mini layers   2    530.45      6    618.01     10    933.42 tok/s
                4    546.08      8    734.79
```
Monotonic: the Mini is the FASTER stage for GPT-2 124M, the reverse of Qwen-27B where
every layer given to it costs throughput. The comparison used split 10, which is the
**alternating-optimal, not a measured balance point** — the stages were never timed
separately, so this is a fair split rather than rolling's best case.

## Paging deltas (why they are here)

Sampled per row because a cumulative swap total proves nothing about a given run.
Every row shows `swapout+0` except one:
```
  r-alt (alternating, reversed)   swapin+14093  swapout+12741
```
So the Mini does hit real memory pressure occasionally even on a 124M model, and it is
now visible per row instead of inferred.

## excluded-contaminated/ — do not use these numbers

An earlier calibration is preserved there and excluded. I edited the runner script while
an instance was still executing; bash reads scripts incrementally from a byte offset, so
the live instance resumed mid-file, fell into the `compare` branch, and ran a second
benchmark concurrently with the calibration. Values from that window were 8–18% off
(e.g. split 2 read 480.7 contaminated vs 530.5 clean; split 10 read 790.0 vs 933.4).
Both later passes were run from frozen copies, included here as
`run_rolling.frozen.sh` and `run_rolling.reversed.sh`.
