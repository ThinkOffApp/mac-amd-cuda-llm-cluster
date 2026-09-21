# GPT-2 communication optimization experiments

Baseline: `a710efc`, matched Mini/M5 timing receipts in
`results/timing-20260918`. These experiments retain the FP32 tensors, root SUM,
per-token root broadcast, exact HuggingFace output-ID gate and global validity
verdict. Neither experiment establishes a physical-pair speedup yet.

## Last-position, root-only generation head

Pass `--generation-head last-root` to `bench.py`; default `baseline` retains the
original behavior for paired comparisons. Only rank 0 computes the final layer
norm and tied vocabulary projection, only for the final position. Argmax stays
on the device and only its scalar ID is copied to CPU. The authoritative token
broadcast remains, so we do not assume independently computed GPU logits choose
the same token. Solo receives the same optimization for a fair comparison.

At a 512-token prompt, the baseline transfers 512 * 50,257 * 4 = 102,926,336
logit bytes per rank from device to host during prefill. The optimized path
transfers a scalar on rank 0 and no logits on rank 1. This is device/host traffic,
not inter-machine traffic: all 24 layer reductions and the token broadcast
remain. The projection also performs less arithmetic. Real timings are required
to quantify either benefit. Different GEMM shapes can change floating-point
rounding, so every measured output still must match the independent reference.

## Optional profile

Pass `--profile` for diagnostics, separately from performance trials. Reports
withhold `timings` (including throughput) and expose `profile_runs` only after
both ranks pass. Each measured generation resets counters after warmups and the
outside-timer barrier. Each run separates prefill and decode, including actual
wall times. Profiled runs add GPU synchronization and Python overhead: neither
the totals nor the shares estimate the uninstrumented run exactly.

Compute regions include embedding, attention (including norm, QKV, cache,
masking, softmax and projection), MLP, final residual and the head. The head
region includes the baseline's logit copy; the optimized scalar copy is in token
selection. Layer reduction copies are separate device-to-host/host-to-device
regions. Unclassified wall time includes Python/measurement overhead and token
timestamp synchronization. Do not normalize selected regions and label the
result the complete runtime budget.

`collective` includes serialization, reduction, socket I/O and peer readiness.
`socket_send` and `socket_receive` are nested inside collective/broadcast times;
never add them again. A blocked socket receive may mean the peer is still
computing, not that the network is slow. Send completion does not establish peer
receipt. Socket byte counts are tensor payloads excluding framing. Collective
payload counts are input tensor sizes, not total bidirectional traffic. Never
sum elapsed times across ranks as if they were one serial timeline.

## Validation and next hardware gate

Set `GPT2_DIR` to the pinned GPT-2 checkpoint and run:

```sh
python validate_bench_profile.py --out /path/to/receipts
python -m unittest test_tcp_collectives.py
```

The integration matrix checks solo/two CPU ranks, baseline/last-root and prompt
lengths 128/512. Each has a warmup and two recorded runs with eight generated
tokens; it checks independent reference IDs, per-run counter reset, reduction
counts, and exact per-rank sent/received payload totals. These are local
correctness checks, not Mini/M5 or Mac/Spark speed evidence.

Next: run baseline/last-root with profiling off, interleaved five trials per
configuration, two warmups, 64 new tokens, prompt lengths 128/512, both solo
hosts and the pair. Run separate profiles of each configuration in the same
coordinated window. Preserve full config, hashes, IDs and both-rank reports,
then restore and verify affected services. Larger model experiments require
architecture-specific adapters or a supported engine and a separate memory
plan; changing a checkpoint name does not make this GPT-2 adapter support 27B
or Flash-Next.


## Actual Flash-Next experiment

[Flash-Next receipts](results/engine-flash-next-20260918/README.md) extend the large-model investigation to actual UD-IQ4_XS weights on one M5 Max. A local engine patch fixes a CPU-view handling crash; synthetic prefill and incremental tests pass. Actual single-partition tensor mode matches solo logits exactly. Two local partitions match 48/48 choices but fail the unchanged full-logit gate. A routing trace and diagnostic intervention identify a near-tied expert selection as the first discontinuity. These are correctness diagnostics, not RPC, physical-pair performance or a production support claim. The normal helper never forces routing; the intervention source is preserved only with its clearly labelled experimental receipts.
