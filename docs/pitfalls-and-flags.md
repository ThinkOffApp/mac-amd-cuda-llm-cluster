# The non-obvious flags, and the pitfalls that cost us time

Read this before your first split run. Each item here cost real time at least once.

> Moved verbatim from the [README](../README.md) on 23 Sep 2026 to keep the README short.
> Nothing was cut. Where the text says "above", "below" or "this README", it refers to
> the README as it was before the move; the [README's last section](../README.md#details-and-all-numbers)
> lists where each part now lives.

## The non-obvious flags

- `ggml-rpc-server` started plain offers only the GPU. At the factory 64 GB carve-out,
  **`-d ROCm0,CPU`** made it offer both devices (~125 GiB instead of 64), which is how the
  200 GB model first fitted on two boxes in August. At the 1 GB carve-out the Vulkan device
  alone offers ~119 GiB and the CPU device is no longer worth routing through (it halved
  prompt speed in our Q3 tests).
- **`-ts` lists the RPC device first.** `-ts 15/85` means 15 % on the Strix, 85 % on the Mac.
  A comma (`-ts 15,85`) is a list of separate runs, not a ratio, and pushed a 200 GB file whole
  into the 122 GB box (a 2-hour hard hang). Use the slash, and size-check the RPC share before
  every run (`scripts` in StrixLink carry a guard).
- The default placement is the slow one: see the table. Put the layers where the bandwidth is.
- Build the Strix `ggml-rpc-server` with **`GGML_RPC_RDMA=OFF`** unless both ends carry the same
  RDMA transport; a mismatch aborts every split load mid-way with no useful message.
- **When both ends DO carry it, llama.cpp's RPC finds it by itself.** Between the two Sparks the
  stock `434ddbbc0` build announces `transport: TCP (RDMA auto-negotiate enabled)` and then, per
  client connection, `RDMA probed: dev=rocep1s0f0 gid=5 RoCEv2` / `RDMA activated: qpn=N->N
  mtu=1024`. So the control channel is TCP and the data path is RDMA over RoCEv2, without a flag
  at run time. `libggml-rpc.so` links `libibverbs.so.1`, which is the compile-time half of the
  same fact. Logs, binary hashes and both ends' output:
  [`benchmarks/rpc-rdma-2026-09-17/`](../benchmarks/rpc-rdma-2026-09-17/). This says nothing about
  whether it is *faster* for inference: no model has loaded across that pair yet.
- The server's `-c` file cache makes warm restarts ~64x faster but writes every shard of every
  run to `~/.cache/llama.cpp/rpc` (441 GB after one day of sweeps). Use it, and clear it.

## Honest pitfalls (each cost us real time)

- **Metal OOM presents as `res = -3`** with the true cause
  (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) hidden unless you pass
  `-v` — llama-bench's default verbosity filters even error-level log lines
  (upstream issue ggml-org/llama.cpp#28107).
- A model file's NAME is not its contents: unsloth UD-IQ3_XXS ships IQ3_S
  expert tensors and zero IQ3_XXS ones. Read the tensor table before
  reasoning about kernels.
- The RPC server wedges under connect storms; supervise it
  (`Restart=on-failure`) rather than discovering it dead mid-bench.
- Measure the memory ceiling PER PATH: the same box offered us 64 GiB over
  RPC and ran an 82 GB model locally, on the same afternoon.
- A bad split does not fail, it just measures slowly: 92 GB on the Mac gave
  pp 20 with a ±6 error bar and a Metal OOM on the generation test. Read the
  buffer-size lines in the load log before trusting a row.
- A watchdog that says "ssh down" during a split is usually a box at load
  average 20 answering slowly; verify on both addresses before reacting.
- **Never grade a reasoning model on whether it produced visible output.** A
  harness that keys "did it work" on answer text will silently mis-grade every
  reasoning model, because a run can spend its entire token budget inside the
  reasoning block and return empty content with `finish_reason: length`. It bit
  us twice on 20 Sep 2026: a GLM-5.3-Flash-EXL3 measurement produced 384
  completion tokens, 1634 reasoning characters and **zero answer characters**
  (see the two-Spark section above), and separately a freshly loaded reasoning
  model given a 40-token cap on the MacBook returned empty content and was
  nearly recorded as a broken model. **Key success on completion tokens
  produced, and treat `finish_reason: length` WITH tokens as a success, not a
  failure.** Then keep the two quantities apart in the write-up: *generation
  rate* and *completed-task throughput* are different measurements, and a run
  that emits no answer measures only the first.
