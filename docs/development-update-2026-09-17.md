# Development update: Mac + AMD + CUDA tensor parallelism

Status of the tensor-parallel work as of 17 September 2026: what is implemented, the first failed Mini-AMD attempt, and the next gates.

> Moved verbatim from the [README](../README.md) on 23 Sep 2026 to keep the README short.
> Nothing was cut. Where the text says "above", "below" or "this README", it refers to
> the README as it was before the move; the [README's last section](../README.md#details-and-all-numbers)
> lists where each part now lives.

### Development update — 17 September 2026

We are now developing **Mac + AMD + CUDA tensor parallelism**, beginning with
Mac correctness tests and AMD support alongside them. This is separate from
the existing llama.cpp RPC benchmarks below. The performance target is both
prefill and generation faster than the fastest single host under matched
model, precision, context and workload; we have not demonstrated that target.

- **Implemented and locally tested:** a [sharded FP32 MLP harness](../prototypes/heterogeneous-tp/README.md)
  with local MPS/CUDA/ROCm device selection and explicit CPU-staged Gloo
  collectives. Two local MPS ranks on one Mac passed at 1, 17 and 128 tokens
  against an unsharded CPU reference (maximum absolute error `9.54e-7`).
  The two-rank CPU control also passed. These are two processes on one machine,
  not a successful cross-host GPU test or a speed benchmark.
- **First Mini–AMD attempt:** initialization failed before GPU computation,
  with a Gloo UV address-size mismatch (`136 vs 177`), using Mini PyTorch
  2.11.0 and AMD PyTorch 2.12.0a0. PyTorch's platform defaults select different
  Gloo transports. Transport availability and wire compatibility must be
  checked in the actual builds; this is not proof that every Mac–Linux Gloo
  configuration is impossible. See the [upstream device factory](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/distributed/c10d/GlooDeviceFactory.cpp).
- **Portable transport investigation:** the Helsinki team reports a working
  two-host CPU socket reduction probe. Its raw results and framing need review
  before inclusion as benchmark data. This does not yet validate an integrated
  MPS–ROCm model run, accelerator transfer costs, or negligible transport overhead.
- **Spark–Spark control:** the MiaLab GLM-5.3-Flash EXL3/DFlash recipe has been
  attempted. The latest diagnosed failure was worker-side DFlash weight loading;
  the worker exited while the head container remained running without a healthy
  API. Repair/relaunch is underway. No successful served-token result from this
  recipe is recorded here yet.

Next gates are a physical mixed-host correctness run, a transformer block,
a small complete model, then repeatable end-to-end timing. Experimental RDMA
drivers and hardware compatibility are separate work; they are not prerequisites
for the first correctness gate.
