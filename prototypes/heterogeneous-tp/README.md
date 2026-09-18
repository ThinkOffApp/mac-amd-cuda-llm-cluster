# Heterogeneous tensor parallel: first correctness gate

Owner: Codex (shared harness/Mac); ClaudeMM (AMD readiness); ClaudeMB (separate Mia Spark control).

`smoke.py` computes a two-layer GELU MLP with the hidden dimension sharded across ranks. Each device computes only its shard; the output partials move explicitly to CPU for a SUM using Gloo or the framed TCP backend. Output bias is added once. Canonical inputs and weights come from rank zero, avoiding cross-version random-number assumptions. Each rank validates against an unsharded CPU reference for token counts 1, 17 and 128. Failure returns a nonzero exit status. GPU requests fail rather than silently falling back to CPU.

This is a correctness harness, not an inference engine or performance benchmark. All ranks retain full CPU weights for the reference. It does not yet implement attention, KV cache, a full transformer, asynchronous communication, RDMA, or efficient weight loading. Local tests do not establish cross-host or mixed-build interoperability.

Install a suitable PyTorch build in an isolated environment on each host; use the same version where possible. ROCm uses PyTorch's `cuda` API internally, but selecting `rocm` checks that this is a HIP build. Distributed backend and MPS documentation: https://docs.pytorch.org/docs/stable/distributed.html and https://docs.pytorch.org/docs/stable/notes/mps.html .

Local Mac test (two processes share one GPU; not two physical devices):

```sh
python -m torch.distributed.run --standalone --nproc-per-node=2 smoke.py --device mps
```

CPU control: use `--device cpu`. For two hosts, run once on each with matching rendezvous address/port and run ID; use `--node-rank=0` on Mac and `--node-rank=1` on the other host:

```sh
TP_DEVICE=mps python -m torch.distributed.run --nnodes=2 --nproc-per-node=1 \
  --node-rank=0 --master-addr=REACHABLE_MAC_IP --master-port=29517 smoke.py
TP_DEVICE=rocm python -m torch.distributed.run --nnodes=2 --nproc-per-node=1 \
  --node-rank=1 --master-addr=REACHABLE_MAC_IP --master-port=29517 smoke.py
```

For a Spark rank use `TP_DEVICE=cuda`. Each host needs the same script. Set `GLOO_SOCKET_IFNAME` to the host's intended reachable interface if automatic selection is wrong. Use only a trusted private network; Gloo transport has no application authentication. No public listener or firewall changes are required by this harness. Collect the rank-zero JSON and both process exit codes.

Next gates: AMD local test plus physical Mini–AMD pair; transformer block; small model. Measure performance only afterward with matched model, precision, context, input/output lengths, warmups and repeated trials. The user's target is both prefill and generation faster than the fastest single host; no current result establishes that.


## Portable TCP backend (two ranks)

`--transport tcp` avoids Gloo wire-format dependencies. Run directly with Python
(no torchrun rendezvous needed). Use the Mac's actual private interface address;
the listener binds only that address. Rank 1 retries connection for up to 60 seconds.

```sh
# Mac
RANK=0 WORLD_SIZE=2 MASTER_ADDR=MAC_PRIVATE_IP MASTER_PORT=29625 \
  python smoke.py --device mps --transport tcp
# Spark (same files, CUDA-enabled PyTorch)
RANK=1 WORLD_SIZE=2 MASTER_ADDR=MAC_PRIVATE_IP MASTER_PORT=29625 \
  python smoke.py --device cuda --transport tcp
```

For AMD use `--device rocm`. This backend supports exactly two ranks and FP32
CPU tensors. Headers carry version, collective sequence, operation, dtype, shape
and length; payloads are little-endian float32 with a 64 MiB limit. Report exchange
uses JSON, never pickle. Protocol violations, truncated frames and timeouts fail
the run. It is unauthenticated: use a trusted private link, not a public listener.
The reduction is a simple exchange through rank zero; it is not optimized RDMA.

Validated on 18 September (Berlin time): two local CPU ranks, two local MPS ranks,
and physical Mac MPS (PyTorch 2.14.0) + Spark CUDA (2.13.0+cu130). All three token
counts passed on both ranks; physical pair maximum absolute error `9.54e-7`.
Both physical processes exited 0; Spark serving health remained HTTP 200.
Six protocol tests check byte order, sequence, shape, payload length, oversized
headers and early EOF. No measured speedup or Mini–AMD result is claimed.
