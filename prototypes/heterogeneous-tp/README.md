# Heterogeneous tensor parallel: first correctness gate

Owner: Codex (shared harness/Mac); ClaudeMM (AMD readiness); ClaudeMB (separate Mia Spark control).

`smoke.py` computes a two-layer GELU MLP with the hidden dimension sharded across ranks. Each device computes only its shard; the output partials move explicitly to CPU for a Gloo SUM. Output bias is added once. Canonical inputs and weights come from rank zero, avoiding cross-version random-number assumptions. Each rank validates against an unsharded CPU reference for token counts 1, 17 and 128. Failure returns a nonzero exit status. GPU requests fail rather than silently falling back to CPU.

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

Next gates: physical Mac + Spark pair; AMD local test plus physical mixed pair; transformer block; small model. Measure performance only afterward with matched model, precision, context, input/output lengths, warmups and repeated trials. The user's target is both prefill and generation faster than the fastest single host; no current result establishes that.
