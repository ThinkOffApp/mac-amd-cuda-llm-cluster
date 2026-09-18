"""FP32 tensor-parallel decoder block, correctness against an unsharded reference.

Sharding follows the Megatron column/row pattern:
  attention   QKV column-parallel by HEAD, output projection row-parallel
  MLP         w1 column-parallel by HIDDEN CHANNEL, w2 row-parallel
  norms/residual replicated: every rank holds the full hidden state
Row-parallel outputs are summed with all_reduce; their biases are added ONCE,
after the sum, or they would be counted world_size times.

Full CPU weights are retained for the reference; this is not a serving engine.
GPU math is local; every collective receives CPU tensors explicitly.
"""
import argparse
from datetime import timedelta
import json
import os
import platform

import torch
import torch.distributed as dist
import torch.nn.functional as F


def rms_norm(x, weight, eps=1e-5):
    # Replicated: identical inputs on every rank, so no collective is needed.
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def attention(q, k, v, heads, head_dim):
    # q/k/v: (tokens, heads*head_dim) -> per-head causal softmax attention.
    tokens = q.shape[0]
    q = q.view(tokens, heads, head_dim).transpose(0, 1)
    k = k.view(tokens, heads, head_dim).transpose(0, 1)
    v = v.view(tokens, heads, head_dim).transpose(0, 1)
    scores = q @ k.transpose(-2, -1) * (head_dim ** -0.5)
    mask = torch.full((tokens, tokens), float('-inf'),
                      device=scores.device, dtype=scores.dtype).triu(1)
    weights = torch.softmax(scores + mask, dim=-1)
    return (weights @ v).transpose(0, 1).reshape(tokens, heads * head_dim)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("TP_DEVICE", "cpu"),
                        choices=["cpu", "mps", "cuda", "rocm"])
    parser.add_argument("--transport", choices=["gloo", "tcp"], default="tcp")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS was requested but is unavailable")
    if args.device in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            parser.error("GPU was requested but is unavailable")
        if bool(torch.version.hip) != (args.device == "rocm"):
            parser.error("Requested GPU vendor does not match this PyTorch build")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    device = torch.device("cuda" if args.device == "rocm" else args.device)
    torch.set_num_threads(1)
    if args.transport == "tcp":
        from tcp_collectives import TCPCollectives
        collective = TCPCollectives()
    else:
        dist.init_process_group("gloo", timeout=timedelta(seconds=60))
        collective = dist
    try:
        rank, world = collective.get_rank(), collective.get_world_size()
        heads, head_dim, hidden = 8, 16, 256
        dim = heads * head_dim
        if world < 2 or heads % world or hidden % world:
            raise ValueError("Use >=2 ranks dividing both head and hidden counts")
        head_slice = heads // world
        h0, h1 = rank * head_slice, (rank + 1) * head_slice
        q0, q1 = h0 * head_dim, h1 * head_dim
        width = hidden // world
        m0, m1 = rank * width, (rank + 1) * width
        generator = torch.Generator().manual_seed(20260918)
        collectives = 0

        def shared(shape, scale=1.0):
            value = (torch.randn(shape, generator=generator) * scale
                     if rank == 0 else torch.empty(shape))
            collective.broadcast(value, src=0)
            return value

        with torch.inference_mode():
            n1 = shared((dim,), 0.0) + 1.0            # norm weights, replicated
            n2 = shared((dim,), 0.0) + 1.0
            wq = shared((dim, dim), dim ** -0.5)
            wk = shared((dim, dim), dim ** -0.5)
            wv = shared((dim, dim), dim ** -0.5)
            wo = shared((dim, dim), dim ** -0.5)
            bo = shared((dim,), 0.1)
            w1 = shared((dim, hidden), dim ** -0.5)
            b1 = shared((hidden,), 0.1)
            w2 = shared((hidden, dim), hidden ** -0.5)
            b2 = shared((dim,), 0.1)

            # Column-parallel: this rank owns heads [h0, h1) and channels [m0, m1).
            local_wq = wq[:, q0:q1].contiguous().to(device)
            local_wk = wk[:, q0:q1].contiguous().to(device)
            local_wv = wv[:, q0:q1].contiguous().to(device)
            local_wo = wo[q0:q1, :].contiguous().to(device)   # row-parallel
            local_w1 = w1[:, m0:m1].contiguous().to(device)
            local_b1 = b1[m0:m1].contiguous().to(device)
            local_w2 = w2[m0:m1, :].contiguous().to(device)   # row-parallel

            cases = []
            for tokens in (1, 17, 128):
                x = shared((tokens, dim))
                state = x.to(device)

                # ---- attention, replicated norm -> sharded heads -> all_reduce
                h = rms_norm(state, n1.to(device))
                context = attention(h @ local_wq, h @ local_wk, h @ local_wv,
                                    head_slice, head_dim)
                partial = (context @ local_wo).to("cpu").contiguous()
                collective.all_reduce(partial, op=dist.ReduceOp.SUM)
                collectives += 1
                state = state + (partial.to(device) + bo.to(device))

                # ---- MLP, replicated norm -> sharded hidden -> all_reduce
                h = rms_norm(state, n2.to(device))
                partial = (F.gelu(h @ local_w1 + local_b1)
                           @ local_w2).to("cpu").contiguous()
                collective.all_reduce(partial, op=dist.ReduceOp.SUM)
                collectives += 1
                result = (state + (partial.to(device) + b2.to(device))).to("cpu")

                # ---- unsharded reference, full weights, CPU
                ref = x
                h = rms_norm(ref, n1)
                ctx = attention(h @ wq, h @ wk, h @ wv, heads, head_dim)
                ref = ref + (ctx @ wo + bo)
                h = rms_norm(ref, n2)
                ref = ref + (F.gelu(h @ w1 + b1) @ w2 + b2)

                passed = bool(torch.allclose(result, ref, atol=2e-5, rtol=2e-4))
                cases.append({"tokens": tokens, "passed": passed,
                              "max_abs_error": (result - ref).abs().max().item(),
                              "collective_bytes": tokens * dim * 4})
        report = {"rank": rank, "device": str(device), "requested": args.device,
                  "torch": torch.__version__, "hip": torch.version.hip,
                  "cuda": torch.version.cuda, "host": platform.node(),
                  "heads": heads, "local_heads": head_slice, "dim": dim,
                  "hidden": hidden, "local_hidden": width,
                  "collectives_per_block": 2, "collectives_total": collectives,
                  "cases": cases}
        reports = [None] * world
        collective.all_gather_object(reports, report)
        success = all(c["passed"] for r in reports for c in r["cases"])
        if rank == 0:
            print(json.dumps({"passed": success, "transport": "CPU " + args.transport,
                              "dtype": "float32", "atol": 2e-5, "rtol": 2e-4,
                              "block": "prenorm decoder, causal, GELU MLP",
                              "ranks": reports}, indent=2), flush=True)
        if not success:
            raise SystemExit(1)
    finally:
        collective.destroy_process_group()


if __name__ == "__main__":
    main()
