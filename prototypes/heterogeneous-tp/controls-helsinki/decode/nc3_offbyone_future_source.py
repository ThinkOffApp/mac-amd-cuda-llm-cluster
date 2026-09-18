"""FP32 tensor-parallel CACHED DECODE, correctness against an unsharded reference.

Extends block.py with per-rank head-local K/V cache:
  prefill a prefix (optionally in chunks, exercising a nonzero query offset),
  then decode successive single tokens, comparing each last-token output to a
  freshly recomputed full-prefix unsharded reference.

The reference deliberately shares NO cache or offset machinery with the cached
path: it recomputes the whole prefix with a plain triu(1) mask. An off-by-one in
the offset mask must therefore show up as a mismatch, not cancel out on both
sides.

Toy decoder block with synthetic weights. This is not trained-model generation.
Full CPU weights are retained for the reference; GPU math is local; every
collective receives CPU tensors explicitly.
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


def causal_mask(q_len, k_len, offset, device, dtype):
    """Query i sits at absolute position offset+i and may see keys 0..offset+i."""
    q_pos = torch.arange(q_len, device=device).unsqueeze(1) + offset + 1
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)
    mask = torch.zeros(q_len, k_len, device=device, dtype=dtype)
    return mask.masked_fill(k_pos > q_pos, float('-inf'))


def heads_view(t, heads, head_dim):
    # (tokens, heads*head_dim) -> (heads, tokens, head_dim)
    return t.view(t.shape[0], heads, head_dim).transpose(0, 1)


def attend(q, k, v, offset):
    """q/k/v: (heads, *, head_dim). k/v carry the whole prefix, q only the new tokens."""
    head_dim = q.shape[-1]
    scores = q @ k.transpose(-2, -1) * (head_dim ** -0.5)
    scores = scores + causal_mask(q.shape[1], k.shape[1], offset,
                                  scores.device, scores.dtype)
    ctx = torch.softmax(scores, dim=-1) @ v
    return ctx.transpose(0, 1).reshape(q.shape[1], q.shape[0] * head_dim)


def _ref_norm(x, weight, eps=1e-5):
    """Deliberately NOT rms_norm(): written out with different arithmetic so that a
    bug inside rms_norm cannot cancel on both sides of the comparison. Calling the
    shared helper here made the norm untestable — control nc5 passed when the weight
    multiply was deleted from both paths at once."""
    mean_square = (x * x).sum(-1, keepdim=True) / x.shape[-1]
    return weight * (x / torch.sqrt(mean_square + eps))


def reference_block(x, w, heads, head_dim):
    """Unsharded, uncached, full-prefix. Plain triu(1) causal mask."""
    h = _ref_norm(x, w['n1'])
    q = heads_view(h @ w['wq'], heads, head_dim)
    k = heads_view(h @ w['wk'], heads, head_dim)
    v = heads_view(h @ w['wv'], heads, head_dim)
    scores = q @ k.transpose(-2, -1) * (head_dim ** -0.5)
    scores = scores + torch.full((x.shape[0], x.shape[0]), float('-inf'),
                                 dtype=scores.dtype).triu(1)
    ctx = (torch.softmax(scores, dim=-1) @ v).transpose(0, 1).reshape(x.shape[0], -1)
    x = x + (ctx @ w['wo'] + w['bo'])
    h = _ref_norm(x, w['n2'])
    return x + (F.gelu(h @ w['w1'] + w['b1']) @ w['w2'] + w['b2'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("TP_DEVICE", "cpu"),
                        choices=["cpu", "mps", "cuda", "rocm"])
    parser.add_argument("--transport", choices=["gloo", "tcp"], default="tcp")
    parser.add_argument("--steps", type=int, default=4, help="decode steps per case")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive: cached decode cannot be validated "
                     "by running zero decode steps")
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
        local_heads = heads // world
        q0, q1 = rank * local_heads * head_dim, (rank + 1) * local_heads * head_dim
        width = hidden // world
        m0, m1 = rank * width, (rank + 1) * width
        generator = torch.Generator().manual_seed(20260918)
        counters = {"collectives": 0}

        def shared(shape, scale=1.0):
            value = (torch.randn(shape, generator=generator) * scale
                     if rank == 0 else torch.empty(shape))
            collective.broadcast(value, src=0)
            return value

        with torch.inference_mode():
            w = {'n1': shared((dim,), 0.1) + 1.0, 'n2': shared((dim,), 0.1) + 1.0,
                 'wq': shared((dim, dim), dim ** -0.5),
                 'wk': shared((dim, dim), dim ** -0.5),
                 'wv': shared((dim, dim), dim ** -0.5),
                 'wo': shared((dim, dim), dim ** -0.5), 'bo': shared((dim,), 0.1),
                 'w1': shared((dim, hidden), dim ** -0.5), 'b1': shared((hidden,), 0.1),
                 'w2': shared((hidden, dim), hidden ** -0.5), 'b2': shared((dim,), 0.1)}
            local = {'wq': w['wq'][:, q0:q1].contiguous().to(device),
                     'wk': w['wk'][:, q0:q1].contiguous().to(device),
                     'wv': w['wv'][:, q0:q1].contiguous().to(device),
                     'wo': w['wo'][q0:q1, :].contiguous().to(device),
                     'w1': w['w1'][:, m0:m1].contiguous().to(device),
                     'b1': w['b1'][m0:m1].contiguous().to(device),
                     'w2': w['w2'][m0:m1, :].contiguous().to(device),
                     'n1': w['n1'].to(device), 'n2': w['n2'].to(device),
                     'bo': w['bo'].to(device), 'b2': w['b2'].to(device)}

            def new_cache():
                return {'k': None, 'v': None}

            def cached_forward(x_new, cache, offset):
                """x_new: (q_len, dim) on device. Returns (q_len, dim) on CPU."""
                state = x_new
                h = rms_norm(state, local['n1'])
                q = heads_view(h @ local['wq'], local_heads, head_dim)
                k_new = heads_view(h @ local['wk'], local_heads, head_dim)
                v_new = heads_view(h @ local['wv'], local_heads, head_dim)
                cache['k'] = k_new if cache['k'] is None else torch.cat([cache['k'], k_new], 1)
                cache['v'] = v_new if cache['v'] is None else torch.cat([cache['v'], v_new], 1)
                expected = offset + x_new.shape[0]
                assert cache['k'].shape == (local_heads, expected, head_dim), \
                    f"K cache {tuple(cache['k'].shape)} != {(local_heads, expected, head_dim)}"
                assert cache['v'].shape == cache['k'].shape, "V cache shape != K cache shape"
                ctx = attend(q, cache['k'], cache['v'], offset)
                partial = (ctx @ local['wo']).to("cpu").contiguous()
                collective.all_reduce(partial, op=dist.ReduceOp.SUM)
                counters['collectives'] += 1
                state = state + (partial.to(device) + local['bo'])
                h = rms_norm(state, local['n2'])
                partial = (F.gelu(h @ local['w1'] + local['b1'])
                           @ local['w2']).to("cpu").contiguous()
                collective.all_reduce(partial, op=dist.ReduceOp.SUM)
                counters['collectives'] += 1
                return (state + (partial.to(device) + local['b2'])).to("cpu")

            def run_prefill(x, cache, chunks):
                """Feed x in `chunks` pieces; every piece after the first has offset>0."""
                out, offset = [], 0
                sizes = ([x.shape[0]] if chunks == 1 else
                         [x.shape[0] // 2, x.shape[0] - x.shape[0] // 2])
                for size in sizes:
                    if size == 0:
                        continue
                    piece = x[offset:offset + size].to(device)
                    out.append(cached_forward(piece, cache, offset))
                    offset += size
                return torch.cat(out, 0), offset

            cases = []
            for prefix_len in (1, 17, 128):
                for chunks in (1, 2):
                    if chunks == 2 and prefix_len < 2:
                        continue
                    steps = args.steps
                    x_all = shared((prefix_len + steps, dim))
                    cache = new_cache()          # independent cache per case
                    pre_out, offset = run_prefill(x_all[:prefix_len], cache, chunks)
                    assert offset == prefix_len, f"prefill offset {offset} != {prefix_len}"
                    ref_pre = reference_block(x_all[:prefix_len], w, heads, head_dim)
                    errs = [(pre_out - ref_pre).abs().max().item()]
                    ok = bool(torch.allclose(pre_out, ref_pre, atol=2e-5, rtol=2e-4))

                    for step in range(steps):
                        pos = prefix_len + step
                        out = cached_forward(x_all[pos:pos + 1].to(device), cache, pos)
                        # Fresh full-prefix reference, recomputed from scratch each step.
                        ref = reference_block(x_all[:pos + 1], w, heads, head_dim)[-1:]
                        errs.append((out - ref).abs().max().item())
                        ok = ok and bool(torch.allclose(out, ref, atol=2e-5, rtol=2e-4))
                    assert cache['k'].shape[1] == prefix_len + steps, "final cache length"

                    # No future-token visibility: perturbing the LAST prefix token
                    # must not change any earlier prefill output.
                    causal_ok = None   # None = not applicable, never a silent pass
                    if prefix_len > 1:
                        x_mod = x_all[:prefix_len].clone()
                        x_mod[-1] = x_mod[-1] + 7.0
                        mod_out, _ = run_prefill(x_mod, new_cache(), chunks)
                        causal_ok = bool(torch.equal(mod_out[:-1], pre_out[:-1]))
                        ok = ok and causal_ok

                    cases.append({"prefix_len": prefix_len, "chunks": chunks,
                                  "steps": steps, "passed": ok,
                                  "no_future_visibility": (causal_ok if causal_ok
                                      is not None else "n/a: needs prefix_len>1"),
                                  "max_abs_error": max(errs),
                                  "payload_bytes_per_collective": dim * 4,
                                  "aggregate_link_payload_bytes_per_decoded_token":
                                      dim * 4 * 2 * 2})
        report = {"rank": rank, "device": str(device), "requested": args.device,
                  "torch": torch.__version__, "hip": torch.version.hip,
                  "cuda": torch.version.cuda, "host": platform.node(),
                  "heads": heads, "local_heads": local_heads, "dim": dim,
                  "hidden": hidden, "local_hidden": width,
                  "collectives_per_block": 2,
                  "collectives_total": counters['collectives'], "cases": cases}
        reports = [None] * world
        collective.all_gather_object(reports, report)
        success = all(c["passed"] for r in reports for c in r["cases"])
        if rank == 0:
            print(json.dumps({"passed": success, "transport": "CPU " + args.transport,
                              "dtype": "float32", "atol": 2e-5, "rtol": 2e-4,
                              "block": "prenorm decoder, causal, GELU MLP, KV cache",
                              "note": ("payload_bytes_per_collective is ONE payload; "
                                       "the two-rank exchange sends each both ways, so "
                                       "aggregate link payload per decoded token is "
                                       "4x that, before framing and setup. No speed "
                                       "claim follows from these byte counts."),
                              "ranks": reports}, indent=2), flush=True)
        if not success:
            raise SystemExit(1)
    finally:
        collective.destroy_process_group()


if __name__ == "__main__":
    main()
