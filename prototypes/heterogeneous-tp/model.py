"""FP32 tensor-parallel TWO-BLOCK autoregressive model, greedy generation.

Two independently weighted decoder blocks, a separate KV cache per layer, token
embeddings and an LM head. Generates >=8 tokens greedily and compares BOTH the
per-step logits and the generated token IDs against an unsharded reference.

SYNTHETIC WEIGHTS. The generated token IDs are arithmetic, not language; nothing
here is meaningful text generation or a trained-model result.

Independence rule, learned the hard way when a control passed: the reference
shares no norm helper, no mask helper and no cache machinery with the sharded
path. A bug in a shared helper cancels on both sides and the comparison passes.

Sharded:   attention heads, MLP hidden channels.
Replicated: embeddings, LM head, norms, residual stream. Sharding the LM head
would need an all_gather the transport does not implement; stated, not hidden.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
import platform

import torch
import torch.distributed as dist
import torch.nn.functional as F

LAYERS = 2


def rms_norm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _ref_norm(x, weight, eps=1e-5):
    """Deliberately different arithmetic from rms_norm; see the independence rule."""
    mean_square = (x * x).sum(-1, keepdim=True) / x.shape[-1]
    return weight * (x / torch.sqrt(mean_square + eps))


def causal_mask(q_len, k_len, offset, device, dtype):
    q_pos = torch.arange(q_len, device=device).unsqueeze(1) + offset
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)
    return torch.zeros(q_len, k_len, device=device, dtype=dtype).masked_fill(
        k_pos > q_pos, float('-inf'))


def heads_view(t, heads, head_dim):
    return t.view(t.shape[0], heads, head_dim).transpose(0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=os.environ.get("TP_DEVICE", "cpu"),
                        choices=["cpu", "mps", "cuda", "rocm"])
    parser.add_argument("--transport", choices=["gloo", "tcp"], default="tcp")
    parser.add_argument("--generate", type=int, default=8)
    args = parser.parse_args()
    if args.generate < 8:
        parser.error("--generate must be at least 8: fewer steps does not exercise "
                     "autoregressive cache growth")
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
        heads, head_dim, hidden, vocab = 8, 16, 256, 64
        dim = heads * head_dim
        if world < 2 or heads % world or hidden % world:
            raise ValueError("Use >=2 ranks dividing both head and hidden counts")
        local_heads, width = heads // world, hidden // world
        q0, q1 = rank * local_heads * head_dim, (rank + 1) * local_heads * head_dim
        m0, m1 = rank * width, (rank + 1) * width
        generator = torch.Generator().manual_seed(20260918)
        counters = {"collectives": 0}

        def shared(shape, scale=1.0):
            value = (torch.randn(shape, generator=generator) * scale
                     if rank == 0 else torch.empty(shape))
            collective.broadcast(value, src=0)
            return value

        with torch.inference_mode():
            def make_layer():
                return {'n1': shared((dim,), 0.1) + 1.0, 'n2': shared((dim,), 0.1) + 1.0,
                        'wq': shared((dim, dim), dim ** -0.5),
                        'wk': shared((dim, dim), dim ** -0.5),
                        'wv': shared((dim, dim), dim ** -0.5),
                        'wo': shared((dim, dim), dim ** -0.5), 'bo': shared((dim,), 0.1),
                        'w1': shared((dim, hidden), dim ** -0.5), 'b1': shared((hidden,), 0.1),
                        'w2': shared((hidden, dim), hidden ** -0.5), 'b2': shared((dim,), 0.1)}
            # Independently drawn per layer: layer 1 is not a copy of layer 0.
            layers = [make_layer() for _ in range(LAYERS)]
            emb = shared((vocab, dim))
            nf = shared((dim,), 0.1) + 1.0          # final norm before the LM head
            lm_head = shared((dim, vocab), dim ** -0.5)

            def localise(w):
                return {'wq': w['wq'][:, q0:q1].contiguous().to(device),
                        'wk': w['wk'][:, q0:q1].contiguous().to(device),
                        'wv': w['wv'][:, q0:q1].contiguous().to(device),
                        'wo': w['wo'][q0:q1, :].contiguous().to(device),
                        'w1': w['w1'][:, m0:m1].contiguous().to(device),
                        'b1': w['b1'][m0:m1].contiguous().to(device),
                        'w2': w['w2'][m0:m1, :].contiguous().to(device),
                        'n1': w['n1'].to(device), 'n2': w['n2'].to(device),
                        'bo': w['bo'].to(device), 'b2': w['b2'].to(device)}
            def tensor_sha(t):
                return hashlib.sha256(t.contiguous().numpy()
                                      .astype('<f4', copy=False).tobytes()).hexdigest()

            # Hash the ACTUAL post-transform tensors this run uses, per layer.
            # Computed here rather than replicated in a second script, because a
            # separate replica of the draw order can silently drift from the run.
            combined = hashlib.sha256()
            weight_sha = {}
            for i, lw in enumerate(layers):
                weight_sha[f"layer{i}"] = {}
                for name in sorted(lw):
                    full = tensor_sha(lw[name])
                    combined.update(bytes.fromhex(full))
                    weight_sha[f"layer{i}"][name] = full[:16]
            for name, tensor in (("emb", emb), ("final_norm", nf), ("lm_head", lm_head)):
                full = tensor_sha(tensor)
                combined.update(bytes.fromhex(full))
                weight_sha[name] = full[:16]
            weight_sha["combined"] = combined.hexdigest()

            local = [localise(w) for w in layers]
            emb_d, nf_d, lm_d = emb.to(device), nf.to(device), lm_head.to(device)

            def fresh_caches():
                return [{'k': None, 'v': None} for _ in range(LAYERS)]

            def layer_forward(state, w, cache, offset):
                h = rms_norm(state, w['n1'])
                q = heads_view(h @ w['wq'], local_heads, head_dim)
                k_new = heads_view(h @ w['wk'], local_heads, head_dim)
                v_new = heads_view(h @ w['wv'], local_heads, head_dim)
                cache['k'] = k_new if cache['k'] is None else torch.cat([cache['k'], k_new], 1)
                cache['v'] = v_new if cache['v'] is None else torch.cat([cache['v'], v_new], 1)
                expected = offset + state.shape[0]
                assert cache['k'].shape == (local_heads, expected, head_dim), \
                    f"K cache {tuple(cache['k'].shape)} != {(local_heads, expected, head_dim)}"
                assert cache['v'].shape == cache['k'].shape, "V cache shape mismatch"
                scores = q @ cache['k'].transpose(-2, -1) * (head_dim ** -0.5)
                scores = scores + causal_mask(state.shape[0], cache['k'].shape[1],
                                              offset, scores.device, scores.dtype)
                ctx = (torch.softmax(scores, -1) @ cache['v']).transpose(0, 1).reshape(
                    state.shape[0], local_heads * head_dim)
                partial = (ctx @ w['wo']).to("cpu").contiguous()
                collective.all_reduce(partial, op=dist.ReduceOp.SUM)
                counters['collectives'] += 1
                state = state + (partial.to(device) + w['bo'])
                h = rms_norm(state, w['n2'])
                partial = (F.gelu(h @ w['w1'] + w['b1']) @ w['w2']).to("cpu").contiguous()
                collective.all_reduce(partial, op=dist.ReduceOp.SUM)
                counters['collectives'] += 1
                return state + (partial.to(device) + w['b2'])

            def model_forward(ids, caches, offset):
                state = emb_d[torch.tensor(ids, device=device)]
                for w, cache in zip(local, caches):
                    state = layer_forward(state, w, cache, offset)
                return (rms_norm(state, nf_d) @ lm_d).to("cpu")

            def sharded_generate(prompt, n_gen, chunks):
                caches = fresh_caches()
                offset, logits = 0, None
                sizes = ([len(prompt)] if chunks == 1 else
                         [len(prompt) // 2, len(prompt) - len(prompt) // 2])
                for size in sizes:
                    if size == 0:
                        continue
                    logits = model_forward(prompt[offset:offset + size], caches, offset)
                    offset += size
                out_ids, step_logits = [], []
                for i in range(n_gen):
                    last = logits[-1]
                    step_logits.append(last)
                    nxt = int(torch.argmax(last))
                    out_ids.append(nxt)
                    # Do NOT forward the final token: its logits are never consumed,
                    # and the wasted pass would inflate the collective count.
                    if i < n_gen - 1:
                        logits = model_forward([nxt], caches, offset)
                        offset += 1
                return out_ids, torch.stack(step_logits), caches

            # ---- unsharded reference: full weights, no cache, recomputed each step
            def reference_forward(ids):
                state = emb[torch.tensor(ids)]
                n = len(ids)
                for w in layers:
                    h = _ref_norm(state, w['n1'])
                    q = heads_view(h @ w['wq'], heads, head_dim)
                    k = heads_view(h @ w['wk'], heads, head_dim)
                    v = heads_view(h @ w['wv'], heads, head_dim)
                    scores = q @ k.transpose(-2, -1) * (head_dim ** -0.5)
                    scores = scores + torch.full((n, n), float('-inf'),
                                                 dtype=scores.dtype).triu(1)
                    ctx = (torch.softmax(scores, -1) @ v).transpose(0, 1).reshape(n, -1)
                    state = state + (ctx @ w['wo'] + w['bo'])
                    h = _ref_norm(state, w['n2'])
                    state = state + (F.gelu(h @ w['w1'] + w['b1']) @ w['w2'] + w['b2'])
                return _ref_norm(state, nf) @ lm_head

            def reference_generate(prompt, n_gen):
                ids = list(prompt)
                out_ids, step_logits = [], []
                for _ in range(n_gen):
                    last = reference_forward(ids)[-1]
                    step_logits.append(last)
                    nxt = int(torch.argmax(last))
                    out_ids.append(nxt)
                    ids.append(nxt)
                return out_ids, torch.stack(step_logits)

            cases = []
            for prompt_len in (1, 5, 17):
                for chunks in (1, 2):
                    if chunks == 2 and prompt_len < 2:
                        continue
                    prompt = [(7 * i + 3) % vocab for i in range(prompt_len)]
                    ref_ids, ref_logits = reference_generate(prompt, args.generate)
                    got_ids, got_logits, caches = sharded_generate(prompt, args.generate, chunks)
                    # Cache reset: a second run with fresh caches must be identical.
                    again_ids, again_logits, _ = sharded_generate(prompt, args.generate, chunks)
                    err = (got_logits - ref_logits).abs().max().item()
                    # Top-1 vs top-2 margin: a near-tie means matching IDs was luck.
                    top2 = ref_logits.topk(2, dim=-1).values
                    margin = (top2[:, 0] - top2[:, 1]).min().item()
                    ids_match = got_ids == ref_ids
                    reset_ok = (again_ids == got_ids and
                                torch.equal(again_logits, got_logits))
                    final_len = prompt_len + args.generate - 1
                    cache_ok = all(c['k'].shape[1] == final_len for c in caches)
                    cases.append({
                        "prompt_len": prompt_len, "chunks": chunks,
                        "generated": args.generate, "passed": bool(
                            ids_match and reset_ok and cache_ok and err < 2e-4),
                        "token_ids_match_reference": ids_match,
                        "cache_reset_reproducible": reset_ok,
                        "final_cache_len_ok": cache_ok,
                        "max_abs_logit_error": err,
                        "min_top1_top2_margin": margin,
                        "reference_ids": ref_ids, "sharded_ids": got_ids})
        report = {"rank": rank, "device": str(device), "requested": args.device,
                  "torch": torch.__version__, "hip": torch.version.hip,
                  "cuda": torch.version.cuda, "host": platform.node(),
                  "layers": LAYERS, "heads": heads, "local_heads": local_heads,
                  "dim": dim, "hidden": hidden, "local_hidden": width, "vocab": vocab,
                  "weight_sha256_post_transform": weight_sha,
                  "collectives_per_token": 2 * LAYERS,
                  "collectives_total": counters['collectives'], "cases": cases}
        reports = [None] * world
        collective.all_gather_object(reports, report)
        success = all(c["passed"] for r in reports for c in r["cases"])
        if rank == 0:
            print(json.dumps({
                "passed": success, "transport": "CPU " + args.transport,
                "dtype": "float32", "logit_atol": 2e-4,
                "model": f"{LAYERS}-block prenorm decoder, embeddings + LM head, greedy",
                "WARNING": ("SYNTHETIC WEIGHTS. Generated token IDs are arithmetic, "
                            "not language. This is not text generation and not a "
                            "trained-model result. No speed claim is made."),
                "ranks": reports}, indent=2), flush=True)
        if not success:
            raise SystemExit(1)
    finally:
        collective.destroy_process_group()


if __name__ == "__main__":
    main()
