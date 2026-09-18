"""Batched GPT-2 solo/two-rank benchmark with per-sequence correctness gates.

Uniform-length independent prompts; finished sequences remain inert EOS lanes
until the active sequences finish. This is static batching, not communication
and compute overlap or continuous batching.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
import torch.distributed as dist
import torch.nn.functional as F

MODEL_DIR = os.environ.get("GPT2_DIR", "/tmp/het-tp/gpt2")
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
EOS = 50256
# Distinct prompts, all the same token length after truncation, so the batch is
# uniform without padding. Different content on purpose: identical prompts would
# let a broken implementation look correct by symmetry.
SEEDS = [
    " The history of computing is a long story about people who wanted machines",
    " In the beginning the ocean was empty and the sky above it was very dark",
    " She opened the door slowly because the hallway light had been broken since",
    " Every program that has ever been written contains at least one mistake that",
    " The train arrived late and the passengers waiting on the platform had begun",
    " Nobody in the village could remember when the old bridge had been built or",
    " He wrote the number down carefully and then checked it against the list of",
    " Scientists studying the region reported that the ice had thinned more than",
    " The recipe called for three eggs but the kitchen had only two and a half a",
    " When the music stopped everyone in the room turned to look at the doorway",
    " A small grey cat had been sitting on the wall outside the shop all morning",
    " The manual said that the device should never be operated while it was con",
    " After the storm passed the farmers went out to see what had been left of a",
    " Three of the four engines failed during the test and the fourth was shut d",
    " Language models predict the next token given everything that came before it",
    " The library closed at six but the reading room stayed open for another two",
]


def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def causal_mask(q_len, k_len, offset, device, dtype):
    q_pos = torch.arange(q_len, device=device).unsqueeze(1) + offset
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)
    return torch.zeros(q_len, k_len, device=device, dtype=dtype).masked_fill(
        k_pos > q_pos, float('-inf'))


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.environ.get("TP_DEVICE", "cpu"),
                    choices=["cpu", "mps", "cuda", "rocm"])
    ap.add_argument("--transport", choices=["gloo", "tcp"], default="tcp")
    ap.add_argument("--mode", choices=["solo", "tp"], default="tp")
    ap.add_argument("--batch", type=int, default=1)
    # 12 is the shortest seed; every prompt is truncated to exactly this so the
    # batch is uniform without padding. The assertion below is deliberate: a
    # silently short prompt would make the batch ragged and the timing wrong.
    ap.add_argument("--prompt-tokens", type=int, default=12)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--check", action="store_true", help="compatibility flag; correctness is always checked")
    args = ap.parse_args()
    if args.batch <= 0 or args.batch > len(SEEDS):
        ap.error(f"--batch must be 1..{len(SEEDS)}")
    if args.new_tokens <= 0 or args.runs <= 0 or args.warmups < 0:
        ap.error("bad run counts")
    if args.device == "mps" and not torch.backends.mps.is_available():
        ap.error("MPS unavailable")
    if args.device in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            ap.error("GPU unavailable")
        if bool(torch.version.hip) != (args.device == "rocm"):
            ap.error("GPU vendor does not match this torch build")
    device = torch.device("cuda" if args.device == "rocm" else args.device)
    torch.set_num_threads(1)

    from safetensors.torch import load_file
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    n_layer, n_head, dim = cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
    vocab, head_dim, eps = cfg["vocab_size"], dim // cfg["n_head"], cfg["layer_norm_epsilon"]
    if args.prompt_tokens <= 0 or args.prompt_tokens + args.new_tokens > cfg["n_positions"]:
        ap.error("prompt plus generation must fit the context")
    weight_path = os.path.join(MODEL_DIR, "model.safetensors")
    with open(weight_path, "rb") as f:
        weight_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    if weight_sha256 != "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707":
        ap.error("this comparison requires the pinned openai-community/gpt2 checkpoint")
    sd = load_file(weight_path)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompts = []
    for s in SEEDS[:args.batch]:
        ids = tok(s * (args.prompt_tokens + 1))["input_ids"][:args.prompt_tokens]
        assert len(ids) == args.prompt_tokens, f"seed too short: {s!r}"
        prompts.append(ids)

    if args.mode == "solo":
        from bench import NullCollective
        collective = NullCollective()
    elif args.transport == "tcp":
        from tcp_collectives import TCPCollectives
        collective = TCPCollectives()
    else:
        dist.init_process_group("gloo")
        collective = dist
    try:
        rank, world = collective.get_rank(), collective.get_world_size()
        if n_head % world or (4 * dim) % world:
            raise ValueError("ranks must divide head count and MLP hidden size")
        hpr, cpr = n_head // world, (n_head // world) * head_dim
        mpr = (4 * dim) // world
        a0, m0 = rank * cpr, rank * mpr
        B = args.batch

        with torch.inference_mode():
            blocks = []
            for i in range(n_layer):
                p = f"h.{i}."
                idx = torch.cat([torch.arange(j * dim + a0, j * dim + a0 + cpr)
                                 for j in range(3)])
                blocks.append({
                    "ln1_w": sd[p + "ln_1.weight"].to(device), "ln1_b": sd[p + "ln_1.bias"].to(device),
                    "ln2_w": sd[p + "ln_2.weight"].to(device), "ln2_b": sd[p + "ln_2.bias"].to(device),
                    "ca_w": sd[p + "attn.c_attn.weight"][:, idx].contiguous().to(device),
                    "ca_b": sd[p + "attn.c_attn.bias"][idx].contiguous().to(device),
                    "cp_w": sd[p + "attn.c_proj.weight"][a0:a0 + cpr, :].contiguous().to(device),
                    "cp_b": sd[p + "attn.c_proj.bias"].to(device),
                    "fc_w": sd[p + "mlp.c_fc.weight"][:, m0:m0 + mpr].contiguous().to(device),
                    "fc_b": sd[p + "mlp.c_fc.bias"][m0:m0 + mpr].contiguous().to(device),
                    "mp_w": sd[p + "mlp.c_proj.weight"][m0:m0 + mpr, :].contiguous().to(device),
                    "mp_b": sd[p + "mlp.c_proj.bias"].to(device)})
            wte_d, wpe_d = sd["wte.weight"].to(device), sd["wpe.weight"].to(device)
            lnf_w, lnf_b = sd["ln_f.weight"].to(device), sd["ln_f.bias"].to(device)
            counters = {"collectives": 0, "payload_bytes": 0}

            def reduce_part(part):
                if world == 1:
                    return part
                part = part.to("cpu").contiguous()
                collective.all_reduce(part, op=dist.ReduceOp.SUM)
                counters["collectives"] += 1
                counters["payload_bytes"] += part.numel() * 4
                return part.to(device)

            def block_forward(x, b, cache, offset):
                # x: (B, T, dim)
                Bx, T = x.shape[0], x.shape[1]
                h = F.layer_norm(x, (dim,), b["ln1_w"], b["ln1_b"], eps)
                q, k, v = (h @ b["ca_w"] + b["ca_b"]).split(cpr, dim=-1)
                shape = lambda t: t.view(Bx, T, hpr, head_dim).transpose(1, 2)
                q, k, v = shape(q), shape(k), shape(v)          # (B, hpr, T, hd)
                cache["k"] = k if cache["k"] is None else torch.cat([cache["k"], k], 2)
                cache["v"] = v if cache["v"] is None else torch.cat([cache["v"], v], 2)
                want = (Bx, hpr, offset + T, head_dim)
                assert cache["k"].shape == want, f"K cache {tuple(cache['k'].shape)} != {want}"
                assert cache["v"].shape == want, f"V cache {tuple(cache['v'].shape)} != {want}"
                scores = q @ cache["k"].transpose(-2, -1) / math.sqrt(head_dim)
                scores = scores + causal_mask(T, cache["k"].shape[2], offset,
                                              scores.device, scores.dtype)
                ctx = (torch.softmax(scores, -1) @ cache["v"]).transpose(1, 2).reshape(Bx, T, cpr)
                part = reduce_part(ctx @ b["cp_w"])
                x = x + (part + b["cp_b"])
                h = F.layer_norm(x, (dim,), b["ln2_w"], b["ln2_b"], eps)
                part = reduce_part(gelu_new(h @ b["fc_w"] + b["fc_b"]) @ b["mp_w"])
                return x + (part + b["mp_b"])

            def model_forward(ids, caches, offset):
                # ids: (B, T) python lists
                t = torch.tensor(ids, device=device)
                pos = torch.arange(offset, offset + t.shape[1], device=device)
                x = wte_d[t] + wpe_d[pos]
                for b, c in zip(blocks, caches):
                    x = block_forward(x, b, c, offset)
                if rank != 0:
                    return None
                x = F.layer_norm(x[:, -1:, :], (dim,), lnf_w, lnf_b, eps)
                return x @ wte_d.T                  # (B, T, vocab)

            def generate(n_new, stamps=None):
                caches = [{"k": None, "v": None} for _ in range(n_layer)]
                logits = model_forward(prompts, caches, 0)
                offset = args.prompt_tokens
                out = [[] for _ in range(B)]
                finished = [False] * B
                for i in range(n_new):
                    choice = (torch.argmax(logits[:, -1, :], dim=-1).to("cpu", dtype=torch.float32)
                              if rank == 0 else torch.zeros(B, dtype=torch.float32))
                    collective.broadcast(choice, src=0)
                    nxt = choice.to(torch.int64).tolist()
                    sync(device)
                    stamp = time.perf_counter()
                    for bi, token in enumerate(nxt):
                        if finished[bi]:
                            nxt[bi] = EOS
                            continue
                        out[bi].append(int(token))
                        if stamps is not None:
                            stamps[bi].append(stamp)
                        finished[bi] = token == EOS
                    if all(finished):
                        break
                    if i < n_new - 1:
                        logits = model_forward([[t] for t in nxt], caches, offset)
                        offset += 1
                return out

            def agree(flag):
                value = torch.tensor([float(flag)])
                collective.all_reduce(value, op=dist.ReduceOp.SUM)
                return int(value.item()) == world

            def barrier():
                collective.all_reduce(torch.zeros(1), op=dist.ReduceOp.SUM)

            ref_model = GPT2LMHeadModel.from_pretrained(
                MODEL_DIR, dtype=torch.float32, attn_implementation="eager").eval()
            refs = []
            for pid in prompts:
                ids, seq, past = list(pid), [], None
                for _ in range(args.new_tokens):
                    result = ref_model(input_ids=torch.tensor([ids]), past_key_values=past, use_cache=True)
                    past = result.past_key_values
                    nxt = int(torch.argmax(result.logits[0, -1]))
                    seq.append(nxt)
                    if nxt == EOS:
                        break
                    ids = [nxt]
                refs.append(seq)
            del ref_model
            got = generate(args.new_tokens)
            correct = agree(got == refs)
            runs, invalid_runs = [], []
            if correct:
                for _ in range(args.warmups):
                    if not agree(generate(args.new_tokens) == refs):
                        correct = False
                        break
            if correct:
                for run_index in range(args.runs):
                    stamps = [[] for _ in range(B)]
                    barrier()
                    sync(device)
                    counters.update(collectives=0, payload_bytes=0)
                    t0 = time.perf_counter()
                    produced = generate(args.new_tokens, stamps)
                    sync(device)
                    t1 = time.perf_counter()
                    valid = agree(produced == refs)
                    if not valid:
                        invalid_runs.append({"run_index": run_index, "produced_ids": produced})
                        continue
                    per_sequence = []
                    for ids, ts in zip(produced, stamps):
                        elapsed = ts[-1] - ts[0]
                        per_sequence.append({"generated_tokens": len(ids), "ttft_s": ts[0]-t0,
                            "latency_s": ts[-1]-t0, "decode_tokens": max(0,len(ids)-1),
                            "decode_s": elapsed, "decode_tokens_per_s": (len(ids)-1)/elapsed if elapsed > 0 else None})
                    count = sum(map(len, produced))
                    decode_count = sum(max(0,len(ids)-1) for ids in produced)
                    first = min(ts[0] for ts in stamps)
                    last = max(ts[-1] for ts in stamps)
                    runs.append({"run_index": run_index, "produced_ids": produced,
                        "total_wall_s": t1-t0, "generated_tokens_total": count,
                        "end_to_end_tokens_per_s": count/(t1-t0),
                        "decode_tokens_total": decode_count, "decode_elapsed_s": last-first,
                        "aggregate_decode_tokens_per_s": decode_count/(last-first) if last > first else None,
                        "per_sequence": per_sequence, "reduction_counters": dict(counters)})
            globally_valid = correct and not invalid_runs and len(runs) == args.runs
            if not globally_valid:
                runs = []

        report = {"rank": rank, "world": world, "host": platform.node(),
                  "mode": args.mode, "device": str(device), "torch": torch.__version__,
                  "hip": torch.version.hip, "model": "openai-community/gpt2", "revision": REVISION,
                  "weight_sha256": weight_sha256,
                  "source_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                  "batch": B, "prompt_tokens": args.prompt_tokens, "prompt_ids": prompts,
                  "new_tokens": args.new_tokens, "layers": n_layer, "warmups": args.warmups,
                  "threads": torch.get_num_threads(), "generation_head": "last-root",
                  "reference_ids": refs, "initial_ids": got,
                  "correctness_checked": True, "valid": globally_valid,
                  "all_sequences_match_reference": correct, "runs": runs, "invalid_runs": invalid_runs}
        out = [None] * world
        collective.all_gather_object(out, report)
        if rank == 0:
            print(json.dumps({"ranks": out}, indent=2), flush=True)
        if not globally_valid:
            raise SystemExit(1)
    finally:
        collective.destroy_process_group()


if __name__ == "__main__":
    main()
