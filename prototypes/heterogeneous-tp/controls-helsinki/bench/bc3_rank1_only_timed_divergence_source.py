"""Matched GPT-2 timing: Mini-solo vs M5-solo vs Mini-M5 tensor parallel.

Same revision, tokenizer, FP32, eval, eager-equivalent math, one torch thread,
same KV cache and same greedy/EOS policy in every configuration. The only
difference between solo and TP is whether the weights are sharded and whether
an all_reduce happens; solo runs the FULL model on its own GPU, never a half
shard and never on CPU.

Everything that is not generation - load, tokenize, hash, correctness check -
happens before the timed region. The correctness gate runs FIRST: if the output
IDs do not match the reference, the timings are not reported as valid.
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
# A fixed passage, sliced to an exact token count. Same text for every config.
SEED_TEXT = (" The history of computing is a history of people trying to make "
             "machines do arithmetic faster than they could do it themselves. "
             "Every generation believed it had reached the limit, and every "
             "generation was wrong about where the limit was. ") * 64


class NullCollective:
    """Solo: no peers. all_reduce over one rank is the identity."""
    def __init__(self):
        self.rank, self.world_size = 0, 1

    def all_reduce(self, tensor, op=None):
        return tensor

    def broadcast(self, tensor, src=0):
        return tensor

    def get_rank(self):
        return 0

    def get_world_size(self):
        return 1

    def all_gather_object(self, out, obj):
        out[0] = obj

    def destroy_process_group(self):
        pass


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
    ap.add_argument("--mode", choices=["solo", "tp"], required=True)
    ap.add_argument("--device", default=os.environ.get("TP_DEVICE", "cpu"),
                    choices=["cpu", "mps", "cuda", "rocm"])
    ap.add_argument("--prompt-tokens", type=int, required=True)
    ap.add_argument("--new-tokens", type=int, default=64)
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stage", choices=["auto", "cpu", "gpu"], default="auto",
                    help="auto: CPU staging for tp, GPU-resident for solo. "
                         "'cpu' with --mode solo is the STAGED CONTROL, not a "
                         "single-host baseline.")
    args = ap.parse_args()
    # Validate everything answerable locally BEFORE opening a socket, so a bad
    # invocation cannot leave a peer waiting on a connection that never comes.
    if args.prompt_tokens <= 0:
        ap.error("--prompt-tokens must be positive")
    if args.new_tokens <= 0:
        ap.error("--new-tokens must be positive")
    if args.runs <= 0:
        ap.error("--runs must be positive: zero runs is a vacuous success")
    if args.warmups < 0:
        ap.error("--warmups must not be negative")
    if args.prompt_tokens + args.new_tokens > 1024:
        ap.error("GPT-2 has 1024 positions: prompt + new must not exceed it")
    if args.device == "mps" and not torch.backends.mps.is_available():
        ap.error("MPS requested but unavailable")
    if args.device in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            ap.error("GPU requested but unavailable")
        if bool(torch.version.hip) != (args.device == "rocm"):
            ap.error("requested GPU vendor does not match this PyTorch build")
    device = torch.device("cuda" if args.device == "rocm" else args.device)
    torch.set_num_threads(1)

    # ---------- setup: outside every timed region ----------
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()
    import transformers as _tf, tokenizers as _tk
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    n_layer, n_head, dim = cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
    head_dim, eps = dim // n_head, cfg["layer_norm_epsilon"]
    sd = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    prompt_ids = tok(SEED_TEXT)["input_ids"][:args.prompt_tokens]
    assert len(prompt_ids) == args.prompt_tokens, "seed text too short"

    stage_cpu = (args.mode == "tp") if args.stage == "auto" else (args.stage == "cpu")
    config_label = args.mode if not (args.mode == "solo" and stage_cpu) else "solo-staged"
    if args.mode == "tp":
        if not stage_cpu:
            ap.error("tp requires CPU staging: the transport moves CPU tensors")
        from tcp_collectives import TCPCollectives
        collective = TCPCollectives()
    else:
        collective = NullCollective()
    try:
        rank, world = collective.get_rank(), collective.get_world_size()
        if n_head % world or (4 * dim) % world:
            raise ValueError("ranks must divide head count and MLP hidden size")
        hpr = n_head // world
        cpr = hpr * head_dim
        mpr = (4 * dim) // world
        a0, m0 = rank * cpr, rank * mpr
        with torch.inference_mode():
            blocks = []
            for i in range(n_layer):
                p = f"h.{i}."
                idx = torch.cat([torch.arange(j * dim + a0, j * dim + a0 + cpr)
                                 for j in range(3)])
                blocks.append({
                    "ln1_w": sd[p + "ln_1.weight"].to(device),
                    "ln1_b": sd[p + "ln_1.bias"].to(device),
                    "ln2_w": sd[p + "ln_2.weight"].to(device),
                    "ln2_b": sd[p + "ln_2.bias"].to(device),
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

            def block_forward(x, b, cache, offset):
                h = F.layer_norm(x, (dim,), b["ln1_w"], b["ln1_b"], eps)
                qkv = h @ b["ca_w"] + b["ca_b"]
                q, k, v = qkv.split(cpr, dim=-1)
                sh = lambda t: t.view(t.shape[0], hpr, head_dim).transpose(0, 1)
                q, k, v = sh(q), sh(k), sh(v)
                cache["k"] = k if cache["k"] is None else torch.cat([cache["k"], k], 1)
                cache["v"] = v if cache["v"] is None else torch.cat([cache["v"], v], 1)
                scores = q @ cache["k"].transpose(-2, -1) / math.sqrt(head_dim)
                scores = scores + causal_mask(x.shape[0], cache["k"].shape[1],
                                              offset, scores.device, scores.dtype)
                ctx = (torch.softmax(scores, -1) @ cache["v"]).transpose(0, 1).reshape(
                    x.shape[0], cpr)
                # Solo keeps this on the GPU. Staging it through CPU with no
                # peer to reduce with would make the single-host baseline slower
                # than it is, and flatter any TP comparison against it.
                part = ctx @ b["cp_w"]
                if stage_cpu:
                    part = part.to("cpu").contiguous()
                    collective.all_reduce(part, op=dist.ReduceOp.SUM)
                    part = part.to(device)
                x = x + (part + b["cp_b"])
                h = F.layer_norm(x, (dim,), b["ln2_w"], b["ln2_b"], eps)
                part = gelu_new(h @ b["fc_w"] + b["fc_b"]) @ b["mp_w"]
                if stage_cpu:
                    part = part.to("cpu").contiguous()
                    collective.all_reduce(part, op=dist.ReduceOp.SUM)
                    part = part.to(device)
                return x + (part + b["mp_b"])

            def forward(ids, caches, offset):
                pos = torch.arange(offset, offset + len(ids), device=device)
                x = wte_d[torch.tensor(ids, device=device)] + wpe_d[pos]
                for b, c in zip(blocks, caches):
                    x = block_forward(x, b, c, offset)
                x = F.layer_norm(x, (dim,), lnf_w, lnf_b, eps)
                return (x @ wte_d.T).to("cpu")

            def generate(n_new, stamps=None):
                caches = [{"k": None, "v": None} for _ in range(n_layer)]
                logits = forward(prompt_ids, caches, 0)
                offset = len(prompt_ids)
                out = []
                for i in range(n_new):
                    choice = torch.tensor([float(torch.argmax(logits[-1]))])
                    collective.broadcast(choice, src=0)
                    nxt = int(choice.item())
                    out.append(nxt)
                    if stamps is not None:
                        # Stamp WHEN THE TOKEN IS AVAILABLE, not after the next
                        # forward: stamps[0] is then exactly TTFT, and
                        # stamps[-1] - stamps[0] spans len(out)-1 tokens.
                        sync(device)
                        stamps.append(time.perf_counter())
                    if nxt == EOS:
                        break
                    if i < n_new - 1:
                        logits = forward([nxt], caches, offset)
                        offset += 1
                return out

            # ---------- correctness gate, BEFORE any timing ----------
            ref_model = GPT2LMHeadModel.from_pretrained(
                MODEL_DIR, dtype=torch.float32, attn_implementation="eager").eval()
            ref_ids, ids_ref = [], list(prompt_ids)
            past = None
            for _ in range(args.new_tokens):          # cached reference, not recompute
                o = ref_model(input_ids=torch.tensor([ids_ref if past is None
                                                      else [ids_ref[-1]]]),
                              past_key_values=past, use_cache=True)
                past = o.past_key_values
                nxt = int(torch.argmax(o.logits[0, -1]))
                ref_ids.append(nxt)
                if nxt == EOS:
                    break
                ids_ref.append(nxt)
            del ref_model
            got = generate(args.new_tokens)
            correct = got == ref_ids

            def agree(flag):
                """Both ranks must reach the same verdict, or neither proceeds.
                Otherwise one rank skips the collectives the other is waiting in."""
                t = torch.tensor([1.0 if flag else 0.0])
                collective.all_reduce(t, op=dist.ReduceOp.SUM)
                return int(t.item()) == world

            def barrier():
                """Sync point OUTSIDE the timed region, so one rank's reference
                work or warmup is not billed to the other rank's measurement."""
                collective.all_reduce(torch.tensor([0.0]), op=dist.ReduceOp.SUM)

            both_correct = agree(correct)
            globally_valid = False

            # ---------- timed region ----------
            runs = []
            invalid_runs = []
            timed_mismatch = False
            if both_correct:
                for _ in range(args.warmups):
                    generate(args.new_tokens)
                barrier()
                for _ in range(args.runs):
                    stamps = []
                    barrier()
                    sync(device)
                    t0 = time.perf_counter()
                    produced = generate(args.new_tokens, stamps)
                    sync(device)
                    t1 = time.perf_counter()
                    # Check THIS run's output, not just the untimed one earlier:
                    # a timed run could diverge and go unnoticed.
                    if rank == 1 and len(runs) == 0:
                        produced = produced[:-1] + [123]   # INJECTED: rank 1 only
                    run_ok = produced == ref_ids
                    if not run_ok:
                        timed_mismatch = True
                        if len(produced) != len(ref_ids):
                            why = "length %d != reference %d" % (len(produced), len(ref_ids))
                        else:
                            why = "first divergence at step %d" % next(
                                i for i, (a, b) in enumerate(zip(produced, ref_ids)) if a != b)
                        invalid_runs.append({"run_index": len(runs), "reason": why})
                    ttft = stamps[0] - t0          # first token available
                    decode_n = len(produced) - 1   # tokens after the first
                    decode_s = stamps[-1] - stamps[0]
                    assert len(stamps) == len(produced), (len(stamps), len(produced))
                    runs.append({
                        "output_matches_reference": run_ok,
                        "generated_count": len(produced),
                        "eos_emitted": bool(produced and produced[-1] == EOS),
                        "total_wall_s": t1 - t0, "ttft_s": ttft,
                        "decode_tokens": decode_n, "decode_elapsed_s": decode_s,
                        "decode_tokens_per_s": (decode_n / decode_s) if decode_s > 0 else None,
                        "step_timestamps_rel_s": [s - t0 for s in stamps]})
                barrier()
            # GLOBAL verdict. The pre-timing gate cannot cover results that did
            # not exist yet: without this, rank 0 reports success while rank 1
            # alone saw a timed divergence.
            globally_valid = bool(agree(both_correct and not timed_mismatch) and runs)
        report = {
            "mode": args.mode, "rank": rank, "world": world,
            "host": platform.node(), "device": str(device), "requested": args.device,
            "torch": torch.__version__, "hip": torch.version.hip,
            "libraries": {"transformers": _tf.__version__, "tokenizers": _tk.__version__},
            "revision": REVISION,
            "weights_sha256_prefix": hashlib.sha256(
                open(os.path.join(MODEL_DIR, "model.safetensors"), 'rb').read()
            ).hexdigest()[:16],
            "prompt_tokens": args.prompt_tokens, "new_tokens": args.new_tokens,
            "warmups": args.warmups, "runs": args.runs, "batch": 1,
            "torch_threads": torch.get_num_threads(),
            "config": config_label, "cpu_staged": stage_cpu,
            "dtype": "float32",
            "correctness_gate_passed": correct,
            "both_ranks_correct": both_correct,
            "all_timed_runs_matched_reference": not timed_mismatch,
            "this_rank_invalid_run_count": len(invalid_runs),
            "this_rank_invalid_run_reasons": invalid_runs,
            "valid": globally_valid,
            "validity_note": ("valid is the GLOBAL verdict: a timed divergence on "
                              "either rank invalidates both reports and both exit "
                              "codes. Per-rank invalid-run diagnostics are kept even "
                              "when timings are withheld."),
            "scope": ("Compares THIS GPT-2 adapter, GPU-resident solo against "
                      "two-host TP. Not a comparison against the fastest available "
                      "inference engine, and no such claim follows from it."),
            "generated_ids": got, "reference_ids": ref_ids,
            "reference_count": len(ref_ids),
            "tokenizer_sha256_prefix": hashlib.sha256(
                open(os.path.join(MODEL_DIR, "tokenizer.json"), 'rb').read()).hexdigest()[:16],
            "config_sha256_prefix": hashlib.sha256(
                open(os.path.join(MODEL_DIR, "config.json"), 'rb').read()).hexdigest()[:16],
            "timings": runs if globally_valid else [],
            "coordinator_statistic": ("rank 0 total_wall_s; per-rank raw timings "
                                      "are kept in ranks[] and not averaged"),
            "note": ("Timings are only reported when the correctness gate passes. "
                     "Load, tokenize, hash and the reference check are outside every "
                     "timed region. The reference uses its own KV cache."),
        }
        if args.mode == "tp":
            allr = [None] * world
            collective.all_gather_object(allr, report)
            report = {"ranks": allr}
        if rank == 0:
            text = json.dumps(report, indent=2)
            print(text, flush=True)
            if args.out:
                open(args.out, "w").write(text)
        if not globally_valid:
            raise SystemExit(1)
    finally:
        collective.destroy_process_group()


if __name__ == "__main__":
    main()
