"""Pipeline-parallel GPT-2 across two hosts, with micro-batches genuinely in flight.

THE QUESTION THIS EXISTS TO ANSWER
    A llama.cpp RPC layer split makes the two machines take turns on the same
    token: while A computes, B waits. Batching makes each turn bigger, never
    simultaneous, so each box is busy at best half the wall clock — which is
    why no split ratio measured today beat the faster machine alone.

    The standard fix is micro-batch pipelining: cut the batch into chunks so B
    works chunk 1's upper layers while A already runs chunk 2's lower layers.
    vLLM's pipeline parallel does this; llama.cpp's RPC does not, and stock
    llama.cpp cannot even be made to measure it, because the knob needed IS the
    missing feature (`-ub` never splits a decode step).

    So this measures it directly, on the same Mac<->Linux pair, with real GPT-2
    weights: does overlap beat alternation, and by how much.

WHY THE CHUNKS ARE SEQUENCES, NOT TOKENS
    Sequences in a decode batch are independent — sequence i's next token needs
    only sequence i's own KV. So splitting the batch by sequence needs no
    cross-chunk dependency inside a step, and the pipelined and alternating
    runs compute exactly the same arithmetic in a different order. That is what
    makes "identical output tokens" a usable correctness gate rather than an
    approximation.

    There is still a barrier at each step boundary: step t+1 cannot start until
    stage 1 has returned step t's tokens for every chunk. With C chunks the
    bubble is about one chunk wide, so the saving should grow with C and then
    flatten. Measuring that curve is the point.

NOT A SERVING ENGINE
    FP32, one model, no batching across steps, weights loaded on both hosts.
    It answers a go/no-go question about the transport, nothing more.
"""

import argparse
import hashlib
import json
import math
import os
import socket
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = os.environ.get("GPT2_DIR", "/tmp/het-tp/gpt2")
EOS = 50256
MAX_PAYLOAD = 64 * 1024 * 1024
# Distinct prompts on purpose: a batch of identical sequences would make a
# chunking bug invisible, because every chunk would produce the same tokens.
PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists discovered",
    "def add(a, b):",
    "The history of the Roman empire begins",
    "She opened the door and found",
    "Water boils at a temperature of",
    "The best way to learn a language is",
    "Once upon a time there was",
]


def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def causal_mask(q_len, k_len, offset, device, dtype):
    q_pos = torch.arange(q_len, device=device).unsqueeze(1) + offset
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)
    return torch.zeros(q_len, k_len, device=device, dtype=dtype).masked_fill(
        k_pos > q_pos, float('-inf'))


def read_exact(sock, size):
    chunks = bytearray()
    while len(chunks) < size:
        part = sock.recv(size - len(chunks))
        if not part:
            raise ConnectionError('Peer closed before frame completed')
        chunks.extend(part)
    return bytes(chunks)


class PipeChannel:
    """One-way-at-a-time framed point-to-point channel over a TCP socket.

    DELIBERATELY NOT tcp_collectives.TCPCollectives: that class enforces a
    single shared sequence number across both peers, which assumes they perform
    the same collective in lockstep. Pipelining is the opposite — stage 0 sends
    C frames while stage 1 interleaves recv and send — so a shared counter
    would reject the very pattern under test. TCP already gives each direction
    an independent FIFO; the tag below is a correctness check on top of that,
    not ordering.
    """

    def __init__(self, rank, host, port, timeout=120):
        self.rank = rank
        if rank == 0:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.settimeout(timeout)
                listener.bind((host, port))
                listener.listen(1)
                self.sock, _ = listener.accept()
        else:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    self.sock = socket.create_connection(
                        (host, port), timeout=max(.1, deadline - time.monotonic()))
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.1)
        self.sock.settimeout(timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.bytes_sent = 0

    def send(self, tag, tensor):
        payload = tensor.detach().cpu().contiguous().numpy().astype('<f4', copy=False).tobytes()
        header = json.dumps({"tag": tag, "shape": list(tensor.shape),
                             "nbytes": len(payload)}).encode()
        if len(payload) > MAX_PAYLOAD:
            raise ValueError('frame exceeds transport limit')
        self.sock.sendall(struct.pack('!I', len(header)) + header + payload)
        self.bytes_sent += len(payload)

    def recv(self, tag):
        length, = struct.unpack('!I', read_exact(self.sock, 4))
        header = json.loads(read_exact(self.sock, length))
        if header.get("tag") != tag:
            raise ValueError(f"expected frame {tag!r}, got {header.get('tag')!r}")
        shape = tuple(header["shape"])
        body = read_exact(self.sock, header["nbytes"])
        return torch.from_numpy(
            np.frombuffer(body, dtype='<f4').astype(np.float32, copy=True).reshape(shape))

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def load_stage(sd, layers, dim, device):
    """Weights for the layers this rank owns. Each rank holds WHOLE layers, so
    there is no sharding here and no collective inside a block — the only thing
    that crosses the wire is the hidden state at the stage boundary."""
    blocks = []
    for i in layers:
        p = f"h.{i}."
        blocks.append({
            "ln1_w": sd[p + "ln_1.weight"].to(device), "ln1_b": sd[p + "ln_1.bias"].to(device),
            "ca_w": sd[p + "attn.c_attn.weight"].to(device),
            "ca_b": sd[p + "attn.c_attn.bias"].to(device),
            "cp_w": sd[p + "attn.c_proj.weight"].to(device),
            "cp_b": sd[p + "attn.c_proj.bias"].to(device),
            "ln2_w": sd[p + "ln_2.weight"].to(device), "ln2_b": sd[p + "ln_2.bias"].to(device),
            "fc_w": sd[p + "mlp.c_fc.weight"].to(device), "fc_b": sd[p + "mlp.c_fc.bias"].to(device),
            "mp_w": sd[p + "mlp.c_proj.weight"].to(device),
            "mp_b": sd[p + "mlp.c_proj.bias"].to(device),
        })
    return blocks


def block_forward(x, b, cache, offset, n_head, head_dim, dim, eps):
    """One unsharded GPT-2 block over a batch. x: (B, T, dim).

    Same arithmetic as the tensor-parallel harness that produced token-identical
    output against HuggingFace this morning, with the sharding and the
    all_reduce removed — in a pipeline each rank owns the whole layer.
    """
    Bx, T = x.shape[0], x.shape[1]
    h = F.layer_norm(x, (dim,), b["ln1_w"], b["ln1_b"], eps)
    q, k, v = (h @ b["ca_w"] + b["ca_b"]).split(dim, dim=-1)
    shape = lambda t: t.view(Bx, T, n_head, head_dim).transpose(1, 2)
    q, k, v = shape(q), shape(k), shape(v)
    cache["k"] = k if cache["k"] is None else torch.cat([cache["k"], k], 2)
    cache["v"] = v if cache["v"] is None else torch.cat([cache["v"], v], 2)
    want = (Bx, n_head, offset + T, head_dim)
    assert cache["k"].shape == want, f"K cache {tuple(cache['k'].shape)} != {want}"
    scores = q @ cache["k"].transpose(-2, -1) / math.sqrt(head_dim)
    scores = scores + causal_mask(T, cache["k"].shape[2], offset, scores.device, scores.dtype)
    ctx = (torch.softmax(scores, -1) @ cache["v"]).transpose(1, 2).reshape(Bx, T, dim)
    x = x + (ctx @ b["cp_w"] + b["cp_b"])
    h = F.layer_norm(x, (dim,), b["ln2_w"], b["ln2_b"], eps)
    return x + (gelu_new(h @ b["fc_w"] + b["fc_b"]) @ b["mp_w"] + b["mp_b"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.environ.get("TP_DEVICE", "cpu"),
                    choices=["cpu", "mps", "cuda", "rocm"])
    ap.add_argument("--batch", type=int, default=8, help="sequences decoded together")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--chunks", type=int, default=1,
                    help="micro-batches in flight per step; 1 = alternating (the baseline)")
    ap.add_argument("--split", type=int, default=6,
                    help="layers on stage 0; the rest go to stage 1")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.batch % args.chunks:
        ap.error(f"--batch {args.batch} must divide by --chunks {args.chunks}; "
                 "an uneven split would change the arithmetic between runs and "
                 "make the identical-tokens gate meaningless")
    if args.batch > len(PROMPTS):
        ap.error(f"--batch must be <= {len(PROMPTS)} distinct prompts")
    if args.device == "mps" and not torch.backends.mps.is_available():
        ap.error("MPS requested but unavailable")
    if args.device in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            ap.error("GPU requested but unavailable")
        if bool(torch.version.hip) != (args.device == "rocm"):
            ap.error("requested GPU vendor does not match this torch build")
    device = torch.device("cuda" if args.device == "rocm" else args.device)
    torch.set_num_threads(1)

    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    n_layer, n_head, dim = cfg["n_layer"], cfg["n_head"], cfg["n_embd"]
    head_dim, eps = dim // n_head, cfg["layer_norm_epsilon"]
    if not 0 < args.split < n_layer:
        ap.error(f"--split must be between 1 and {n_layer - 1}")
    sd = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    rank = int(os.environ["RANK"])
    host = os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = int(os.environ.get("MASTER_PORT", "29901"))
    chan = PipeChannel(rank, host, port)

    stage0_layers = list(range(args.split))
    stage1_layers = list(range(args.split, n_layer))
    mine = stage0_layers if rank == 0 else stage1_layers
    blocks = load_stage(sd, mine, dim, device)
    if rank == 0:
        wte = sd["wte.weight"].to(device)
        wpe = sd["wpe.weight"].to(device)
    else:
        wte = sd["wte.weight"].to(device)         # tied LM head lives with stage 1
        lnf_w = sd["ln_f.weight"].to(device)
        lnf_b = sd["ln_f.bias"].to(device)

    # Tokenise identically on both ranks: only rank 0 embeds, but rank 1 needs
    # the prompt lengths to keep its cache offsets aligned.
    prompts = [tok(p, return_tensors=None)["input_ids"] for p in PROMPTS[:args.batch]]
    plen = min(len(p) for p in prompts)
    ids = torch.tensor([p[:plen] for p in prompts], dtype=torch.long)

    caches = [{"k": None, "v": None} for _ in mine]
    busy = 0.0

    def stage_forward(x, cache_slice, offset):
        for b, c in zip(blocks, cache_slice):
            x = block_forward(x, b, c, offset, n_head, head_dim, dim, eps)
        return x

    with torch.inference_mode():
        # ---- prefill: whole batch, never chunked. Chunking prefill would change
        # ---- a second variable at the same time as the one under test.
        t0 = time.perf_counter()
        if rank == 0:
            pos = torch.arange(0, plen, device=device)
            x = wte[ids.to(device)] + wpe[pos]
            x = stage_forward(x, caches, 0)
            chan.send("prefill", x)
            nxt = chan.recv("tokens").to(torch.long)
        else:
            x = chan.recv("prefill").to(device)
            x = stage_forward(x, caches, 0)
            h = F.layer_norm(x, (dim,), lnf_w, lnf_b, eps)
            logits = h[:, -1, :] @ wte.T
            nxt = logits.argmax(-1).to("cpu")
            chan.send("tokens", nxt.to(torch.float32))
        busy += time.perf_counter() - t0

        generated = [[int(t)] for t in nxt.reshape(-1)]
        offset = plen
        chunk_size = args.batch // args.chunks
        # PER-CHUNK caches, split once after the unchunked prefill. The first
        # version sliced one batch-wide cache each step and wrote the result
        # back, which cannot work: the block appends a token, so the result is
        # longer than the slice it came from. Giving each chunk its own cache
        # removes the write-back entirely and keeps the chunks independent,
        # which is the property the whole experiment rests on.
        chunk_caches = [
            [{"k": c["k"][i * chunk_size:(i + 1) * chunk_size],
              "v": c["v"][i * chunk_size:(i + 1) * chunk_size]} for c in caches]
            for i in range(args.chunks)
        ]
        wall0 = time.perf_counter()
        compute = 0.0

        for _step in range(args.new_tokens - 1):
            cur = nxt.reshape(-1)
            if rank == 0:
                # Send every chunk BEFORE waiting for any reply: that is what
                # puts passes in flight. Stage 1 is already working on chunk 0
                # while this loop computes chunk 1.
                for c in range(args.chunks):
                    lo, hi = c * chunk_size, (c + 1) * chunk_size
                    t = time.perf_counter()
                    pos = torch.arange(offset, offset + 1, device=device)
                    x = wte[cur[lo:hi].to(device)].unsqueeze(1) + wpe[pos]
                    x = stage_forward(x, chunk_caches[c], offset)
                    compute += time.perf_counter() - t
                    chan.send(f"h{c}", x)
                outs = []
                for c in range(args.chunks):
                    outs.append(chan.recv(f"t{c}"))
                nxt = torch.cat(outs).to(torch.long)
            else:
                outs = []
                for c in range(args.chunks):
                    x = chan.recv(f"h{c}").to(device)
                    t = time.perf_counter()
                    x = stage_forward(x, chunk_caches[c], offset)
                    h = F.layer_norm(x, (dim,), lnf_w, lnf_b, eps)
                    tokens = (h[:, -1, :] @ wte.T).argmax(-1).to("cpu")
                    compute += time.perf_counter() - t
                    chan.send(f"t{c}", tokens.to(torch.float32))
                    outs.append(tokens)
                nxt = torch.cat(outs).to(torch.long)
            for i, t in enumerate(nxt.reshape(-1).tolist()):
                generated[i].append(int(t))
            offset += 1

        wall = time.perf_counter() - wall0

    tokens_out = args.batch * (args.new_tokens - 1)
    report = {
        "rank": rank, "device": str(device), "chunks": args.chunks,
        "split_layers": [len(stage0_layers), len(stage1_layers)],
        "batch": args.batch, "new_tokens": args.new_tokens,
        "decode_wall_s": round(wall, 4),
        "decode_tokens_per_s": round(tokens_out / wall, 2),
        "this_rank_compute_s": round(compute, 4),
        "this_rank_busy_fraction": round(compute / wall, 4),
        "bytes_sent": chan.bytes_sent,
        "sequences": [tok.decode(g) for g in generated] if rank == 0 else None,
        "token_ids_sha": hashlib.sha256(
            json.dumps(generated).encode()).hexdigest()[:16],
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    chan.close()


if __name__ == "__main__":
    main()
