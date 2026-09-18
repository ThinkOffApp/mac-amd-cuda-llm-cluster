"""Rolling whole-layer GPT-2 pipeline, derived from claudemm 51103d9.

Experimental algorithm/correctness harness, not a larger-model performance claim.
Each chunk advances as soon as its reply arrives; no global token-step barrier.
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
        if length > 4096:
            raise ValueError("oversized header")
        header = json.loads(read_exact(self.sock, length))
        if header.get("tag") != tag:
            raise ValueError(f"expected frame {tag!r}, got {header.get('tag')!r}")
        shape = tuple(header["shape"])
        if not shape or any(not isinstance(d, int) or d <= 0 for d in shape):
            raise ValueError("invalid shape")
        if not 0 <= header["nbytes"] <= MAX_PAYLOAD or math.prod(shape)*4 != header["nbytes"]:
            raise ValueError("invalid payload length")
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
    from pathlib import Path
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, GPT2LMHeadModel
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'rocm'], default='cpu')
    ap.add_argument('--schedule', choices=['rolling', 'barrier', 'alternating'], default='rolling')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--chunks', type=int, default=2)
    ap.add_argument('--split', type=int, default=6)
    ap.add_argument('--prompt-tokens', type=int, default=12)
    ap.add_argument('--new-tokens', type=int, default=16)
    ap.add_argument('--warmups', type=int, default=2)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if not 1 <= args.batch <= 64 or not 1 <= args.chunks <= args.batch or args.batch % args.chunks:
        ap.error('positive batch <=64 must be divisible by chunks')
    if min(args.new_tokens, args.prompt_tokens, args.runs) < 1 or args.warmups < 0:
        ap.error('invalid token/run counts')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        ap.error('MPS unavailable')
    if args.device in ('cuda', 'rocm') and (not torch.cuda.is_available() or bool(torch.version.hip) != (args.device == 'rocm')):
        ap.error('requested GPU vendor unavailable')
    device = torch.device('cuda' if args.device == 'rocm' else args.device)
    torch.set_num_threads(1)
    cfg = json.loads(Path(MODEL_DIR, 'config.json').read_text())
    if cfg.get('model_type') != 'gpt2' or not 0 < args.split < cfg['n_layer']:
        ap.error('requires GPT2 and an interior layer split')
    if args.prompt_tokens + args.new_tokens > cfg['n_positions']:
        ap.error('context exceeded')
    weights = Path(MODEL_DIR, 'model.safetensors')
    with weights.open('rb') as f:
        weight_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    if weight_sha != '248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707':
        ap.error('requires pinned GPT2-124M checkpoint; larger models need a separate validated implementation')
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    prompts = []
    for i in range(args.batch):
        # Put lane identity first so short prompts remain distinct at large B.
        seed = f'{i}: ' + PROMPTS[i % len(PROMPTS)] + ' '
        ids = tok(seed * (args.prompt_tokens + 1))['input_ids'][:args.prompt_tokens]
        prompts.append(ids)
    assert len({tuple(p) for p in prompts}) == args.batch
    rank = int(os.environ['RANK'])
    assert rank in (0, 1)
    refs = []
    if rank == 0:
        reference = GPT2LMHeadModel.from_pretrained(MODEL_DIR, local_files_only=True,
                     attn_implementation='eager', dtype=torch.float32).eval()
        with torch.inference_mode():
            for prompt in prompts:
                current, cache, out_ids = torch.tensor([prompt]), None, []
                for _ in range(args.new_tokens):
                    output = reference(current, past_key_values=cache, use_cache=True)
                    choice = int(output.logits[0, -1].argmax())
                    out_ids.append(choice)
                    if choice == EOS:
                        break
                    cache, current = output.past_key_values, torch.tensor([[choice]])
                refs.append(out_ids)
        del reference
    chan = PipeChannel(rank, os.environ.get('MASTER_ADDR', '127.0.0.1'),
                       int(os.environ.get('MASTER_PORT', '29901')))
    # Reject mismatched peer weights/config before any model traffic.
    signature = hashlib.sha256(json.dumps({'sha': weight_sha, 'args': {k:v for k,v in vars(args).items()
                if k not in ('device','out')}, 'prompts': prompts}, sort_keys=True).encode()).hexdigest()
    if rank == 0:
        chan.send('config-' + signature, torch.zeros(1))
        chan.recv('config-ok')
    else:
        chan.recv('config-' + signature)
        chan.send('config-ok', torch.zeros(1))
    sd = load_file(str(weights))
    dim, nhead = cfg['n_embd'], cfg['n_head']
    layers = range(args.split) if rank == 0 else range(args.split, cfg['n_layer'])
    blocks = load_stage(sd, layers, dim, device)
    wte = sd['wte.weight'].to(device)
    if rank == 0:
        wpe = sd['wpe.weight'].to(device)
    else:
        lnf_w, lnf_b = sd['ln_f.weight'].to(device), sd['ln_f.bias'].to(device)
    del sd
    b = args.batch // args.chunks

    def forward(x, cache, offset):
        for block, kv in zip(blocks, cache):
            x = block_forward(x, block, kv, offset, nhead, dim // nhead, dim, cfg['layer_norm_epsilon'])
        return x

    def choice(x):
        h = F.layer_norm(x[:, -1], (dim,), lnf_w, lnf_b, cfg['layer_norm_epsilon'])
        return (h @ wte.T).argmax(-1).cpu()

    def generate(run_id):
        prefix = f'run{run_id}'
        cache = [{'k': None, 'v': None} for _ in blocks]
        # Ready handshake outside timing, including completion of previous run.
        if rank == 0:
            chan.send(prefix+'ready', torch.zeros(1)); chan.recv(prefix+'ready')
        else:
            chan.recv(prefix+'ready'); chan.send(prefix+'ready', torch.zeros(1))
        started = time.perf_counter()
        if rank == 0:
            inp = torch.tensor(prompts, device=device)
            x = forward(wte[inp] + wpe[torch.arange(args.prompt_tokens, device=device)], cache, 0)
            chan.send(prefix+'prefill', x)
            first = chan.recv(prefix+'tokens').long()
        else:
            x = forward(chan.recv(prefix+'prefill').to(device), cache, 0)
            first = choice(x)
            chan.send(prefix+'tokens', first.float())
        first_at = time.perf_counter()
        generated = [[int(v)] for v in first]
        finished = [g[-1] == EOS for g in generated]
        current = [first[c*b:(c+1)*b].clone() for c in range(args.chunks)]
        caches = [[{'k': kv['k'][c*b:(c+1)*b], 'v': kv['v'][c*b:(c+1)*b]} for kv in cache]
                  for c in range(args.chunks)]
        trace = []
        steps = args.new_tokens - 1

        def send_chunk(c, step):
            offset = args.prompt_tokens + step
            x = wte[current[c].to(device)].unsqueeze(1) + wpe[offset]
            x = forward(x, caches[c], offset)
            chan.send(f'{prefix}-h-{c}-{step}', x)
            trace.append(['send', c, step])

        def receive_chunk(c, step):
            result = chan.recv(f'{prefix}-t-{c}-{step}').long()
            trace.append(['recv', c, step])
            for j, v in enumerate(result.tolist()):
                i = c*b+j
                if not finished[i]:
                    generated[i].append(v)
                    finished[i] = v == EOS
                # EOS lanes keep their shape but are never counted again.
                result[j] = EOS if finished[i] else v
            current[c] = result

        if rank == 1:
            for step in range(steps):
                for c in range(args.chunks):
                    x = chan.recv(f'{prefix}-h-{c}-{step}').to(device)
                    tokens = choice(forward(x, caches[c], args.prompt_tokens+step))
                    chan.send(f'{prefix}-t-{c}-{step}', tokens.float())
        elif args.schedule == 'rolling':
            if steps:
                for c in range(args.chunks):
                    send_chunk(c, 0)
                for step in range(steps):
                    for c in range(args.chunks):
                        receive_chunk(c, step)
                        if step+1 < steps:
                            send_chunk(c, step+1)
        else:
            for step in range(steps):
                for c in range(args.chunks):
                    send_chunk(c, step)
                    if args.schedule == 'alternating':
                        receive_chunk(c, step)
                if args.schedule == 'barrier':
                    for c in range(args.chunks):
                        receive_chunk(c, step)
        ended = time.perf_counter()
        if rank == 0:
            valid = generated == refs
            chan.send(prefix+'valid', torch.tensor([float(valid)]))
            chan.recv(prefix+'validated')
            return {'valid': valid, 'produced_ids': generated, 'generated_tokens': sum(map(len, generated)),
                'decode_tokens': sum(len(g)-1 for g in generated), 'ttft_s': first_at-started,
                'decode_wall_s': ended-first_at, 'total_wall_s': ended-started, 'host_event_order': trace}
        valid = bool(chan.recv(prefix+'valid').item())
        chan.send(prefix+'validated', torch.zeros(1))
        return {'valid': valid}

    runs, checks = [], []
    with torch.inference_mode():
        for i in range(1 + args.warmups + args.runs):
            result = generate(i)
            checks.append(result['valid'])
            if i > args.warmups:
                runs.append(result)
    valid = all(checks)
    if rank == 0:
        if valid:
            for run in runs:
                run['aggregate_decode_tokens_per_s'] = run['decode_tokens']/run['decode_wall_s'] if run['decode_tokens'] else None
                run['end_to_end_tokens_per_s'] = run['generated_tokens']/run['total_wall_s']
        report = {'valid': valid, 'weight_sha256': weight_sha,
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'config': vars(args), 'prompt_ids': prompts, 'reference_ids': refs, 'all_run_checks': checks,
            'runs': runs if valid else [], 'invalid_runs': [] if valid else runs,
            'limits': ['host_event_order proves submission order, not GPU overlap',
                       'fixed computation budget retains finished EOS lanes; output/counts freeze at EOS',
                       'no GPU utilization claims; decode wall includes scheduling, transfer and compute',
                       'GPT2 algorithm gate only; not evidence for larger-model speedup']}
        Path(args.out).write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({'valid': valid, 'schedule': args.schedule, 'out': args.out}), flush=True)
    chan.close()
    raise SystemExit(0 if valid else 1)


if __name__ == '__main__':
    main()
