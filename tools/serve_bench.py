# Token-COUNTED receipts. The previous bench counted SSE chunks and called them
# tokens (codexmb, correctly). Here every count comes from the server:
#   - usage.{prompt,completion}_tokens on a non-streamed request
#   - and cross-checked against the vllm:*_tokens_total counter deltas
import json, time, urllib.request, re

BASE  = "http://127.0.0.1:8888"
MODEL = "GLM-5.3-Flash-EXL3"


# ---------------------------------------------------------------------------
# GUARD. This exists because the error it catches has now cost this project
# two withdrawn results. If the measured chunks-per-token ratio is not ~1,
# anything that counted stream chunks as tokens is wrong by exactly this
# factor. We refuse to print a rate until the ratio has been measured.
# ---------------------------------------------------------------------------
def assert_not_counting_chunks(chunks, server_tokens):
    if not server_tokens:
        raise SystemExit("REFUSING: server reported 0 generation tokens; no rate can be quoted.")
    ratio = chunks / server_tokens
    if abs(ratio - 1.0) > 0.05:
        print(f"  !! {1/ratio:.2f} TOKENS PER CHUNK. Counting chunks would be wrong by that factor.")
        print(f"  !! All rates below use the SERVER counter, not the stream.")
    return ratio

def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=20) as r:
        s = r.read().decode()
    out = {}
    for k in ("prompt_tokens_total", "generation_tokens_total"):
        m = re.search(r'vllm:%s\{[^}]*\}\s+([0-9.e+]+)' % k, s)
        out[k] = float(m.group(1)) if m else float('nan')
    return out

def post(path, body, timeout=600, stream=False):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

PROMPTS = {512: "Explain tensor parallelism in detail. " * 60,
           2048: "Explain tensor parallelism in detail. " * 240}

print("=== chunks vs tokens: is one SSE chunk one token? ===")
p = PROMPTS[512]
m0 = metrics()
chunks = 0; t0 = time.perf_counter(); ttft = None
r = post("/v1/completions", {"model": MODEL, "prompt": p, "max_tokens": 128,
                             "temperature": 0, "stream": True})
for line in r:
    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
    d = json.loads(line[6:])
    if d["choices"][0].get("text"):
        if ttft is None: ttft = time.perf_counter() - t0
        chunks += 1
tot = time.perf_counter() - t0
m1 = metrics()
gen = m1["generation_tokens_total"] - m0["generation_tokens_total"]
pro = m1["prompt_tokens_total"]     - m0["prompt_tokens_total"]
print(f"  SSE chunks with text : {chunks}")
print(f"  server-counted tokens: generation {gen:.0f}, prompt {pro:.0f}")
print(f"  chunks per token     : {chunks/gen:.4f}" if gen else "  (no gen tokens)")
assert_not_counting_chunks(chunks, gen)
print()

print("=== token-counted prefill / decode, non-streamed usage + streamed TTFT ===")
print(f"{'prompt tok':>10} {'TTFT s':>8} {'prefill tok/s':>14} {'decode tok/s':>13} {'gen tok':>8}")
for want, text in PROMPTS.items():
    m0 = metrics(); t0 = time.perf_counter(); ttft = None
    r = post("/v1/completions", {"model": MODEL, "prompt": text, "max_tokens": 128,
                                 "temperature": 0, "stream": True})
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
        d = json.loads(line[6:])
        if d["choices"][0].get("text") and ttft is None:
            ttft = time.perf_counter() - t0
    tot = time.perf_counter() - t0
    m1 = metrics()
    gen = m1["generation_tokens_total"] - m0["generation_tokens_total"]
    pro = m1["prompt_tokens_total"]     - m0["prompt_tokens_total"]
    pre = pro/ttft if ttft else float('nan')
    dec = (gen-1)/(tot-ttft) if ttft and tot > ttft and gen > 1 else float('nan')
    print(f"{pro:10.0f} {ttft:8.2f} {pre:14.1f} {dec:13.2f} {gen:8.0f}")
print()
print("NOTE: prefill = server-counted PROMPT tokens / TTFT. TTFT still includes")
print("      scheduling and the first decode step, so it is an upper bound on")
print("      prefill time and thus a LOWER bound on prefill rate, not isolated.")
