"""Is a cross-backend KV difference the BACKEND, or just a different chunk boundary?

WHY THERE IS A .bak BESIDE THIS FILE
    The first version of this control was void, and @codexmb found it in the raw log
    I had committed. It saved a 2107-token cache and then sent back the SAME 2107
    tokens. That triggers llama.cpp's rewind path: the server re-prefilled all 2107
    tokens (tasks 158 and 166, 38.6 s and 39.2 s in receipts/decoder-server.log),
    discarded the restored history, and built a fresh native cache on both sides.
    Two fresh native prefills of course agree exactly. The 0.000000 it printed was
    真 nothing about chunking.

    The failed control and its log are kept deliberately. A discarded experiment that
    looked like a result is the most useful thing to be able to re-read.

WHAT THIS VERSION DOES DIFFERENTLY
    Saves the first N-1 tokens and requests all N, so exactly one token is evaluated
    against the restored history -- the same N-1 boundary the real split uses. It
    then ASSERTS cache_n == N-1 and prompt_n == 1 before comparing anything, so the
    re-prefill path cannot masquerade as agreement a second time.

WHAT IT STILL DOES NOT DO
    The two feeding patterns here are an arbitrary chunk plan, not the producer's
    actual chunk boundaries. Matching those requires the producer, and therefore the
    GPU window. This narrows the confound; it does not eliminate it.
"""
import json
import math
import subprocess
import sys

BASE = "http://127.0.0.1:8091"
MARK = "\n__S__:"


def req(path, payload):
    o = subprocess.run(["curl", "-sS", "--max-time", "600", "-X", "POST", BASE + path,
                        "-H", "Content-Type: application/json",
                        "-w", MARK + "%{http_code}", "-d", json.dumps(payload)],
                       capture_output=True, text=True)
    if o.returncode:
        sys.exit(f"curl exit {o.returncode} on {path}: {o.stderr[:200]}")
    body, _, st = o.stdout.rpartition(MARK)
    if not 200 <= int(st) < 300:
        sys.exit(f"{path}: HTTP {st}: {body[:300]}")
    d = json.loads(body)
    if isinstance(d, dict) and "error" in d:
        sys.exit(f"{path}: {d['error']}")
    return d


toks = json.load(open("/tmp/q_full.json"))["prompt"]
N = 2107
prompt, prefix = toks[:N], toks[:N - 1]
print(f"prompt {N} tokens; both caches hold the first {N-1} and are built on THIS machine\n")


def save_as(name, feed):
    req("/slots/0?action=erase", {})
    for part in feed:
        r = req("/completion", {"prompt": part, "n_predict": 1, "cache_prompt": True,
                                "temperature": 0.0})
        t = r.get("timings", {})
        print(f"  fed {len(part):>5} tok -> prompt_n={t.get('prompt_n')} "
              f"cache_n={t.get('cache_n')} prompt_ms={t.get('prompt_ms'):.1f}")
    s = req("/slots/0?action=save", {"filename": name})
    n_saved = s.get("n_saved", s.get("n_restored"))
    print(f"  saved {name}: {s.get('n_written', s.get('n_read')):,} bytes\n")


print(f"A: one request for all {N-1} tokens")
save_as("chunkA.bin", [prefix])
print(f"B: 1107 tokens, then extended to {N-1}")
save_as("chunkB.bin", [prefix[:1107], prefix])


def steps(slotfile):
    r = req("/slots/0?action=restore", {"filename": slotfile})
    if r.get("n_restored") != N - 1:
        sys.exit(f"{slotfile}: restored {r.get('n_restored')}, expected {N-1}")
    c = req("/completion", {"prompt": prompt, "n_predict": 4, "cache_prompt": True,
                            "temperature": 0.0, "top_k": 0, "top_p": 1.0, "n_probs": 10})
    t = c.get("timings", {})
    if t.get("cache_n") != N - 1 or t.get("prompt_n") != 1:
        sys.exit(f"{slotfile}: cache_n={t.get('cache_n')} prompt_n={t.get('prompt_n')}; "
                 f"expected {N-1} and 1. The restored cache was not used -- VOID, "
                 "exactly like the first version of this control.")
    print(f"  {slotfile}: cache_n={t['cache_n']} prompt_n={t['prompt_n']}  (cache hit confirmed)")
    ps = c.get("completion_probabilities")
    if not ps:
        sys.exit(f"{slotfile}: no completion_probabilities")
    out = []
    for i, s in enumerate(ps):
        es = s.get("top_logprobs") or s.get("probs")
        if not es:
            sys.exit(f"{slotfile}: step {i} empty")
        d = {}
        for e in es:                       # keyed by token ID: distinct ids can share text
            tid, lp = e.get("id"), e.get("logprob", e.get("prob"))
            if tid is None or lp is None or not math.isfinite(lp):
                sys.exit(f"{slotfile}: step {i} bad entry {e}")
            if tid in d:
                sys.exit(f"{slotfile}: step {i} duplicate token id {tid}")
            d[tid] = (lp, e.get("token"))
        out.append(d)
    return c.get("content"), out


print("comparing:")
ca, sa = steps("chunkA.bin")
cb, sb = steps("chunkB.bin")
if len(sa) != len(sb):
    sys.exit("step count mismatch")

worst = 0.0
print(f"\n{'step':<6}{'A top':<20}{'B top':<20}{'|dlogprob|':>12}  shared")
for i, (da, db) in enumerate(zip(sa, sb)):
    sh = set(da) & set(db)
    if not sh:
        sys.exit(f"step {i}: disjoint top-10")
    d = max(abs(da[k][0] - db[k][0]) for k in sh)
    worst = max(worst, d)
    ta = max(da, key=lambda k: da[k][0])
    tb = max(db, key=lambda k: db[k][0])
    print(f"{i:<6}{repr(da[ta][1])+'#'+str(ta):<20}{repr(db[tb][1])+'#'+str(tb):<20}"
          f"{d:>12.6f}  {len(sh)}/{len(da)}")

print(f"\nsame text: {ca == cb}")
print(f"SAME BACKEND, DIFFERENT CHUNKING: worst |dlogprob| = {worst:.6f}")
print("cross-machine figure for comparison:                 0.670641")
print("\nA near-zero here narrows chunking as a confound but does not eliminate it:")
print("these are an arbitrary chunk plan, not the producer's actual boundaries.")
