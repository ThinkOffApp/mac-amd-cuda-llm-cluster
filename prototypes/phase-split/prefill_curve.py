"""Does the phase-split ceiling move with prompt length?

@claudeMB's ceiling -- TTFT cannot fall below producer_prefill + save + restore +
prefill(token N) -- is correct, but it is a ceiling AT ONE PROMPT LENGTH. Which way it
moves matters for every decision downstream of it, and the two terms scale differently:

    KV bytes      O(n)      -- one entry per token, always
    prefill time  O(n) MLP + O(n^2) attention

If prefill really is superlinear here, the split gets BETTER on longer prompts, which
would explain why Volatile Markets saw a large win at 81k tokens where we see 3.7x at
2107. If it is linear at these lengths, the speedup is scale-invariant and nobody should
expect long prompts to rescue it.

Mini only, on the already-running 27B server. No GPU window, no second machine.
"""
import json, subprocess, sys, time

BASE = "http://127.0.0.1:8091"

def post(path, payload, timeout="600"):
    o = subprocess.run(["curl", "-sS", "--max-time", timeout, "-X", "POST", BASE + path,
                        "-H", "Content-Type: application/json", "-d", json.dumps(payload)],
                       capture_output=True, text=True)
    if o.returncode:
        sys.exit(f"curl failed on {path}: {o.stderr.strip()}")
    try:
        return json.loads(o.stdout)
    except json.JSONDecodeError:
        sys.exit(f"non-JSON from {path}: {o.stdout[:300]}")

# Reuse the same token stream the 2107-token run used, truncated, so the content is
# identical across lengths and only the length varies.
base_tokens = json.load(open("/tmp/q_full.json"))["prompt"]
print(f"source prompt: {len(base_tokens)} tokens\n")

LENGTHS = [256, 512, 1024, 2048, 3072]
rows = []
for n in LENGTHS:
    if n > len(base_tokens):
        # Repeat rather than skip: a shorter curve cannot show curvature.
        toks = (base_tokens * (n // len(base_tokens) + 1))[:n]
    else:
        toks = base_tokens[:n]

    # A cached prefix would make the next row look free, so clear the slot first.
    post("/slots/0?action=erase", {})
    r = post("/completion", {"prompt": toks, "n_predict": 1, "cache_prompt": False,
                             "temperature": 0.0})
    t = r.get("timings", {})
    pms = t.get("prompt_ms")
    pn = t.get("prompt_n")
    if pms is None:
        sys.exit(f"no timings at n={n}: {json.dumps(r)[:300]}")

    s = post(f"/slots/0?action=save", {"filename": f"curve{n}.bin"})
    kv = s.get("n_written") or s.get("n_read") or 0
    rows.append((pn, pms, kv))
    print(f"n={pn:>5}  prefill={pms:>9.1f} ms  {pn/(pms/1000):>7.1f} tok/s  "
          f"KV={kv/1e6:>7.1f} MB  {kv/pn if pn else 0:>8.0f} B/tok")

print(f"\n{'from':>6} {'to':>6} {'length x':>9} {'prefill x':>10} {'KV x':>7}  exponent")
for (n0, t0, k0), (n1, t1, k1) in zip(rows, rows[1:]):
    import math
    rl, rt, rk = n1/n0, t1/t0, (k1/k0 if k0 else float('nan'))
    # log-log slope: 1.0 = linear, 2.0 = quadratic
    exp = math.log(rt)/math.log(rl)
    print(f"{n0:>6} {n1:>6} {rl:>9.2f} {rt:>10.2f} {rk:>7.2f}  {exp:>6.2f}")
print("\nexponent 1.0 = linear (split is scale-invariant);"
      " >1 = longer prompts favour the split")
