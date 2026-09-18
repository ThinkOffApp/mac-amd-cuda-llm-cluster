"""Per-step comparison, because a ratio of two maxima is not a decomposition.

@codexmb's point: setting the chunk control's worst |dlogprob| against the
cross-machine one compares maxima from different experiments, taken at different
steps over different tokens. It asserts a share of the gap that was never measured.

A per-step table does not fix that -- only matched producer boundaries can -- but it
at least puts the two effects on the same axis, step by step, so a reader can see
where each one is large instead of comparing two single numbers.

Reads the caches both experiments left on disk. Asserts the cache hit on every read.
"""
import json, math, subprocess, sys

BASE, MARK, N = "http://127.0.0.1:8091", "\n__S__:", 2107

def req(path, payload):
    o = subprocess.run(["curl","-sS","--max-time","900","-X","POST",BASE+path,
                        "-H","Content-Type: application/json","-w",MARK+"%{http_code}",
                        "-d",json.dumps(payload)], capture_output=True, text=True)
    if o.returncode: sys.exit(f"curl exit {o.returncode} on {path}")
    body,_,st = o.stdout.rpartition(MARK)
    if not 200 <= int(st) < 300: sys.exit(f"{path}: HTTP {st}: {body[:200]}")
    return json.loads(body)

toks = json.load(open("/tmp/q_full.json"))["prompt"]

def dist(slotfile, expect, prompt):
    r = req("/slots/0?action=restore", {"filename": slotfile})
    if r.get("n_restored") != expect:
        sys.exit(f"{slotfile}: restored {r.get('n_restored')}, expected {expect}")
    c = req("/completion", {"prompt": prompt, "n_predict": 4, "cache_prompt": True,
                            "temperature": 0.0, "top_k": 0, "top_p": 1.0, "n_probs": 10})
    t = c.get("timings", {})
    if t.get("cache_n") != expect or t.get("prompt_n") != 1:
        sys.exit(f"{slotfile}: cache_n={t.get('cache_n')} prompt_n={t.get('prompt_n')} -- VOID")
    out = []
    for s in c["completion_probabilities"]:
        es = s.get("top_logprobs") or s.get("probs")
        d = {}
        for e in es:
            tid, lp = e.get("id"), e.get("logprob")
            if tid is None or lp is None or not math.isfinite(lp): sys.exit("bad entry")
            d[tid] = lp
        out.append(d)
    return out

def per_step(a, b):
    out = []
    for da, db in zip(a, b):
        sh = set(da) & set(db)
        out.append(max(abs(da[k]-db[k]) for k in sh) if sh else float("nan"))
    return out

# Cross-machine pair: caches hold 2107, prompt is 2108 tokens.
xm = per_step(dist("pd.bin", 2107, toks[:2108]), dist("q27.bin", 2107, toks[:2108]))
# Chunk-plan pair: caches hold 2106, prompt is 2107.
ck = per_step(dist("plan-one-shot.bin", 2106, toks[:2107]),
              dist("plan-half.bin", 2106, toks[:2107]))

print("Two DIFFERENT experiments -- different caches, different prompt lengths, and")
print("therefore different generated tokens. They are shown on one axis to make the")
print("shapes comparable, NOT to license a ratio between them.\n")
print(f"{'step':<6}{'cross-machine':>16}{'chunk plan':>14}")
for i, (x, c) in enumerate(zip(xm, ck)):
    print(f"{i:<6}{x:>16.6f}{c:>14.6f}")
print(f"{'max':<6}{max(xm):>16.6f}{max(ck):>14.6f}")
print("\nThe maxima fall on different steps, which is exactly why quoting one over the")
print("other as a percentage was wrong. Attributing any share of the cross-machine")
print("number to chunking needs the producer's real boundaries matched.")
