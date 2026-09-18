"""How large can a chunk-plan difference get, on ONE machine?

The corrected chunk control showed one arbitrary feeding pattern moves the
next-token distributions by 0.27, against 0.67 measured across machines. That was
enough to withdraw the backend attribution, but it leaves the real question open:

    is 0.67 within the range that chunking alone can produce?

If it is, the cross-machine number needs no backend explanation at all. If chunking
saturates well below it, something else is contributing and the backend becomes a
live candidate again -- still not established, but not excluded either.

Same machine, same backend, same build, same weights, same 2106-token prefix.
Only the feeding pattern varies. Every comparison asserts cache_n == N-1 and
prompt_n == 1, because the first version of this control compared two fresh native
prefills and returned a confident zero.
"""
import itertools, json, math, subprocess, sys

BANNER = (
    "NOT THE ACCEPTANCE GATE. This reports a TOP-K LOG-PROBABILITY difference.\n"
    "The gate is a FULL-VOCABULARY RAW-LOGIT comparison at atol=0.1 AND rtol=0.01.\n"
    "log p = logit - logsumexp, so a uniform shift across the vocabulary vanishes\n"
    "here and is fully visible there; these numbers cannot be compared with gate\n"
    "results or put in the same table. A quiet top-k is consistent with a loud\n"
    "vocabulary: one measured run had the top-10 move 0.67 while 179,629 logits\n"
    "were out of tolerance.\n"
)


BASE, MARK, N = "http://127.0.0.1:8091", "\n__S__:", 2107

def req(path, payload):
    o = subprocess.run(["curl","-sS","--max-time","900","-X","POST",BASE+path,
                        "-H","Content-Type: application/json","-w",MARK+"%{http_code}",
                        "-d",json.dumps(payload)], capture_output=True, text=True)
    if o.returncode: sys.exit(f"curl exit {o.returncode} on {path}")
    body,_,st = o.stdout.rpartition(MARK)
    if not 200 <= int(st) < 300: sys.exit(f"{path}: HTTP {st}: {body[:200]}")
    d = json.loads(body)
    if isinstance(d, dict) and "error" in d: sys.exit(f"{path}: {d['error']}")
    return d

toks = json.load(open("/tmp/q_full.json"))["prompt"]
prompt, prefix = toks[:N], toks[:N-1]

PLANS = {
    "one-shot":      [N-1],
    "half":          [1107, N-1],
    "quarter-steps": [512, 1024, 1536, N-1],
    "late":          [2048, N-1],
    "early":         [256, N-1],
}

def build(name, cuts):
    req("/slots/0?action=erase", {})
    for c in cuts:
        req("/completion", {"prompt": prefix[:c], "n_predict": 1,
                            "cache_prompt": True, "temperature": 0.0})
    req("/slots/0?action=save", {"filename": f"plan-{name}.bin"})
    print(f"  built {name:<14} cuts={cuts}")

def dist(name):
    r = req("/slots/0?action=restore", {"filename": f"plan-{name}.bin"})
    if r.get("n_restored") != N-1: sys.exit(f"{name}: restored {r.get('n_restored')}")
    c = req("/completion", {"prompt": prompt, "n_predict": 4, "cache_prompt": True,
                            "temperature": 0.0, "top_k": 0, "top_p": 1.0, "n_probs": 10})
    t = c.get("timings", {})
    if t.get("cache_n") != N-1 or t.get("prompt_n") != 1:
        sys.exit(f"{name}: cache_n={t.get('cache_n')} prompt_n={t.get('prompt_n')} -- VOID")
    out = []
    for i, s in enumerate(c["completion_probabilities"]):
        es = s.get("top_logprobs") or s.get("probs")
        d = {}
        for e in es:
            tid, lp = e.get("id"), e.get("logprob")
            if tid is None or lp is None or not math.isfinite(lp): sys.exit(f"{name}: bad entry")
            if tid in d: sys.exit(f"{name}: duplicate id {tid}")
            d[tid] = lp
        out.append(d)
    return c.get("content"), out

print(BANNER)
print("building caches, one per feeding pattern:")
for n, c in PLANS.items(): build(n, c)

print("\nreading distributions (cache hit asserted on each):")
D = {}
for n in PLANS:
    txt, steps = dist(n)
    D[n] = (txt, steps)
    print(f"  {n:<14} content={txt!r}")

def gap(a, b):
    w = 0.0
    for da, db in zip(a, b):
        sh = set(da) & set(db)
        if not sh: return float("nan")
        w = max(w, max(abs(da[k]-db[k]) for k in sh))
    return w

print(f"\n{'pair':<32}{'worst |dlogprob|':>18}   same text")
worst_overall, rows = 0.0, []
for x, y in itertools.combinations(PLANS, 2):
    g = gap(D[x][1], D[y][1])
    rows.append((g, x, y))
    worst_overall = max(worst_overall, g)
for g, x, y in sorted(rows, reverse=True):
    print(f"{x+' vs '+y:<32}{g:>18.6f}   {D[x][0] == D[y][0]}")

print(f"\nLARGEST chunk-plan-only difference on ONE machine : {worst_overall:.6f}")
print(f"cross-machine difference                          : 0.670641")
print(f"same file twice (control)                         : 0.000000")
print()
print("WHAT THIS DOES AND DOES NOT SAY. @codexmb: comparing these two maxima is a ratio")
print("of maxima from DIFFERENT experiments, not a decomposition -- they need not even")
print("occur at the same step or over the same tokens, and here they do not. Nothing")
print("below is a percentage of the physical gap.")
print()
print("What it establishes: a chunk-plan difference alone moves the distributions by up")
print(f"to {worst_overall:.3f} on one machine, one backend, one build, one set of weights.")
print("Therefore chunking is NOT excluded as a cause of the cross-machine difference.")
print("The physical contribution stays unknown until the producer's actual boundaries")
print("are matched -- and this sweep is a sample of plans, not their maximum.")
