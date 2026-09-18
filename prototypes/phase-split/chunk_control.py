"""Is the 0.67 gap the BACKEND, or just a different prefill chunk boundary?

@codexmb's objection: a same-file control excludes run-to-run variation in one
fixture, but it cannot separate "ROCm computed this KV" from "the producer
happened to chunk the prefill differently". His own local failures moved when he
matched chunk boundaries, so the distinction is not hypothetical.

This settles it WITHOUT a second machine. Both caches below are produced on this
Mac mini, same build, same weights, same backend -- and differ only in how the
prompt was fed:

    chunkA   one request for all N tokens
    chunkB   N-1000 tokens first, then extended to N with cache_prompt

If chunkA vs chunkB is ~0, chunking is not a confound and the cross-machine 0.67
is attributable to the backend. If it is ~0.67, my attribution is unproven and I
have to say so.
"""
import json, math, subprocess, sys

BANNER = (
    "NOT THE ACCEPTANCE GATE. This reports a TOP-K LOG-PROBABILITY difference.\n"
    "The gate is a FULL-VOCABULARY RAW-LOGIT comparison at atol=0.1 AND rtol=0.01.\n"
    "log p = logit - logsumexp, so a uniform shift across the vocabulary vanishes\n"
    "here and is fully visible there; these numbers cannot be compared with gate\n"
    "results or put in the same table. A quiet top-k is consistent with a loud\n"
    "vocabulary: one measured run had the top-10 move 0.67 while 179,629 logits\n"
    "were out of tolerance.\n"
)


BASE = "http://127.0.0.1:8091"
MARK = "\n__S__:"

def req(path, payload):
    o = subprocess.run(["curl","-sS","--max-time","600","-X","POST",BASE+path,
                        "-H","Content-Type: application/json",
                        "-w",MARK+"%{http_code}","-d",json.dumps(payload)],
                       capture_output=True, text=True)
    if o.returncode: sys.exit(f"curl exit {o.returncode} on {path}: {o.stderr[:200]}")
    body,_,st = o.stdout.rpartition(MARK)
    if not 200 <= int(st) < 300: sys.exit(f"{path}: HTTP {st}: {body[:300]}")
    d = json.loads(body)
    if isinstance(d, dict) and "error" in d: sys.exit(f"{path}: {d['error']}")
    return d

toks = json.load(open("/tmp/q_full.json"))["prompt"]
N = 2107
prompt = toks[:N]
print(f"prompt {N} tokens, both caches produced on THIS machine\n")

def save_as(name, feed):
    req("/slots/0?action=erase", {})
    for part in feed:
        r = req("/completion", {"prompt": part, "n_predict": 1, "cache_prompt": True,
                                "temperature": 0.0})
        t = r.get("timings", {})
        print(f"  fed {len(part):>5} tok -> prompt_n={t.get('prompt_n')} "
              f"prompt_ms={t.get('prompt_ms'):.1f}")
    s = req("/slots/0?action=save", {"filename": name})
    print(f"  saved {name}: {s.get('n_written', s.get('n_read'))} bytes, "
          f"{s.get('n_saved','?')} tokens\n")

print("A: single request")
save_as("chunkA.bin", [prompt])
print("B: 1107 tokens, then extended to 2107")
save_as("chunkB.bin", [prompt[:1107], prompt])

def steps(slotfile):
    r = req("/slots/0?action=restore", {"filename": slotfile})
    if r.get("n_restored") != N:
        sys.exit(f"{slotfile}: restored {r.get('n_restored')}, expected {N}")
    c = req("/completion", {"prompt": prompt, "n_predict": 4, "cache_prompt": True,
                            "temperature": 0.0, "top_k": 0, "top_p": 1.0, "n_probs": 10})
    ps = c.get("completion_probabilities")
    if not ps: sys.exit(f"{slotfile}: no completion_probabilities")
    out=[]
    for i,s in enumerate(ps):
        es = s.get("top_logprobs") or s.get("probs")
        if not es: sys.exit(f"{slotfile}: step {i} empty")
        d={}
        for e in es:
            lp=e.get("logprob", e.get("prob"))
            if lp is None or not math.isfinite(lp): sys.exit(f"{slotfile}: bad logprob")
            d[e["token"]]=lp
        out.append(d)
    return c.get("content"), out

ca,sa = steps("chunkA.bin")
cb,sb = steps("chunkB.bin")
if len(sa)!=len(sb): sys.exit("step count mismatch")
worst=0.0
print(f"{'step':<6}{'A top':<16}{'B top':<16}{'|dlogprob|':>12}  shared")
for i,(da,db) in enumerate(zip(sa,sb)):
    sh=set(da)&set(db)
    if not sh: sys.exit(f"step {i}: disjoint top-10")
    d=max(abs(da[k]-db[k]) for k in sh); worst=max(worst,d)
    print(f"{i:<6}{max(da,key=da.get)[:14]:<16}{max(db,key=db.get)[:14]:<16}{d:>12.6f}  {len(sh)}/{len(da)}")
print(f"\nsame text: {ca==cb}")
print(f"SAME BACKEND, DIFFERENT CHUNKING: worst |dlogprob| = {worst:.6f}")
print(f"cross-machine figure for comparison:                 0.670641")
print("\n-> if these are comparable, 'cross-backend' is NOT established")
