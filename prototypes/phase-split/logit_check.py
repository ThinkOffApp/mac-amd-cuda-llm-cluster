"""Does an M5-produced KV cache give the SAME distribution as a Mini-produced one?

The two slot files are NOT byte-identical (Metal vs ROCm produce different KV),
so "the file copied intact" proves nothing about the output. Only the logits do.
Greedy alone is too weak a check: two distributions can agree on the argmax and
disagree everywhere else, which is exactly the failure a user would see later in
a long generation.
"""
import json, subprocess, sys

BASE = "http://127.0.0.1:8091"
REQ = json.load(open("/tmp/q_full.json"))

def post(path, payload):
    out = subprocess.run(
        ["curl", "-sS", "--max-time", "300", "-X", "POST", BASE + path,
         "-H", "Content-Type: application/json", "-d", json.dumps(payload)],
        capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"curl failed on {path}: {out.stderr.strip()}")
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        sys.exit(f"non-JSON from {path}: {out.stdout[:400]}")

def run(slotfile):
    r = post("/slots/0?action=restore", {"filename": slotfile})
    if "n_restored" not in json.dumps(r) and "err" in json.dumps(r).lower():
        sys.exit(f"restore of {slotfile} failed: {r}")
    body = dict(REQ)
    body.update(n_predict=4, cache_prompt=True, temperature=0.0,
                top_k=0, top_p=1.0, n_probs=10, post_sampling_probs=False)
    c = post("/completion", body)
    probs = c.get("completion_probabilities") or []
    return r, c, probs

def top(probs, i):
    # llama.cpp has used both 'probs' and 'top_logprobs' for the per-token list.
    step = probs[i]
    lst = step.get("top_logprobs") or step.get("probs") or []
    return [(e.get("token"), e.get("logprob", e.get("prob"))) for e in lst]

ra, ca, pa = run("pd.bin")   # produced on the M5 (ROCm)
rb, cb, pb = run("q27.bin")  # produced on this Mac mini (Metal)

print(f"restore pd.bin : {json.dumps(ra)[:200]}")
print(f"restore q27.bin: {json.dumps(rb)[:200]}")
print(f"\nM5-KV   content: {ca.get('content')!r}")
print(f"Mini-KV content: {cb.get('content')!r}")
print(f"tokens identical: {ca.get('content') == cb.get('content')}")

if not pa or not pb:
    print("\nNO PER-TOKEN PROBABILITIES RETURNED -- n_probs unsupported on this build;"
          " the token-identity above is all this run establishes.")
    raise SystemExit(0)

print(f"\n{'step':<5} {'M5 top token':<18} {'Mini top token':<18} "
      f"{'max |dlogprob| over the shared top-10':<12}")
worst = 0.0
for i in range(min(len(pa), len(pb))):
    ta, tb = top(pa, i), top(pb, i)
    da, db = dict(ta), dict(tb)
    shared = set(da) & set(db)
    d = max((abs(da[k] - db[k]) for k in shared if
             isinstance(da[k], (int, float)) and isinstance(db[k], (int, float))), default=float("nan"))
    worst = max(worst, 0.0 if d != d else d)
    print(f"{i:<5} {str(ta[0][0])[:16]:<18} {str(tb[0][0])[:16]:<18} {d:.6f}"
          f"   (shared {len(shared)}/10)")
print(f"\nworst logprob difference across all steps: {worst:.6f}")
